from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from src.agents.action_space import DEFAULT_BIT_CHOICES, build_action_bits
from src.agents.actor import SharedActor, decode_actions
from src.agents.policy_learner import PolicyLearningConfig, ReinforcePolicyLearner
from src.agents.reward import RewardConfig, RewardEvaluator
from src.agents.state_builder import AgentStateBuilder, summarize_agent_states
from src.datasets.cifar import CIFAR_STATS, build_cifar_loaders
from src.models.resnet_cifar_qat import build_cifar_resnet_qat
from src.quantization.bitops import collect_resource_stats
from src.quantization.lsq import clamp_lsq_scales
from src.quantization.policy_applier import apply_block_policy, set_uniform_bit_widths
from src.training.train_fp32 import build_optimizer, build_scheduler, evaluate, load_config, resolve_device, train_one_epoch
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import save_json
from src.utils.metrics import AverageMeter, accuracy
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive multi-agent QAT.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--fp32-checkpoint", type=str, default=None, help="Optional FP32 checkpoint to warm start from.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Build model, actor, and states without training.")
    return parser.parse_args()


def apply_first_last_bits(model: nn.Module, first_last_bits: int | None) -> None:
    if first_last_bits is None:
        return
    if hasattr(model, "conv1"):
        model.conv1.set_bits(int(first_last_bits), int(first_last_bits))
    if hasattr(model, "fc"):
        model.fc.set_bits(int(first_last_bits), int(first_last_bits))


def applied_policy_to_dict(
    applied: list[dict[str, int | str]],
    actions: torch.Tensor | None = None,
    action_bits: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    policy = {
        str(row["block_name"]): {
            "w_bits": int(row["weight_bits"]),
            "a_bits": int(row["activation_bits"]),
        }
        for row in applied
    }
    payload: dict[str, Any] = {"policy": policy}
    if actions is not None:
        payload["actions"] = [int(action) for action in actions.detach().cpu().reshape(-1).tolist()]
        payload["bits"] = [
            {"weight_bits": w_bits, "activation_bits": a_bits}
            for w_bits, a_bits in decode_actions(actions, action_bits=action_bits)
        ]
    return payload


def model_size_ratio(resource_stats: dict[str, Any]) -> float:
    fp32_bits = float(resource_stats.get("fp32_model_bits", 0.0))
    total_bits = float(resource_stats.get("total_model_bits", 0.0))
    return total_bits / fp32_bits if fp32_bits else 1.0


def build_policy_optimizer(actor: nn.Module, marl_config: dict[str, Any]) -> torch.optim.Optimizer:
    return AdamW(
        actor.parameters(),
        lr=float(marl_config.get("policy_lr", 3e-4)),
        weight_decay=float(marl_config.get("policy_weight_decay", 0.0)),
    )


def train_one_batch(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.train()
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        print("warning: non-finite MARL QAT loss; skipping batch.")
        return {"loss": float("nan"), "top1": 0.0, "skipped": 1.0}
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
    optimizer.step()
    clamp_lsq_scales(model)

    top1 = accuracy(logits, targets, topk=(1,))[0].item()
    return {"loss": float(loss.item()), "top1": float(top1), "skipped": 0.0}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(config)
    dataset_name = config["dataset"]["name"].lower().replace("-", "")
    num_classes = int(CIFAR_STATS[dataset_name]["num_classes"])
    qat_config = config.get("qat", {})
    marl_config = config.get("marl", {})

    weight_bits = int(qat_config.get("weight_bits", 8))
    activation_bits = int(qat_config.get("activation_bits", 8))
    first_last_bits = qat_config.get("first_last_bits")
    warmup_epochs = int(marl_config.get("warmup_epochs", qat_config.get("warmup_epochs", 5)))
    warmup_weight_bits = int(marl_config.get("warmup_weight_bits", 8))
    warmup_activation_bits = int(marl_config.get("warmup_activation_bits", 8))
    policy_interval = max(1, int(marl_config.get("policy_interval", 200)))
    entropy_beta = float(marl_config.get("entropy_beta", 0.01))
    actor_hidden_dim = int(marl_config.get("actor_hidden_dim", 128))
    static_finetune_epochs = int(marl_config.get("static_finetune_epochs", 0))
    bit_choices = tuple(int(bit) for bit in marl_config.get("bit_choices", DEFAULT_BIT_CHOICES))
    action_bits = build_action_bits(bit_choices)

    model = build_cifar_resnet_qat(
        config["model"]["name"],
        num_classes=num_classes,
        w_bits=weight_bits,
        a_bits=activation_bits,
    ).to(device)

    if args.fp32_checkpoint is not None:
        fp32_path = Path(args.fp32_checkpoint)
        if fp32_path.exists():
            checkpoint = torch.load(fp32_path, map_location=device)
            missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
            print(f"loaded fp32 checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
            fp32_checkpoint_info: dict[str, Any] | None = {
                "path": str(fp32_path),
                "missing": len(missing),
                "unexpected": len(unexpected),
            }
        else:
            print(f"warning: fp32 checkpoint not found at {fp32_path}; training from scratch.")
            fp32_checkpoint_info = {
                "path": str(fp32_path),
                "missing": None,
                "unexpected": None,
                "not_found": True,
            }
    else:
        fp32_checkpoint_info = None

    set_uniform_bit_widths(model, warmup_weight_bits, warmup_activation_bits)
    apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)

    if args.dry_run:
        initial_resource_stats = collect_resource_stats(model)
        initial_builder = AgentStateBuilder(initial_resource_stats)
        initial_states = initial_builder.build(model, {"epoch_ratio": 0.0})
        actor = SharedActor(state_dim=initial_states.shape[1], hidden_dim=actor_hidden_dim, action_bits=action_bits).to(device)
        print(
            f"dry-run ok: model={config['model']['name']} num_classes={num_classes} device={device} "
            f"agents={initial_states.shape[0]} state_dim={initial_states.shape[1]} "
            f"actions={actor.num_actions} bit_choices={list(bit_choices)}"
        )
        return

    train_loader, val_loader, _ = build_cifar_loaders(config)
    criterion = nn.CrossEntropyLoss()
    warmstart_metrics = (
        evaluate(model, val_loader, criterion, device)
        if fp32_checkpoint_info and not fp32_checkpoint_info.get("not_found")
        else None
    )
    if warmstart_metrics is not None:
        print(
            f"warmstart_eval top1={warmstart_metrics['top1']:.2f} "
            f"loss={warmstart_metrics['loss']:.4f}"
        )

    initial_resource_stats = collect_resource_stats(model)
    initial_builder = AgentStateBuilder(initial_resource_stats)
    initial_states = initial_builder.build(model, {"epoch_ratio": 0.0})
    actor = SharedActor(state_dim=initial_states.shape[1], hidden_dim=actor_hidden_dim, action_bits=action_bits).to(device)
    model_optimizer = build_optimizer(config, model)
    policy_optimizer = build_policy_optimizer(actor, marl_config)
    policy_learner = ReinforcePolicyLearner(
        actor=actor,
        optimizer=policy_optimizer,
        config=PolicyLearningConfig(
            entropy_beta=entropy_beta,
            max_grad_norm=float(marl_config.get("policy_max_grad_norm", 5.0)),
            advantage_clip=float(marl_config.get("advantage_clip", 10.0)),
        ),
    )

    epochs = int(args.epochs or config["training"].get("epochs", 200))
    scheduler = build_scheduler(config, model_optimizer, epochs)
    reward_config = RewardConfig(
        lambda_bitops=float(marl_config.get("lambda_bitops", 0.1)),
        lambda_local=float(marl_config.get("lambda_local", 0.01)),
        smoothing_alpha=float(marl_config.get("reward_smoothing_alpha", 0.9)),
        val_subset_size=int(marl_config.get("val_subset_size", 2048)),
        clip_min=float(marl_config.get("reward_clip_min", -100.0)),
        clip_max=float(marl_config.get("reward_clip_max", 100.0)),
    )
    reward_evaluator = RewardEvaluator(config=reward_config, criterion=criterion)

    output_config = config.get("output", {})
    checkpoint_path = Path(output_config.get("checkpoint", "outputs/checkpoints/marl_qat_best.pt"))
    last_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.pt")
    policy_path = Path(output_config.get("policy", "outputs/policies/marl_qat_policy.json"))
    log_path = Path(output_config.get("log", "outputs/logs/marl_qat.json"))
    log_interval = int(config["training"].get("log_interval", 100))
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    history: list[dict[str, Any]] = []
    policy_history: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_policy: dict[str, Any] | None = None
    best_policy_epoch = -1
    best_policy_step = -1
    global_step = 0
    total_start = time.time()

    for epoch in range(epochs):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        epoch_start = time.time()

        if epoch < warmup_epochs:
            set_uniform_bit_widths(model, warmup_weight_bits, warmup_activation_bits)
            apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)

        for step, (images, targets) in enumerate(train_loader, start=1):
            adaptive_enabled = epoch >= warmup_epochs
            policy_update = adaptive_enabled and global_step % policy_interval == 0

            if policy_update:
                resource_stats = collect_resource_stats(model)
                state_builder = AgentStateBuilder(resource_stats)
                states = state_builder.build(
                    model,
                    {
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "bitops_budget_ratio": marl_config.get("bitops_budget_ratio", 1.0),
                    },
                )
                actions, log_probs, entropy = actor.sample(states)
                bits = decode_actions(actions, action_bits=action_bits)
                applied = apply_block_policy(model, bits)
                apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)
                updated_resource_stats = collect_resource_stats(model)
                reward_result = reward_evaluator.evaluate(model, val_loader, device, resource_stats=updated_resource_stats)
                if not torch.isfinite(torch.tensor(reward_result.validation_loss, device=device)):
                    print(
                        f"warning: sampled policy produced non-finite validation loss at "
                        f"epoch={epoch} step={step}; reverting to W8A8 and skipping policy update."
                    )
                    set_uniform_bit_widths(model, 8, 8)
                    apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)
                    global_step += 1
                    continue
                policy_update_result = policy_learner.update(
                    log_probs=log_probs,
                    entropy=entropy,
                    local_rewards=reward_result.local_rewards,
                    baseline=reward_result.smoothed_reward,
                )
                policy_record = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "actions": [int(action) for action in actions.detach().cpu().tolist()],
                    "bits": [{"weight_bits": w_bits, "activation_bits": a_bits} for w_bits, a_bits in bits],
                    "policy_update": policy_update_result.to_dict(),
                    "policy_loss": policy_update_result.loss,
                    "entropy": float(entropy.mean().detach().item()),
                    "reward": reward_result.to_dict(),
                    "state_summary": summarize_agent_states(states),
                }
                policy_history.append(policy_record)

                if reward_result.smoothed_reward > best_reward:
                    best_reward = reward_result.smoothed_reward
                    best_policy = {
                        "model": config["model"]["name"],
                        "dataset": config["dataset"]["name"],
                        **applied_policy_to_dict(applied, actions, action_bits=action_bits),
                        "bitops_ratio": updated_resource_stats["bitops_ratio"],
                        "model_size_ratio": model_size_ratio(updated_resource_stats),
                        "reward": reward_result.to_dict(),
                        "bit_choices": list(bit_choices),
                        "action_bits": [
                            {"weight_bits": w_bits, "activation_bits": a_bits}
                            for w_bits, a_bits in action_bits
                        ],
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                    best_policy_epoch = epoch
                    best_policy_step = global_step
                    save_json(policy_path, best_policy)

            batch_metrics = train_one_batch(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                optimizer=model_optimizer,
                device=device,
                grad_clip=grad_clip if grad_clip > 0 else None,
            )
            if batch_metrics.get("skipped", 0.0) == 0.0:
                loss_meter.update(batch_metrics["loss"], targets.size(0))
                acc_meter.update(batch_metrics["top1"], targets.size(0))

            if log_interval > 0 and step % log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} global_step={global_step} "
                    f"loss={loss_meter.avg:.4f} top1={acc_meter.avg:.2f} policies={len(policy_history)}"
                )
            global_step += 1

        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(model_optimizer.param_groups[0]["lr"]),
            "train_loss": loss_meter.avg,
            "train_top1": acc_meter.avg,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "best_reward": best_reward,
            "best_policy_epoch": best_policy_epoch,
            "best_policy_step": best_policy_step,
            "policy_updates": len(policy_history),
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"epoch={epoch} lr={row['lr']:.6f} train_loss={row['train_loss']:.4f} "
            f"train_top1={row['train_top1']:.2f} val_loss={row['val_loss']:.4f} "
            f"val_top1={row['val_top1']:.2f} best_reward={best_reward:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "actor": actor.state_dict(),
            "model_optimizer": model_optimizer.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_reward": best_reward,
            "best_policy": best_policy,
            "config": config,
        }
        save_checkpoint(last_checkpoint_path, payload)
        if best_policy is not None and best_policy_epoch == epoch:
            save_checkpoint(checkpoint_path, payload)

        save_json(
            log_path,
            {
                "config": config,
                "seed": seed,
                "fp32_checkpoint": fp32_checkpoint_info,
                "warmstart_metrics": warmstart_metrics,
                "latest_epoch": epoch,
                "history": history,
                "policy_history": policy_history,
                "best_policy": best_policy,
                "total_seconds": time.time() - total_start,
            },
        )

    if best_policy is not None:
        apply_block_policy(model, best_policy["policy"])
        apply_first_last_bits(model, int(first_last_bits) if first_last_bits is not None else None)

    for fine_tune_epoch in range(static_finetune_epochs):
        train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=model_optimizer,
            device=device,
            epoch=epochs + fine_tune_epoch,
            log_interval=log_interval,
            grad_clip=grad_clip if grad_clip > 0 else None,
        )

    final_metrics = evaluate(model, val_loader, criterion, device)
    final_resource_stats = collect_resource_stats(model)
    final_payload = {
        "epoch": epochs + static_finetune_epochs - 1,
        "model": model.state_dict(),
        "actor": actor.state_dict(),
        "best_reward": best_reward,
        "best_policy": best_policy,
        "final_metrics": final_metrics,
        "final_resource_stats": final_resource_stats,
        "config": config,
    }
    save_checkpoint(last_checkpoint_path, final_payload)
    if best_policy is not None:
        save_checkpoint(checkpoint_path, final_payload)
        save_json(policy_path, best_policy)

    print(
        f"Final Top-1: {final_metrics['top1']:.2f}. "
        f"Best reward: {best_reward:.4f}. Policy: {policy_path}"
    )


if __name__ == "__main__":
    main()
