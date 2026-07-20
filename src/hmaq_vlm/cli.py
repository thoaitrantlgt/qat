from __future__ import annotations

import argparse
from pathlib import Path

from hmaq_vlm.config import load_config
from hmaq_vlm.data import CaptionImage, build_karpathy_manifests, build_manifests, load_coco_karpathy_records, load_flickr30k_records
from hmaq_vlm.reproducibility import atomic_write_json, collect_run_metadata
from hmaq_vlm.smoke import run_acceptance_smoke
from hmaq_vlm.workflow import evaluate_checkpoint, profile_sensitivity, search_policies, train_fp16, train_qat


def _resolve(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    revisions = {"coco": config.data.coco_revision, "vision": config.model.vision_revision, "gpt2": config.model.language_revision}
    atomic_write_json(args.output, collect_run_metadata(config.to_dict(), config.seed, revisions))


def _prepare_flickr(args: argparse.Namespace) -> None:
    records = load_flickr30k_records(args.images, args.annotations)
    build_manifests([CaptionImage(item.image_id, item.image_path, item.captions) for item in records], args.output, seed=args.seed)


def _acceptance_smoke(args: argparse.Namespace) -> None:
    run_acceptance_smoke(args.output, seed=args.seed, search_budget=args.search_budget, timing_budget=args.timing_budget)


def _prepare_coco(args: argparse.Namespace) -> None:
    records = load_coco_karpathy_records(args.cache)
    build_karpathy_manifests(records, args.output, seed=args.seed, policy_fraction=args.policy_fraction)


def _train_fp16(args: argparse.Namespace) -> None:
    train_fp16(args.config, args.manifests, args.output)


def _train_qat(args: argparse.Namespace) -> None:
    train_qat(args.config, args.manifests, args.teacher, args.policy, args.output)


def _search(args: argparse.Namespace) -> None:
    search_policies(args.config, args.manifests, args.teacher, args.method, args.output)


def _evaluate(args: argparse.Namespace) -> None:
    evaluate_checkpoint(args.config, args.manifests, args.checkpoint, args.split, args.output)


def _profile(args: argparse.Namespace) -> None:
    profile_sensitivity(args.config, args.manifests, args.teacher, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hmaq-vlm")
    commands = parser.add_subparsers(dest="command", required=True)
    resolved = commands.add_parser("resolve-config")
    resolved.add_argument("--config", type=Path, required=True)
    resolved.add_argument("--output", type=Path, required=True)
    resolved.set_defaults(function=_resolve)
    flickr = commands.add_parser("prepare-flickr")
    flickr.add_argument("--images", type=Path, required=True)
    flickr.add_argument("--annotations", type=Path, required=True)
    flickr.add_argument("--output", type=Path, required=True)
    flickr.add_argument("--seed", type=int, default=11)
    flickr.set_defaults(function=_prepare_flickr)
    smoke = commands.add_parser("acceptance-smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=11)
    smoke.add_argument("--search-budget", type=int, default=4)
    smoke.add_argument("--timing-budget", type=int, default=2)
    smoke.set_defaults(function=_acceptance_smoke)
    coco = commands.add_parser("prepare-coco")
    coco.add_argument("--cache", type=Path, required=True)
    coco.add_argument("--output", type=Path, required=True)
    coco.add_argument("--seed", type=int, default=11)
    coco.add_argument("--policy-fraction", type=float, default=0.10)
    coco.set_defaults(function=_prepare_coco)
    fp16 = commands.add_parser("train-fp16")
    qat = commands.add_parser("train-qat")
    search = commands.add_parser("search")
    evaluate = commands.add_parser("evaluate")
    profile = commands.add_parser("profile")
    for command in (fp16, qat, search, evaluate, profile):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--manifests", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    fp16.set_defaults(function=_train_fp16)
    qat.add_argument("--teacher", type=Path, required=True)
    qat.add_argument("--policy", type=Path, required=True)
    qat.set_defaults(function=_train_qat)
    search.add_argument("--teacher", type=Path, required=True)
    search.add_argument("--method", choices=("random", "greedy", "ppo", "mappo", "hierarchical_mappo"), required=True)
    search.set_defaults(function=_search)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.set_defaults(function=_evaluate)
    profile.add_argument("--teacher", type=Path, required=True)
    profile.set_defaults(function=_profile)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
