from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from hmaq_vlm.reproducibility import atomic_write_bytes, atomic_write_json


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["cider"] >= candidate["cider"]
            and other["bitops_ratio"] <= candidate["bitops_ratio"]
            and other["model_size_ratio"] <= candidate["model_size_ratio"]
            and (other["cider"] > candidate["cider"] or other["bitops_ratio"] < candidate["bitops_ratio"] or other["model_size_ratio"] < candidate["model_size_ratio"])
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def select_policies(rows: list[dict[str, Any]], *, cost_target: float, quality_drop_target: float) -> dict[str, dict[str, Any]]:
    frontier = pareto_frontier(rows)
    best_quality = max(row["cider"] for row in rows)
    under_cost = [row for row in frontier if row["bitops_ratio"] <= cost_target]
    within_quality = [row for row in frontier if row["cider"] >= best_quality - quality_drop_target]
    return {
        "lowest_cost": min(frontier, key=lambda row: (row["bitops_ratio"], -row["cider"])),
        "highest_quality_under_cost": max(under_cost, key=lambda row: row["cider"]),
        "smallest_under_quality_drop": min(within_quality, key=lambda row: (row["model_size_ratio"], -row["cider"])),
    }


def build_result_artifacts(rows: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "metrics.json"
    csv_path = output / "metrics.csv"
    tex_path = output / "metrics.tex"
    atomic_write_json(json_path, rows)
    fields = list(rows[0]) if rows else []
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(csv_path, csv_buffer.getvalue().encode("utf-8"))
    columns = "l" + "r" * max(0, len(fields) - 1)
    lines = [f"\\begin{{tabular}}{{{columns}}}", " & ".join(fields) + r" \\ \hline"]
    lines.extend(" & ".join(str(row[field]) for field in fields) + r" \\" for row in rows)
    lines.append("\\end{tabular}")
    atomic_write_bytes(tex_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return [json_path, csv_path, tex_path]


def plot_pareto(rows: list[dict[str, Any]], path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Pareto plotting requires the reporting extra") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    for row in rows:
        axis.scatter(row["bitops_ratio"], row["cider"], label=row.get("method", "policy"))
    axis.set(xlabel="BitOps ratio (lower is better)", ylabel="CIDEr (higher is better)")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_policy_heatmap(policy: dict[str, dict[str, int]], path: str | Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("policy plotting requires the reporting extra") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(policy)
    values = [[policy[name]["weight_bits"], policy[name]["activation_bits"]] for name in names]
    figure, axis = plt.subplots(figsize=(5, max(2, len(names) * 0.25)))
    image = axis.imshow(values, aspect="auto", vmin=2, vmax=16)
    axis.set_xticks([0, 1], labels=["W", "A"])
    axis.set_yticks(range(len(names)), labels=names, fontsize=6)
    figure.colorbar(image, ax=axis, label="bits")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
