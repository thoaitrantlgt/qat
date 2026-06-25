from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline comparison charts.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/tables/baselines_comparison.png")
    return parser.parse_args()


def ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = ["fp32", "w8a8", "w4a8", "w4a4", "w3a3"]
    existing = [name for name in preferred if name in set(frame["name"])]
    remaining = [name for name in frame["name"].tolist() if name not in existing]
    order = existing + remaining
    return frame.set_index("name").loc[order].reset_index()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"missing csv: {csv_path}")
        return

    frame = pd.read_csv(csv_path)
    if frame.empty:
        print(f"empty csv: {csv_path}")
        return

    frame = ordered_frame(frame)
    names = frame["name"].tolist()
    colors = ["#111827" if name == "fp32" else "#2563eb" for name in names]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    top1_bars = axes[0].barh(names, frame["top1"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("Top-1 Accuracy")
    axes[0].set_xlabel("Accuracy (%)")
    axes[0].grid(axis="x", linestyle=":", alpha=0.3)
    for bar, value in zip(top1_bars, frame["top1"].tolist()):
        axes[0].text(value + 0.1, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)

    bitops_bars = axes[1].barh(names, frame["bitops_ratio"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("BitOps Ratio vs FP32")
    axes[1].set_xlabel("Ratio")
    axes[1].grid(axis="x", linestyle=":", alpha=0.3)
    for bar, value in zip(bitops_bars, frame["bitops_ratio"].tolist()):
        axes[1].text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
