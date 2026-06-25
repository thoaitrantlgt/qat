from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline results.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--markdown", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    if not rows:
        print(f"no rows found in {csv_path}")
        return

    fp32_row = next((row for row in rows if row.get("name") == "fp32"), rows[0])
    fp32_top1 = as_float(fp32_row.get("top1"), 0.0)

    summary_rows = []
    for row in rows:
        top1 = as_float(row.get("top1"))
        summary_rows.append(
            {
                **row,
                "accuracy_drop": fp32_top1 - top1,
            }
        )

    if args.json is not None:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reference": fp32_row.get("name", "fp32"),
                    "reference_top1": fp32_top1,
                    "rows": summary_rows,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    if args.markdown is not None:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "| name | top1 | accuracy_drop | model_size_mb | compression_ratio | avg_weight_bits | avg_activation_bits | bitops_ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            lines.append(
                "| {name} | {top1:.2f} | {accuracy_drop:.2f} | {model_size_mb:.3f} | {compression_ratio:.3f} | {avg_weight_bits:.2f} | {avg_activation_bits:.2f} | {bitops_ratio:.3f} |".format(
                    name=row.get("name", ""),
                    top1=as_float(row.get("top1")),
                    accuracy_drop=as_float(row.get("accuracy_drop")),
                    model_size_mb=as_float(row.get("model_size_mb")),
                    compression_ratio=as_float(row.get("compression_ratio")),
                    avg_weight_bits=as_float(row.get("avg_weight_bits")),
                    avg_activation_bits=as_float(row.get("avg_activation_bits")),
                    bitops_ratio=as_float(row.get("bitops_ratio")),
                )
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for row in summary_rows:
        print(
            f"{row.get('name', '')}: top1={as_float(row.get('top1')):.2f}, "
            f"accuracy_drop={as_float(row.get('accuracy_drop')):.2f}, "
            f"compression_ratio={as_float(row.get('compression_ratio')):.3f}"
        )


if __name__ == "__main__":
    main()
