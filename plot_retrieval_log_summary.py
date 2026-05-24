#!/usr/bin/env python3
"""Parse retrieval logs and render summary tables/plots with matplotlib."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "casegnn-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "micro_f1": "Micro F1",
    "macro_f1": "Macro F1",
    "ndcg_at_5": "NDCG@5",
    "mrr_at_5": "MRR@5",
    "map": "MAP",
    "micro_f1_yf": "Micro F1 yf",
    "macro_f1_yf": "Macro F1 yf",
    "ndcg_at_5_yf": "NDCG@5 yf",
    "mrr_at_5_yf": "MRR@5 yf",
    "map_yf": "MAP yf",
}

PLOT_METRICS = [
    "micro_f1",
    "ndcg_at_5",
    "micro_f1_yf",
    "ndcg_at_5_yf",
]

NON_YF_METRICS = ["micro_f1", "macro_f1", "ndcg_at_5", "mrr_at_5", "map"]
YF_METRICS = ["micro_f1_yf", "macro_f1_yf", "ndcg_at_5_yf", "mrr_at_5_yf", "map_yf"]

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)")
METRIC_RE = re.compile(r"^(Micro F1|Macro F1|NDCG@5|MRR@5|MAP)( yf)?:\s*(.+?)\s*$")
WEIGHT_RE = re.compile(r"^(Fact/Issue weights|Graph/BM25 weights):\s*\[([^\]]+)\]")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def extract_float(text: str) -> Optional[float]:
    match = NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def extract_weights(text: str) -> List[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def metric_key(label: str, is_year_filtered: bool) -> str:
    key = {
        "Micro F1": "micro_f1",
        "Macro F1": "macro_f1",
        "NDCG@5": "ndcg_at_5",
        "MRR@5": "mrr_at_5",
        "MAP": "map",
    }[label]
    if is_year_filtered:
        key += "_yf"
    return key


def finalize_epoch_record(records: List[Dict[str, float]], record: Dict[str, float]) -> None:
    if any(metric in record for metric in METRIC_LABELS):
        records.append(record.copy())


def parse_training_log(path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    current: Dict[str, float] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                finalize_epoch_record(records, current)
                current = {"epoch": int(epoch_match.group(1))}
                continue

            metric_match = METRIC_RE.match(line)
            if metric_match:
                value = extract_float(metric_match.group(3))
                if value is None:
                    continue
                key = metric_key(metric_match.group(1), bool(metric_match.group(2)))
                current[key] = value
                continue

            weight_match = WEIGHT_RE.match(line)
            if weight_match:
                key = "fact_issue_weights" if weight_match.group(1) == "Fact/Issue weights" else "graph_bm25_weights"
                current[key] = extract_weights(weight_match.group(2))

    finalize_epoch_record(records, current)
    return records


def parse_single_result_log(path: Path) -> Dict[str, float]:
    record: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            metric_match = METRIC_RE.match(line)
            if not metric_match:
                continue
            value = extract_float(metric_match.group(3))
            if value is None:
                continue
            record[metric_key(metric_match.group(1), bool(metric_match.group(2)))] = value
    return record


def resolve_log_path(explicit_path: str, fallback_paths: List[str]) -> Path:
    for candidate in [explicit_path] + fallback_paths:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of: {[explicit_path] + fallback_paths}")


def format_metric(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.4f}"


def compute_best_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for metric in METRIC_LABELS:
        values = [record[metric] for record in records if metric in record]
        if values:
            best[metric] = max(values)
    return best


def summarize_methods(
    method_specs: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for spec in method_specs:
        kind = spec["kind"]
        path = spec["path"]
        method = spec["method"]

        if kind == "single":
            metrics = parse_single_result_log(path)
            summaries.append(
                {
                    "method": method,
                    "path": str(path),
                    "records": [metrics],
                    "best": metrics,
                }
            )
            continue

        records = parse_training_log(path)
        if not records:
            raise ValueError(f"No epoch metrics found in {path}")
        best = compute_best_metrics(records)
        summaries.append(
            {
                "method": method,
                "path": str(path),
                "records": records,
                "best": best,
            }
        )
    return summaries


def write_summary_csv(summaries: List[Dict[str, object]], output_dir: Path) -> Path:
    output_path = output_dir / "retrieval_summary.csv"
    header = ["method", "log_path"] + list(METRIC_LABELS)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for summary in summaries:
            best = summary["best"]
            writer.writerow(
                [
                    summary["method"],
                    summary["path"],
                    *[best.get(metric, "") for metric in METRIC_LABELS],
                ]
            )
    return output_path


def write_epoch_csvs(summaries: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    written: List[Path] = []
    for summary in summaries:
        records = summary["records"]
        if len(records) <= 1:
            continue
        output_path = output_dir / f"{summary['method'].lower().replace(' ', '_')}_epochs.csv"
        header = ["epoch"] + list(METRIC_LABELS)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for record in records:
                writer.writerow([record.get("epoch", ""), *[record.get(metric, "") for metric in METRIC_LABELS]])
        written.append(output_path)
    return written


def save_summary_table(summaries: List[Dict[str, object]], output_dir: Path) -> Path:
    columns = ["Method"] + [METRIC_LABELS[key] for key in METRIC_LABELS]
    rows = []
    for summary in summaries:
        best = summary["best"]
        rows.append(
            [
                summary["method"],
                *[format_metric(best.get(metric)) for metric in METRIC_LABELS],
            ]
        )

    fig, ax = plt.subplots(figsize=(20, 3 + len(rows) * 0.65))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DCE6F2")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F8FB")
    ax.set_title("Retrieval Log Summary", fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    output_path = output_dir / "retrieval_summary_table.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_grouped_bar(
    summaries: List[Dict[str, object]],
    metrics: List[str],
    title: str,
    output_path: Path,
) -> None:
    methods = [summary["method"] for summary in summaries]
    x_positions = list(range(len(metrics)))
    bar_width = 0.18
    offsets = [(-1.5 + idx) * bar_width for idx in range(len(methods))]

    fig, ax = plt.subplots(figsize=(12, 6))
    for method_index, summary in enumerate(summaries):
        values = [summary["best"].get(metric, 0.0) for metric in metrics]
        positions = [x + offsets[method_index] for x in x_positions]
        bars = ax.bar(positions, values, width=bar_width, label=summary["method"])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(summaries: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    trainable = [summary for summary in summaries if len(summary["records"]) > 1]
    if not trainable:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    for axis, metric in zip(axes, PLOT_METRICS):
        for summary in trainable:
            epochs = [record["epoch"] for record in summary["records"] if metric in record]
            values = [record[metric] for record in summary["records"] if metric in record]
            axis.plot(epochs, values, marker="o", linewidth=1.8, markersize=3, label=summary["method"])
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel("Score")
        axis.grid(True, linestyle="--", alpha=0.35)
    for axis in axes[2:]:
        axis.set_xlabel("Epoch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(trainable)))
    fig.suptitle("CaseGNN Metric Curves", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / "casegnn_training_curves.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_fusion_weight_plot(summaries: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    fusion_summary = next((summary for summary in summaries if "fusion" in summary["method"].lower()), None)
    if not fusion_summary:
        return None

    records = [
        record
        for record in fusion_summary["records"]
        if isinstance(record.get("fact_issue_weights"), list) and isinstance(record.get("graph_bm25_weights"), list)
    ]
    if not records:
        return None

    epochs = [record["epoch"] for record in records]
    fact_weights = [record["fact_issue_weights"][0] for record in records]
    issue_weights = [record["fact_issue_weights"][1] for record in records]
    graph_weights = [record["graph_bm25_weights"][0] for record in records]
    bm25_weights = [record["graph_bm25_weights"][1] for record in records]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    axes[0].plot(epochs, fact_weights, marker="o", label="Fact")
    axes[0].plot(epochs, issue_weights, marker="o", label="Issue")
    axes[0].set_title("Fact/Issue Fusion Weights")
    axes[1].plot(epochs, graph_weights, marker="o", label="Graph")
    axes[1].plot(epochs, bm25_weights, marker="o", label="BM25")
    axes[1].set_title("Graph/BM25 Fusion Weights")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Weight")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()
    fig.tight_layout()
    output_path = output_dir / "fusion_weights.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create summary tables and plots from retrieval logs.")
    parser.add_argument("--bm25-log", default="BM25_2017_run.log")
    parser.add_argument("--simple-log", default="CaseGNN2017_simple_run.log")
    parser.add_argument("--fusion-log", default="CaseGNN2017_fusion_run.log")
    parser.add_argument("--legalbert-log", default="LegalBert/Results/legalbert.log")
    parser.add_argument("--output-dir", default="log_plots")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    bm25_path = resolve_log_path(args.bm25_log, ["BM25/Results/BM25_2017_run.log"])
    simple_path = resolve_log_path(args.simple_log, ["CaseGNNNovel/CaseGNN2017_run.log"])
    fusion_path = resolve_log_path(args.fusion_log, ["CaseGNNNovel/casegnn_novel.log"])
    legalbert_path = resolve_log_path(
        args.legalbert_log,
        ["LegalBert/Results/legalbert.log", "CaseGNNNovel/LegalBert/Results/legalbert.log"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = summarize_methods(
        [
            {"method": "BM25", "path": bm25_path, "kind": "single"},
            {"method": "LegalBERT", "path": legalbert_path, "kind": "single"},
            {"method": "Simple CaseGNN", "path": simple_path, "kind": "training"},
            {"method": "Fusion CaseGNN", "path": fusion_path, "kind": "training"},
        ],
    )

    written_paths = [
        write_summary_csv(summaries, output_dir),
        save_summary_table(summaries, output_dir),
    ]
    written_paths.extend(write_epoch_csvs(summaries, output_dir))

    non_yf_path = output_dir / "retrieval_comparison_non_yf.png"
    yf_path = output_dir / "retrieval_comparison_yf.png"
    save_grouped_bar(summaries, NON_YF_METRICS, "Retrieval Comparison", non_yf_path)
    save_grouped_bar(summaries, YF_METRICS, "Retrieval Comparison With Year Filtering", yf_path)
    written_paths.extend([non_yf_path, yf_path])

    training_curve_path = save_training_curves(summaries, output_dir)
    if training_curve_path:
        written_paths.append(training_curve_path)

    fusion_weight_path = save_fusion_weight_plot(summaries, output_dir)
    if fusion_weight_path:
        written_paths.append(fusion_weight_path)

    for summary in summaries:
        best = summary["best"]
        print(
            f"{summary['method']}: "
            f"micro_f1={format_metric(best.get('micro_f1'))} "
            f"ndcg@5={format_metric(best.get('ndcg_at_5'))} "
            f"micro_f1_yf={format_metric(best.get('micro_f1_yf'))} "
            f"ndcg@5_yf={format_metric(best.get('ndcg_at_5_yf'))}"
        )

    print("Wrote:")
    for path in written_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
