import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "thesis_assets"


def parse_simple_log(path):
    text = Path(path).read_text(errors="ignore")
    metrics = {}
    patterns = {
        "micro_f1": r"Micro F1:\s+([0-9.]+)",
        "macro_f1": r"Macro F1:\s+([0-9.]+)",
        "ndcg": r"NDCG@5:\s+([0-9.]+)",
        "mrr": r"MRR@5:\s+([0-9.]+)",
        "map": r"MAP:\s+([0-9.]+)",
        "micro_f1_yf": r"Micro F1 yf:\s+([0-9.]+)",
        "macro_f1_yf": r"Macro F1 yf:\s+([0-9.]+)",
        "ndcg_yf": r"NDCG@5 yf:\s+([0-9.]+)",
        "mrr_yf": r"MRR@5 yf:\s+([0-9.]+)",
        "map_yf": r"MAP yf:\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def parse_epoch_log_best(path):
    text = Path(path).read_text(errors="ignore")
    entries = []
    current = {}

    patterns = {
        "micro_f1": r"Micro F1:\s+([0-9.]+)",
        "macro_f1": r"Macro F1:\s+([0-9.]+)",
        "ndcg": r"NDCG@5:\s+tensor\(([0-9.]+)\)",
        "mrr": r"MRR@5:\s+tensor\(([0-9.]+)\)",
        "map": r"MAP:\s+tensor\(([0-9.]+)\)",
        "micro_f1_yf": r"Micro F1 yf:\s+([0-9.]+)",
        "macro_f1_yf": r"Macro F1 yf:\s+([0-9.]+)",
        "ndcg_yf": r"NDCG@5 yf:\s+tensor\(([0-9.]+)\)",
        "mrr_yf": r"MRR@5 yf:\s+tensor\(([0-9.]+)\)",
        "map_yf": r"MAP yf:\s+tensor\(([0-9.]+)\)",
    }

    for line in text.splitlines():
        epoch_match = re.search(r"Epoch:\s*(\d+)", line)
        if epoch_match:
            if current:
                entries.append(current)
                current = {}
            current["epoch"] = int(epoch_match.group(1))
            continue

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                current[key] = float(match.group(1))

    if current:
        entries.append(current)

    entries = [entry for entry in entries if "ndcg" in entry]
    best = {}
    for key in patterns:
        candidates = [entry for entry in entries if key in entry]
        if candidates:
            best[key] = max(candidates, key=lambda item: item[key])[key]

    return best, entries


def build_results_dataframe():
    bm25 = parse_simple_log(ROOT / "BM25/Results/BM25_2017_run.log")
    legalbert = parse_simple_log(ROOT / "LegalBert/Results/legalbert.log")
    casegnn_best, casegnn_entries = parse_epoch_log_best(ROOT / "CaseGNN2017_run.log")
    fusion_best, fusion_entries = parse_epoch_log_best(ROOT / "casegnn_novel.log")

    df = pd.DataFrame(
        [
            {
                "Method": "BM25",
                "Micro-F1": bm25["micro_f1"],
                "Macro-F1": bm25["macro_f1"],
                "NDCG@5": bm25["ndcg"],
                "MRR@5": bm25["mrr"],
                "MAP": bm25["map"],
                "Micro-F1 yf": bm25["micro_f1_yf"],
                "Macro-F1 yf": bm25["macro_f1_yf"],
                "NDCG@5 yf": bm25["ndcg_yf"],
                "MRR@5 yf": bm25["mrr_yf"],
                "MAP yf": bm25["map_yf"],
            },
            {
                "Method": "Legal-BERT",
                "Micro-F1": legalbert["micro_f1"],
                "Macro-F1": legalbert["macro_f1"],
                "NDCG@5": legalbert["ndcg"],
                "MRR@5": legalbert["mrr"],
                "MAP": legalbert["map"],
                "Micro-F1 yf": legalbert["micro_f1_yf"],
                "Macro-F1 yf": legalbert["macro_f1_yf"],
                "NDCG@5 yf": legalbert["ndcg_yf"],
                "MRR@5 yf": legalbert["mrr_yf"],
                "MAP yf": legalbert["map_yf"],
            },
            {
                "Method": "CaseGNN",
                "Micro-F1": casegnn_best["micro_f1"],
                "Macro-F1": casegnn_best["macro_f1"],
                "NDCG@5": casegnn_best["ndcg"],
                "MRR@5": casegnn_best["mrr"],
                "MAP": casegnn_best["map"],
                "Micro-F1 yf": casegnn_best["micro_f1_yf"],
                "Macro-F1 yf": casegnn_best["macro_f1_yf"],
                "NDCG@5 yf": casegnn_best["ndcg_yf"],
                "MRR@5 yf": casegnn_best["mrr_yf"],
                "MAP yf": casegnn_best["map_yf"],
            },
            {
                "Method": "FusionCaseGNN",
                "Micro-F1": fusion_best["micro_f1"],
                "Macro-F1": fusion_best["macro_f1"],
                "NDCG@5": fusion_best["ndcg"],
                "MRR@5": fusion_best["mrr"],
                "MAP": fusion_best["map"],
                "Micro-F1 yf": fusion_best["micro_f1_yf"],
                "Macro-F1 yf": fusion_best["macro_f1_yf"],
                "NDCG@5 yf": fusion_best["ndcg_yf"],
                "MRR@5 yf": fusion_best["mrr_yf"],
                "MAP yf": fusion_best["map_yf"],
            },
        ]
    )
    return df, casegnn_entries, fusion_entries


def render_table(df):
    fig, ax = plt.subplots(figsize=(18, 3.8))
    ax.axis("off")

    formatted = df.copy()
    for column in formatted.columns[1:]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")

    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    header_color = "#1f3b4d"
    row_colors = ["#eef4f7", "#ffffff"]
    highlight_color = "#d8efe3"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#9eb3bf")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            if df.iloc[row - 1]["Method"] == "FusionCaseGNN":
                cell.set_facecolor(highlight_color)

    ax.set_title("Table 3.1 Comparative Retrieval Performance from Available Logs", fontsize=14, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results_comparison_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_barplots(df):
    metrics = ["Micro-F1", "NDCG@5", "MRR@5", "MAP"]
    metrics_yf = ["Micro-F1 yf", "NDCG@5 yf", "MRR@5 yf", "MAP yf"]
    colors = ["#5b8e7d", "#f2a65a", "#6f4e7c", "#2e86ab"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, cols, title in zip(
        axes,
        [metrics, metrics_yf],
        ["Without Year Filtering", "With Year Filtering"],
    ):
        plot_df = df.set_index("Method")[cols]
        plot_df.plot(kind="bar", ax=ax, color=colors, width=0.78)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Performance Comparison Across Retrieval Models", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results_comparison_barplots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_epoch_plot(casegnn_entries, fusion_entries):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, key, title in [
        (axes[0], "ndcg_yf", "Year-Filtered NDCG@5 Across Epochs"),
        (axes[1], "micro_f1_yf", "Year-Filtered Micro-F1 Across Epochs"),
    ]:
        case_x = [entry["epoch"] for entry in casegnn_entries if key in entry]
        case_y = [entry[key] for entry in casegnn_entries if key in entry]
        fusion_x = [entry["epoch"] for entry in fusion_entries if key in entry]
        fusion_y = [entry[key] for entry in fusion_entries if key in entry]

        ax.plot(case_x, case_y, label="CaseGNN", color="#6f4e7c", linewidth=2)
        ax.plot(fusion_x, fusion_y, label="FusionCaseGNN", color="#2e86ab", linewidth=2.4)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Training-Curve Comparison from Logged 2017 Runs", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results_epoch_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df, casegnn_entries, fusion_entries = build_results_dataframe()
    render_table(df)
    render_barplots(df)
    render_epoch_plot(casegnn_entries, fusion_entries)
    print("Saved:")
    print(OUT_DIR / "results_comparison_table.png")
    print(OUT_DIR / "results_comparison_barplots.png")
    print(OUT_DIR / "results_epoch_curves.png")


if __name__ == "__main__":
    main()
