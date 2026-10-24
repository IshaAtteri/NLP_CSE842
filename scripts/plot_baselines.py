"""Plot saved baseline metrics; no retrieval, evaluation, or estimated measurements."""
import argparse
import json
import math
from pathlib import Path

METHODS = {"bm25": ("BM25", "#0072B2", "o"), "dense": ("Dense", "#D55E00", "s")}
METRICS = {"recall": "Recall@k", "mrr": "MRR@k", "all_hop_coverage": "All-Hop Coverage@k"}
CUTOFFS = (5, 10, 20, 50)


def load_summary(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Evaluation summary not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        baselines = data["baselines"]
        counts = []
        for method in METHODS:
            baseline = baselines[method]
            count = baseline["evaluated_questions"]
            if type(count) is not int or count <= 0:
                raise ValueError(f"Invalid evaluated_questions for {method}")
            counts.append(count)
            for k in CUTOFFS:
                for metric in METRICS:
                    value = baseline["metrics_at_k"][str(k)][metric]
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or not 0 <= value <= 1):
                        raise ValueError(f"Invalid {method}.{metric}@{k}: expected finite value in [0, 1]")
        if len(set(counts)) != 1:
            raise ValueError("Baselines have different evaluated question counts")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing metric or required field in evaluation summary: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in evaluation summary: {path}") from exc
    return baselines


def build_figures(baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.titlesize": 13, "axes.labelsize": 11,
             "axes.spines.top": False, "axes.spines.right": False,
             "pdf.fonttype": 42, "savefig.dpi": 300}
    figures = {}
    with plt.rc_context(style):
        count = baselines["bm25"]["evaluated_questions"]
        subtitle = f"MultiHop-RAG | {count:,} evidence-bearing questions"
        for metric, label in METRICS.items():
            fig, ax = plt.subplots(figsize=(6.6, 4.5), layout="constrained")
            fig.suptitle(subtitle, fontsize=11)
            ax.set_title(label)
            for method, (name, color, marker) in METHODS.items():
                values = [baselines[method]["metrics_at_k"][str(k)][metric] for k in CUTOFFS]
                ax.plot(CUTOFFS, values, label=name, color=color, marker=marker,
                        linewidth=1.8, markersize=6, linestyle="-" if method == "bm25" else "--")
            ax.set(xlabel="Retrieved passages (k)", ylabel=label, xticks=CUTOFFS,
                   xlim=(3, 52), ylim=(0, 1))
            ax.grid(axis="y", alpha=.22)
            ax.legend(loc="lower right", frameon=False)
            fig.supxlabel("Markers show measured cutoffs; lines are visual guides only.", fontsize=8)
            figures[metric + "_at_k"] = fig
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
        fig.suptitle(subtitle, fontsize=11)
        ax.set_title("Baseline comparison at k = 50")
        positions = list(range(len(METRICS)))
        for offset, (method, (name, color, _)) in zip((-.19, .19), METHODS.items()):
            values = [baselines[method]["metrics_at_k"]["50"][metric] for metric in METRICS]
            bars = ax.bar([x + offset for x in positions], values, width=.36,
                          color=color, label=name, hatch="" if method == "bm25" else "//",
                          edgecolor="white", linewidth=.6)
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=10)
        ax.set(xticks=positions, xticklabels=[v.replace("@k", "@50") for v in METRICS.values()],
               ylabel="Score", ylim=(0, 1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncols=2, frameon=False)
        figures["metrics_at_50"] = fig
    return figures


def plot_summary(summary_path, output_dir):
    baselines = load_summary(summary_path)
    output = Path(output_dir)
    names = [m + "_at_k" for m in METRICS] + ["metrics_at_50"]
    paths = [output / f"{name}.{ext}" for name in names for ext in ("png", "pdf")]
    if any(path.exists() for path in paths):
        raise ValueError("Figure files already exist; choose a new --output-dir to preserve them")
    figures = build_figures(baselines)
    import matplotlib.pyplot as plt
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{name}.{ext}", dpi=300, facecolor="white")
    finally:
        for fig in figures.values():
            plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Saved evaluation summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/multihoprag"))
    args = parser.parse_args()
    try:
        paths = plot_summary(args.summary, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
