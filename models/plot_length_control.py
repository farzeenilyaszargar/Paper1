"""Export equal-length depth diagnostics from verified retained predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_dataset_generator.generator import file_checksum
from .paired_statistics import compare, holm, read_predictions, validate_metrics
from .plots import save_svg


def verified_report(path):
    report = json.loads(path.read_text())
    if report.get("status") != "completed" or set(report["depths"]) != {"3", "4", "5"}:
        raise ValueError("Requires all three completed length-control comparisons")
    directory = path.parent
    for name, key in (("results.json", "result_sha256"), ("best.pt", "checkpoint_sha256")):
        if file_checksum(directory/name) != report[key]:
            raise ValueError("Checkpoint or baseline provenance changed")
    comparisons = []
    for depth in (3, 4, 5):
        result = report["depths"][str(depth)]
        if (result["short_depth"], result["long_depth"]) != (2, depth):
            raise ValueError("Incorrect depth labels")
        rows = {}
        for side in ("short", "long"):
            prediction_path = directory/f"length_control_depth{depth}_{side}.predictions.jsonl"
            if file_checksum(prediction_path) != result["prediction_sha256"][side]:
                raise ValueError("Retained prediction checksum changed")
            rows[side] = read_predictions(prediction_path)
            metric = result["metrics"][side]
            validate_metrics(rows[side], metric)
            accuracy = sum(r["prediction"] == r["target"] for r in rows[side].values())/len(rows[side])
            if metric["accuracy"] != accuracy:
                raise ValueError("Saved accuracy disagrees with retained predictions")
        comparison = compare(rows["short"], rows["long"])
        comparison["long_minus_short_accuracy"] = comparison.pop("target_minus_source_accuracy")
        if any(result["paired_comparison"][key] != value for key, value in comparison.items()):
            raise ValueError("Saved paired comparison disagrees with retained predictions")
        comparisons.append(comparison)
    for depth, adjusted in zip((3, 4, 5), holm([c["p_exact_mcnemar"] for c in comparisons])):
        if report["depths"][str(depth)]["paired_comparison"]["p_holm_within_run_three_depths"] != adjusted:
            raise ValueError("Incorrect within-run multiplicity correction")
    return report


def export(path, output):
    # Refuse partial or inconsistent results before creating any graph files.
    report = verified_report(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    depths = (3, 4, 5)
    for side, label, marker in (("short", "Paired 2-hop cores", "o"), ("long", "Longer cores", "s")):
        values = [report["depths"][str(d)]["metrics"][side]["accuracy"] for d in depths]
        axes[0].plot(depths, values, marker=marker, label=label)
    axes[0].set(ylim=(0, 1), ylabel="Accuracy", title="Equal length and label within each pair")
    axes[0].legend(fontsize=8)
    differences = [report["depths"][str(d)]["paired_comparison"]["long_minus_short_accuracy"] for d in depths]
    axes[1].bar(depths, differences, width=.5)
    axes[1].axhline(0, color="grey", linewidth=1)
    axes[1].set(ylim=(-1, 1), ylabel="Long minus paired-short accuracy", title="Paired differences, not a causal depth effect")
    for depth, value in zip(depths, differences):
        axes[1].text(depth, value + (.035 if value >= 0 else -.065), f"{value:+.2%}", ha="center", fontsize=9)
    for axis in axes:
        axis.set(xticks=depths, xlabel="Long-side scaffold / decisive proof depth")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(f"{report['run']} — {report['representation']}\nSource learning gate: {report['source_learning_gate_passed']}", fontsize=11)
    fig.supxlabel("Different core KBs; unknowns have no decisive proof; reused short cores correlate depth conditions.", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    stem = output/f"{report['run']}_length_control"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    save_svg(fig, stem.with_suffix(".svg"))
    plt.close(fig)
    lines = ["# Equal-length depth diagnostics", "", f"Run: `{report['run']}`.", "", report["interpretation"], "",
             "Each depth has its own paired 2-hop reference. Short cores are reused across conditions; these are not independent replications. Unknown examples use scaffold depth. No training-seed uncertainty is estimated.", "",
             "| Long depth | Pairs | Paired 2-hop accuracy | Long accuracy | Long minus short | Exact McNemar p | Holm p (three depths within run) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for depth in depths:
        item = report["depths"][str(depth)]
        short, long = item["metrics"]["short"], item["metrics"]["long"]
        pair = item["paired_comparison"]
        lines.append(f"| {depth} | {pair['count']} | {short['accuracy']:.4f} | {long['accuracy']:.4f} | {pair['long_minus_short_accuracy']:+.4f} | {pair['p_exact_mcnemar']:.6g} | {pair['p_holm_within_run_three_depths']:.6g} |")
    lines.extend(["", f"Source report SHA256: `{file_checksum(path)}`.", ""])
    stem.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("graphs"))
    args = parser.parse_args()
    export(args.report, args.output)
