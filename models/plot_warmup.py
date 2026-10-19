"""Plot completed matched warmups; next-token loss is not inference accuracy."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_dataset_generator.generator import file_checksum, write_json
from .plots import save_svg
from .representations import REPRESENTATIONS
from .syntax_warmup import initialization


def load_pair(directories):
    arms = {}
    keys = ("size", "parameter_count", "seed", "source", "steps", "batch_size", "context",
            "token_budget", "lr", "dataset_manifest_sha256", "train_sha256", "transformer_sha256",
            "renderer_sha256", "warmup_code_sha256")
    shared = None
    for directory in directories:
        report_path = directory/"warmup_report.json"
        report = json.loads(report_path.read_text())
        config = report["config"]
        _, checked = initialization(directory/"initialization.pt", config["size"], config["seed"],
                                    config["source"], config["dataset_manifest_sha256"])
        signature = tuple(config[key] for key in keys)
        if shared is not None and signature != shared:
            raise ValueError("Warmup arms have unmatched budgets, seed, architecture or provenance")
        shared = signature
        treatment = config["treatment"]
        if treatment not in ("source_only", "all_formats") or treatment in arms:
            raise ValueError("Requires one source-only and one all-format arm")
        history_path = directory/"history.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        previous = 0
        for row in history:
            step = row["step"]
            if not previous < step <= config["steps"] or not math.isfinite(row["last_batch_loss"]):
                raise ValueError("Invalid warmup history")
            if row["predicted_tokens"] != step*config["batch_size"]*config["context"]:
                raise ValueError("Incorrect historical token budget")
            previous = step
        if not history or previous != config["steps"]:
            raise ValueError("Incomplete warmup history")
        if any(history[-1][key] != checked[key] for key in ("records_started", "forms_rendered")):
            raise ValueError("History/report exposure mismatch")
        if history[-1]["predicted_tokens"] != config["token_budget"]:
            raise ValueError("Incorrect final warmup budget")
        arms[treatment] = {"report":checked, "history":history,
                           "report_sha256":file_checksum(report_path), "history_sha256":file_checksum(history_path)}
    if set(arms) != {"source_only", "all_formats"}:
        raise ValueError("Requires both matched arms")
    return arms


def export(directories, output):
    arms = load_pair(directories)
    config = arms["source_only"]["report"]["config"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    treatments = ("source_only", "all_formats")
    for treatment, label in zip(treatments, ("Source-only", "All formats")):
        history = arms[treatment]["history"]
        axes[0].plot([h["predicted_tokens"]/1e6 for h in history],
                     [h["last_batch_loss"] for h in history], marker="o", markersize=3, label=label)
    axes[0].set(xlabel="Next-token predictions (millions)", ylabel="Logged minibatch cross-entropy",
                title="Training loss on different text distributions")
    axes[0].legend(fontsize=8)
    bottom = [0, 0]
    for representation in REPRESENTATIONS:
        values = [arms[t]["report"]["forms_rendered"][representation] for t in treatments]
        axes[1].bar((0, 1), values, bottom=bottom, label=representation)
        bottom = [a+b for a, b in zip(bottom, values)]
    axes[1].set(xticks=(0, 1), xticklabels=("Source-only", "All formats"), ylabel="Records started (last may be partial)",
                title=f"Both arms: {config['token_budget']:,} prediction tokens")
    axes[1].set_ylim(0,max(bottom)*1.15)
    for index,total in enumerate(bottom):
        axes[1].text(index,total+max(bottom)*.025,f"{total:,}",ha="center",fontsize=9)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{config['parameter_count']:,} parameters; seed {config['seed']}; source {config['source']}", fontsize=12)
    fig.supxlabel("TRAIN-only exposure; loss is not task accuracy. Equal tokens do not mean equal records or content.", fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    stem = output/f"warmup_{config['size']}_{config['source']}_seed{config['seed']}"
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    save_svg(fig, stem.with_suffix(".svg"))
    plt.close(fig)
    write_json(stem.with_suffix(".json"), {"scope":"Auxiliary TRAIN-only exposure, not inference results",
        "interpretation":"Minibatch losses use different text distributions; do not rank inference ability or syntax competence from these curves. Budgets match but consumed content and record counts differ.",
        "arms":arms})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_only", type=Path)
    parser.add_argument("all_formats", type=Path)
    parser.add_argument("--output", type=Path, default=Path("graphs"))
    args = parser.parse_args()
    export((args.source_only,args.all_formats),args.output)
