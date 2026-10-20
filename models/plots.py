"""Plot measured results only; incomplete runs stay visible as incomplete."""

import argparse
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "paper1"

from synthetic_dataset_generator.generator import write_json
from .representations import REPRESENTATIONS


def save_svg(fig, path):
    fig.savefig(path, metadata={"Date": None})
    # Matplotlib adds trailing spaces to path coordinates; normalize generated SVG.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def export(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for directory in sorted(runs.iterdir()):
        if not directory.is_dir() or not (directory / "config.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()] if (directory / "history.jsonl").exists() else []
        if history:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
            epochs = [h["epoch"] for h in history]
            marker = "o" if len(history) == 1 else None
            axes[0].plot(epochs, [h["train_loss"] for h in history], label="Training",marker=marker)
            axes[0].plot(epochs, [h["validation"]["loss"] for h in history], label="Source validation",marker=marker)
            axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
            axes[1].plot(epochs, [h["train_accuracy"] for h in history], label="Training",marker=marker)
            axes[1].plot(epochs, [h["validation"]["accuracy"] for h in history], label="Source validation",marker=marker)
            axes[1].axhline(1/3, color="grey", linestyle="--", label="Uniform-random expectation")
            axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
            for axis in axes:
                axis.legend(fontsize=8)
                axis.grid(alpha=0.2)
                if len(history) == 1:
                    axis.set_xlim(epochs[0]-.5,epochs[0]+.5)
                    axis.set_xticks(epochs)
            fig.suptitle(f"{directory.name} - measured learning curve")
            fig.savefig(output / f"{directory.name}_learning.png", dpi=180)
            save_svg(fig, output / f"{directory.name}_learning.svg")
            plt.close(fig)
        result_path = directory / "results.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result["status"] != "completed":
            continue
        completed.append({"run": directory.name, "result": result})
        source = config["representation"]
        values = [result["evaluations"][rep]["test_iid"]["accuracy"] for rep in REPRESENTATIONS]
        intervals = [result["evaluations"][rep]["test_iid"]["wilson95"] for rep in REPRESENTATIONS]
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.bar(REPRESENTATIONS, values, yerr=np.array([[v-lo for v,(lo,hi) in zip(values, intervals)],
                                                     [hi-v for v,(lo,hi) in zip(values, intervals)]]), capsize=4)
        ax.axhline(1/3, color="grey", linestyle="--", label="Uniform-random expectation")
        ax.set(ylim=(0,1), ylabel="IID accuracy (95% binomial interval)", xlabel="Evaluation representation",
               title=f"Train: {source}; {config['parameter_count']:,} parameters\n{result['scope']} - {directory.name}")
        ax.legend()
        fig.savefig(output / f"{directory.name}_transfer.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for rep in REPRESENTATIONS:
            metrics = result["evaluations"][rep]
            values = [metrics["test_iid" if d == 2 else f"test_depth_{d}"]["accuracy"] for d in (2,3,4,5)]
            ax.plot((2,3,4,5), values, marker="o", label=rep)
        ax.axhline(1/3, color="grey", linestyle="--")
        ax.set(ylim=(0,1), xticks=(2,3,4,5), xlabel="Matched scaffold / decisive proof depth", ylabel="Accuracy",
               title=f"{directory.name} - {result['scope']}")
        ax.legend()
        fig.savefig(output / f"{directory.name}_depth.png", dpi=180)
        plt.close(fig)
    write_json(output / "measured_results.json", completed)
    # Missing source conditions are explicitly masked, never estimated.
    grouped = {}
    for item in completed:
        result = item["result"]
        config = result["config"]
        key = (result["scope"], config["size"], config["seed"], config["parameter_count"],
               config["train_examples"], config["dataset_manifest_sha256"], config.get("encoding","characters"))
        grouped.setdefault(key, []).append(result)
    for (scope,size,seed,parameters,train_examples,fingerprint,encoding), results in grouped.items():
        sources = [r["config"]["representation"] for r in results]
        if len(sources) != len(set(sources)):
            raise ValueError("Duplicate source runs: comparisons require an explicit treatment grouping")
        matrix = np.full((3,3), np.nan)
        for result in results:
            source = REPRESENTATIONS.index(result["config"]["representation"])
            matrix[source] = [result["evaluations"][rep]["test_iid"]["accuracy"] for rep in REPRESENTATIONS]
        fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eeeeee")
        rendered = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=REPRESENTATIONS, yticklabels=REPRESENTATIONS,
               xlabel="Test representation", ylabel="Training representation",
               title=f"{scope}; {parameters:,} parameters; seed {seed}\n{train_examples:,} training examples; unrun cells are not measurements")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,"not run" if np.isnan(matrix[i,j]) else f"{matrix[i,j]:.3f}",ha="center",va="center",
                        color="white" if matrix[i,j] > 0.65 else "black")
        fig.colorbar(rendered,ax=ax,label="IID accuracy")
        fig.savefig(output/f"{scope.replace(' ','_')}_{size}_seed{seed}_p{parameters}_n{train_examples}_transfer_matrix.png",dpi=180)
        plt.close(fig)
    baseline_path = runs.parent / "baseline_results.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        splits = list(baseline["splits"])
        fig,ax = plt.subplots(figsize=(11,4),constrained_layout=True)
        x = np.arange(len(splits))
        for offset,(key,label) in enumerate((("oracle","Exact inference"),("surface_features","Surface features"),("training_majority","Training majority"))):
            ax.bar(x+(offset-1)*0.25,[baseline["splits"][s][key]["accuracy"] for s in splits],width=0.25,label=label)
        ax.axhline(1/3,color="grey",linestyle="--",linewidth=1)
        ax.set(ylim=(0,1.05),ylabel="Accuracy",xticks=x,
               xticklabels=[s.removeprefix("test_") for s in splits],title="Full-corpus baseline measurements")
        ax.tick_params(axis="x",rotation=30)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
        fig.savefig(output/"full_corpus_baselines.png",dpi=180)
        plt.close(fig)
    lines = ["# Measured experiment results", "", "Smoke/pilot results are not final paper evidence. Confidence intervals describe finite test sampling, not training-seed uncertainty.", "",
             "| Run | Scope | Parameters | Train format | Test format | IID accuracy | Examples |", "| --- | --- | ---: | --- | --- | ---: | ---: |"]
    for item in completed:
        result = item["result"]
        for rep in REPRESENTATIONS:
            metric = result["evaluations"][rep]["test_iid"]
            lines.append(f"| {item['run']} | {result['scope']} | {result['config']['parameter_count']} | {result['config']['representation']} | {rep} | {metric['accuracy']:.4f} | {metric['count']} |")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return len(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("models/runs"))
    parser.add_argument("--output", type=Path, default=Path("graphs"))
    args = parser.parse_args()
    # Primary and auxiliary queues can finish together; serialize CLI exports.
    with Path(".plots.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(f"Exported {export(args.runs, args.output)} completed runs")


if __name__ == "__main__":
    main()
