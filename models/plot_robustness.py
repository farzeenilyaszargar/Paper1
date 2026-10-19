"""Plot measured intervention accuracy separately from prediction invariance."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plots import save_svg


def export(path, output):
    report = json.loads(path.read_text())
    names = ("original","rename","statement_order","irrelevant_components","query_negation","support_fact_ablation")
    labels = ("Original","Renamed","Reordered","Distractors","Negated query","Fact ablation")
    metrics = [report["variants"][name] for name in names]
    x = np.arange(len(names))
    fig,axes = plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    axes[0].bar(x,[m["accuracy"] if m["accuracy"] is not None else np.nan for m in metrics])
    axes[0].set(ylabel="Accuracy against solver-checked gold",title="Accuracy (intervention-specific label mix)")
    for i,metric in enumerate(metrics):
        if metric["accuracy"] is not None:
            axes[0].text(i,metric["accuracy"]+.015,f"{metric['accuracy']:.2%}",ha="center",fontsize=8)
    # Ablation golds are all unknown: a universal 1/3 chance line would mislead.
    for offset,(key,label) in enumerate((("unchanged_predictions","Prediction unchanged"),("jointly_correct","Original AND changed correct"))):
        axes[1].bar(x+(offset-.5)*.36,[m[key]/m["count"] if m["count"] else np.nan for m in metrics],width=.36,label=label)
    axes[1].set(ylabel="Fraction of applicable paired cases",title="Invariance alone does not demonstrate reasoning")
    axes[1].legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    for ax in axes:
        ax.set(ylim=(0,1.05),xticks=x,xticklabels=[f"{label}\nn={m['count']:,}" for label,m in zip(labels,metrics)])
        ax.tick_params(axis="x",labelsize=8,rotation=20)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"{report['run']} — {report['scope']}\n{report['representation']}; source learning gate: {report['source_learning_gate_passed']}",fontsize=12)
    output.mkdir(parents=True,exist_ok=True)
    stem = output/f"{report['run']}_robustness"
    fig.savefig(stem.with_suffix(".png"),dpi=180)
    save_svg(fig,stem.with_suffix(".svg"))
    plt.close(fig)
    lines = ["# Solver-checked robustness diagnostics","",
             f"Run: `{report['run']}`. Scope: {report['scope']}.","",report["interpretation"],"",
             "Fact ablation applies only to originally decisive cases; those modified gold labels are all unknown. Denominators and label mixes differ. No universal chance line is appropriate.","",
             "| Intervention | Applicable | Accuracy | Original & changed correct | Prediction unchanged | Gold changed | Correctly followed gold change |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label,m in zip(labels,metrics):
        accuracy = "N/A" if m["accuracy"] is None else f"{m['accuracy']:.4f}"
        lines.append(f"| {label} | {m['count']} | {accuracy} | {m['jointly_correct']} | {m['unchanged_predictions']} | {m['gold_label_changed']} | {m['correctly_followed_gold_change']} |")
    stem.with_suffix(".md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path,default=Path("graphs"))
    args = parser.parse_args()
    export(args.report,args.output)
