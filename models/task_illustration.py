"""Solver-checked Figure 1 toy example, never an experimental measurement."""

import argparse
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_dataset_generator.generator import file_checksum, write_json
from synthetic_dataset_generator.logic import Rule, canonical_problem, classical_label, infer
from .parsing import parse
from .plots import save_svg
from .representations import REPRESENTATIONS, encode, format_input


def example():
    facts, query = ["p000", "p001"], "p003"
    rules = [Rule((("p000", "p001"),), "p002"), Rule((("p002",),), query)]
    record = {"metadata":{"facts":facts,"rules":[r.to_dict() for r in rules],"query":query}}
    gold = classical_label(facts,rules,query)
    depths = infer(facts,rules)[0]
    if gold != "entailed" or depths[query] != 2:
        raise ValueError("Illustrative example is not a valid two-hop entailment")
    inputs = {}
    for representation in REPRESENTATIONS:
        text = format_input(record,representation)
        parsed = parse(text,representation)["metadata"]
        if canonical_problem(parsed["facts"],[Rule.from_dict(r) for r in parsed["rules"]],parsed["query"]) != canonical_problem(facts,rules,query):
            raise ValueError("Illustration renderings changed the latent problem")
        inputs[representation] = {"text":text,"tokens_including_boundaries":len(encode(text))}
    return {"scope":"Fixed pedagogical toy, not a sampled benchmark record, training input or model result",
            "record":record,"inputs":inputs,"solver_label":gold,"minimum_rule_depth":depths[query],
            "depths":depths,"code_sha256":file_checksum(Path(__file__)),
            "renderer_sha256":file_checksum(Path(__file__).with_name("representations.py"))}


def export(output):
    report = example()
    fig = plt.figure(figsize=(12,6.4),constrained_layout=True)
    grid = fig.add_gridspec(2,3,height_ratios=(1.1,1))
    graph = fig.add_subplot(grid[0,:])
    graph.set(xlim=(0,1),ylim=(0,1))
    graph.axis("off")
    positions = {"p000":(.10,.72),"p001":(.10,.27),"AND":(.32,.50),"p002":(.56,.50),"p003":(.88,.50)}
    for name,(x,y) in positions.items():
        label = "AND" if name == "AND" else f"{name}\ndepth {report['depths'][name]}"
        graph.text(x,y,label,ha="center",va="center",fontsize=12,
                   bbox={"boxstyle":"round,pad=.6","facecolor":"#dceefb" if name != "AND" else "#eeeeee","edgecolor":"#486579"})
    for left,right in (("p000","AND"),("p001","AND"),("AND","p002"),("p002","p003")):
        graph.annotate("",xy=positions[right],xytext=positions[left],
                       arrowprops={"arrowstyle":"->","shrinkA":34,"shrinkB":35,"color":"#486579","lw":1.8})
    graph.text(.44,.68,"rule 1",ha="center",fontsize=10)
    graph.text(.72,.68,"rule 2",ha="center",fontsize=10)
    graph.text(.10,.94,"Given facts",ha="center",fontsize=11)
    graph.text(.88,.94,"Query",ha="center",fontsize=11)
    graph.text(.50,.04,"One conjunction rule application adds one hop; both premises are required.",ha="center",fontsize=10)
    for index,representation in enumerate(REPRESENTATIONS):
        axis = fig.add_subplot(grid[1,index])
        axis.axis("off")
        text = report["inputs"][representation]["text"]
        if representation == "function":
            display = text.replace(";",";\n")
        else:
            display = "\n".join(textwrap.wrap(text,width=39,break_long_words=False,break_on_hyphens=False))
        axis.text(.02,.97,representation.capitalize(),va="top",fontsize=12,fontweight="bold")
        axis.text(.02,.80,display,va="top",fontsize=9,fontfamily="monospace",linespacing=1.6)
        axis.text(.02,.06,f"{report['inputs'][representation]['tokens_including_boundaries']} tokens including BOS/EOS",fontsize=9)
    fig.suptitle("Same logical problem, three lossless representations",fontsize=16)
    fig.supxlabel("Solver-checked answer: entailed; minimum proof depth: 2. Pedagogical toy, not a model result. Display wraps only.",fontsize=9)
    output.mkdir(parents=True,exist_ok=True)
    stem = output/"task_illustration"
    fig.savefig(stem.with_suffix(".png"),dpi=180)
    save_svg(fig,stem.with_suffix(".svg"))
    plt.close(fig)
    write_json(stem.with_suffix(".json"),report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("graphs"))
    args = parser.parse_args()
    export(args.output)
