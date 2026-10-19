"""Measure corpus-level raw-character exposure, not learned transfer accuracy."""

import argparse
import json
from collections import Counter
from pathlib import Path

from synthetic_dataset_generator.generator import file_checksum,write_json
from .representations import CHAR_IDS,REPRESENTATIONS,format_input


def compare_characters(source_counts,target_counts):
    if not target_counts or any(c not in CHAR_IDS for c in set(source_counts)|set(target_counts)):
        raise ValueError("Requires nonempty supported character counts")
    if any(not isinstance(n,int) or isinstance(n,bool) or n <= 0 for n in list(source_counts.values())+list(target_counts.values())):
        raise ValueError("Observed character counts must be positive integers")
    unseen = sorted(set(target_counts)-set(source_counts))
    occurrences = sum(target_counts[c] for c in unseen)
    total = sum(target_counts.values())
    return {"source_unseen_characters":unseen,"source_unseen_character_ids":[CHAR_IDS[c] for c in unseen],
            "unseen_occurrences":occurrences,"total_content_tokens":total,"unseen_fraction":occurrences/total}


def audit(data,output):
    manifest_path = data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prior = json.loads((data/"audit.json").read_text())
    if not prior.get("passed") or not prior.get("classical_solver_verified") or prior["manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Requires an audited parent corpus")
    counts,records = {},{}
    for filename,info in manifest["files"].items():
        if file_checksum(data/filename) != info["sha256"]:
            raise ValueError("Changed source corpus file")
        split = filename.removesuffix(".jsonl")
        counts[split] = {rep:Counter() for rep in REPRESENTATIONS}
        number = 0
        with (data/filename).open() as handle:
            for line in handle:
                row = json.loads(line)
                for rep in REPRESENTATIONS:
                    counts[split][rep].update(format_input(row,rep))
                number += 1
        if number != info["count"]:
            raise ValueError("Incorrect source record count")
        records[split] = number
        print(f"Character exposure: {split}, {number} records in three forms",flush=True)
    comparisons = {split:{source:{target:compare_characters(counts["train"][source],counts[split][target])
                                 for target in REPRESENTATIONS} for source in REPRESENTATIONS}
                   for split in counts if split != "train"}
    result = {"scope":"Full-corpus input-character coverage only; not neural accuracy or a causal syntax-exposure intervention",
              "dataset_manifest_sha256":file_checksum(manifest_path),"files":manifest["files"],"records":records,
              "content_character_counts":counts,"comparisons":comparisons,
              "audit_code_sha256":file_checksum(Path(__file__)),
              "renderer_sha256":file_checksum(Path(__file__).with_name("representations.py")),
              "interpretation":"Counts exclude BOS/EOS/PAD and use all training instances, not pilot subsets. Shared IDs do not imply all target-format characters were observed in source training. Coverage is not learned syntax competence; novel-character effects cannot by themselves explain an accuracy result. Training-only syntax-exposure and compute-matched controls remain necessary."}
    write_json(output,result)
    return result


def plot(report,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from .plots import save_svg
    matrix = np.array([[report["comparisons"]["test_iid"][s][t]["unseen_fraction"] for t in REPRESENTATIONS] for s in REPRESENTATIONS])
    fig,ax = plt.subplots(figsize=(7,5),constrained_layout=True)
    rendered = ax.imshow(matrix,vmin=0,vmax=max(.01,float(matrix.max())),cmap="Oranges")
    ax.set(xticks=range(3),yticks=range(3),xticklabels=REPRESENTATIONS,yticklabels=REPRESENTATIONS,
           xlabel="IID test rendering",ylabel="Training rendering",
           title="Source-unseen input-character occurrences\nCorpus exposure mismatch, NOT transfer accuracy")
    for i in range(3):
        for j in range(3):
            cell = report["comparisons"]["test_iid"][REPRESENTATIONS[i]][REPRESENTATIONS[j]]
            ax.text(j,i,f"{matrix[i,j]:.1%}\n{len(cell['source_unseen_characters'])} unseen chars",ha="center",va="center",
                    color="white" if matrix[i,j] > matrix.max()*.65 else "black")
    fig.colorbar(rendered,ax=ax,label="Fraction of content-character tokens")
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix(".png"),dpi=180)
    save_svg(fig,output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--output",type=Path,default=Path("models/character_exposure.json"))
    parser.add_argument("--plot",action="store_true")
    args = parser.parse_args()
    result = audit(args.data,args.output)
    if args.plot:
        plot(result,Path("graphs/character_exposure"))
