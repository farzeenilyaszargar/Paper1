"""Evaluate a fixed raw checkpoint on audited, equal-token-length depth pairs."""

import argparse
import json
from pathlib import Path

import torch

from synthetic_dataset_generator.generator import file_checksum,write_json
from .paired_statistics import compare,holm,read_predictions,validate_metrics
from .representations import LABELS,encode
from .train import evaluate
from .transformer import TinyTransformer


def run(directory,controls,batch_size=4,threads=1):
    torch.set_num_threads(threads)
    output = directory/"length_control_results.json"
    if output.exists():
        raise FileExistsError("Preserve the existing control report before another evaluation")
    baseline_path = directory/"results.json"
    baseline = json.loads(baseline_path.read_text())
    config = baseline["config"]
    if baseline["status"] != "completed" or config.get("encoding","characters") != "characters":
        raise ValueError("Requires a completed raw-character checkpoint")
    manifest_path = controls/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((controls/"audit.json").read_text())
    if not audit.get("passed") or not audit.get("classical_solver_verified") or audit["manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Requires a passing full control-corpus audit")
    if config["dataset_manifest_sha256"] != manifest["parent_manifest_sha256"]:
        raise ValueError("Control source corpus differs from model's experiment")
    for name,info in manifest["files"].items():
        if file_checksum(controls/name) != info["sha256"]:
            raise ValueError("Control corpus changed since audit")
    checkpoint = directory/"best.pt"
    model = TinyTransformer(config["size"])
    if model.parameter_count != config["parameter_count"]:
        raise ValueError("Checkpoint architecture does not match current model")
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    representation = config["representation"]
    comparisons = []
    report = {"status":"evaluation_in_progress","run":directory.name,"control_scope":manifest["scope"],
              "representation":representation,"source_learning_gate_passed":baseline.get("source_learning_gate_passed"),
              "result_sha256":file_checksum(baseline_path),"checkpoint_sha256":file_checksum(checkpoint),
              "control_manifest_sha256":file_checksum(manifest_path),"device":"cpu","threads":threads,"batch_size":batch_size,
              "code_sha256":file_checksum(Path(__file__)),"depths":{},
              "interpretation":"Diagnostic same-format comparisons with equal length and label but different core KBs. Added distractor, graph and fact/operator-count cues are not fully controlled. Not a causal isolation of depth or a transfer result. Exact paired p-values are checkpoint/test-conditional; Holm covers only this run's three depth comparisons, not other seeds/sizes/controls."}
    for depth in (3,4,5):
        samples = {"short":[],"long":[]}
        path = controls/f"depth_{depth}_{representation}.jsonl"
        with path.open() as handle:
            for line in handle:
                pair = json.loads(line)
                for side in samples:
                    tokens = encode(pair[side]["input"])
                    if len(tokens) != pair[side]["input_tokens"]:
                        raise ValueError("Actual token count differs from audited count")
                    samples[side].append((tokens,LABELS.index(pair["target"]),pair["id"]))
        metrics,predictions,hashes = {},{},{}
        for side in samples:
            prediction_path = directory/f"length_control_depth{depth}_{side}.predictions.jsonl"
            metrics[side] = evaluate(model,samples[side],batch_size,"cpu",prediction_path)
            predictions[side] = read_predictions(prediction_path)
            validate_metrics(predictions[side],metrics[side])
            hashes[side] = file_checksum(prediction_path)
        comparison = compare(predictions["short"],predictions["long"])
        comparison["long_minus_short_accuracy"] = comparison.pop("target_minus_source_accuracy")
        comparisons.append(comparison)
        report["depths"][str(depth)] = {"short_depth":2,"long_depth":depth,"metrics":metrics,
            "paired_comparison":comparison,"prediction_sha256":hashes,
            "note":"Unknown examples use scaffold depths and have no decisive proof"}
        write_json(output,report)
        print(f"Length control 2 vs {depth}: {metrics['short']['accuracy']:.4f} vs {metrics['long']['accuracy']:.4f}",flush=True)
    for comparison,p_adjusted in zip(comparisons,holm([c["p_exact_mcnemar"] for c in comparisons])):
        comparison["p_holm_within_run_three_depths"] = p_adjusted
    report["status"] = "completed"
    write_json(output,report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    parser.add_argument("--controls",type=Path,default=Path("datasets/length_matched"))
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--threads",type=int,default=1)
    args = parser.parse_args()
    if min(args.batch_size,args.threads) < 1:
        parser.error("Batch size and threads must be positive")
    run(args.run_directory,args.controls,args.batch_size,args.threads)
