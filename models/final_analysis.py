"""Full-matrix paired statistics and descriptive behavioral criteria, never partial onsets."""

import argparse
import json
import math
from pathlib import Path

from synthetic_dataset_generator.generator import SPLITS, file_checksum, write_json
from .paired_statistics import compare, holm, read_predictions, validate_metrics
from .representations import LABELS, REPRESENTATIONS
from .suite import completed, jobs
from .transformer import PRIMARY_SIZES, expected_parameter_count

TESTS = tuple(sorted(set(SPLITS)-{"train","validation"}))


def metric_pass(metric, gold_counts):
    matrix = metric["confusion"]
    if len(matrix) != 3 or any(len(row) != 3 or any(type(n) is not int or n < 0 for n in row) for row in matrix):
        raise ValueError("Invalid three-class confusion matrix")
    totals = [sum(row) for row in matrix]
    if totals != [gold_counts[label] for label in LABELS] or any(n == 0 for n in totals):
        raise ValueError("Metrics disagree with source-corpus gold counts")
    count, correct = sum(totals), sum(matrix[i][i] for i in range(3))
    recalls = {label:matrix[i][i]/totals[i] for i,label in enumerate(LABELS)}
    if metric["count"] != count or metric["correct"] != correct or set(metric["recall"]) != set(LABELS):
        raise ValueError("Inconsistent metric counts/classes")
    quantities = [(metric["accuracy"],correct/count),(metric["balanced_accuracy"],sum(recalls.values())/3)]
    quantities += [(metric["recall"][label],recalls[label]) for label in LABELS]
    if any(not math.isfinite(value) or not math.isclose(value,expected,abs_tol=1e-12,rel_tol=0) for value,expected in quantities):
        raise ValueError("Metric accuracy/recall disagrees with confusion matrix")
    return correct/count >= .95 and all(value >= .90 for value in recalls.values())


def verified_run(directory, job, fingerprint, manifest, targets):
    if not completed(directory,job,fingerprint):
        raise ValueError("Final analysis requires complete primary conditions")
    result_path = directory/"results.json"
    result = json.loads(result_path.read_text())
    history_path = directory/"history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    if [row["epoch"] for row in history] != list(range(1,31)) or any(
        row["step"] != row["epoch"]*3125 or row["training_examples_seen"] != row["epoch"]*100000 for row in history):
        raise ValueError("Primary condition lacks the complete 30-epoch retained training budget")
    for row in history:
        metric_pass(row["validation"],manifest["files"]["validation.jsonl"]["labels"])
    for name in ("transformer.py","representations.py"):
        if result["config"]["source_checksums"][name] != file_checksum(Path(__file__).with_name(name)):
            raise ValueError("Primary architecture/renderer provenance differs from analysis code")
    if result["config"].get("warmup") is not None or result["config"].get("initialization_sha256") is not None:
        raise ValueError("Warmup treatments cannot enter the no-warmup primary family")
    source = job["representation"]
    validation_pass = metric_pass(result["selected_validation"],manifest["files"]["validation.jsonl"]["labels"])
    if result["source_learning_gate_passed"] != validation_pass:
        raise ValueError("Saved source-validation gate is inconsistent")
    contrasts, measurements, hashes = [],{},{}
    for split in TESTS:
        source_rows = read_predictions(directory/f"{source}_{split}.predictions.jsonl")
        for representation in REPRESENTATIONS:
            path = directory/f"{representation}_{split}.predictions.jsonl"
            rows = source_rows if representation == source else read_predictions(path)
            if rows.keys() != targets[split].keys() or any(row["target"] != targets[split][key] for key,row in rows.items()):
                raise ValueError("Predictions do not match the actual held-out IDs/gold labels")
            metric = result["evaluations"][representation][split]
            validate_metrics(rows,metric)
            passed = metric_pass(metric,manifest["files"][split+".jsonl"]["labels"])
            measurements[f"{representation}/{split}"] = {"criterion_passed":passed,"accuracy":metric["accuracy"],"recall":metric["recall"]}
            hashes[path.name] = file_checksum(path)
            if representation != source:
                contrasts.append(dict(compare(source_rows,rows),run=job["name"],source=source,target=representation,
                                      split=split,size=job["size"],seed=job["seed"]))
    source_pass = validation_pass and measurements[f"{source}/test_iid"]["criterion_passed"]
    return {"run":job["name"],"source":source,"size":job["size"],"seed":job["seed"],
            "source_gate_with_iid":source_pass,"measurements":measurements,"contrasts":contrasts,
            "result_sha256":file_checksum(result_path),"prediction_sha256":hashes,
            "history_sha256":file_checksum(history_path),"checkpoint_sha256":file_checksum(directory/"best.pt")}


def onset(passing):
    qualifying = [size for size in PRIMARY_SIZES if passing[size]]
    first = qualifying[0] if qualifying else None
    index = PRIMARY_SIZES.index(first) if first else None
    return {"tested_criteria":passing,"qualifying_sizes":qualifying,"smallest_tested_passing_size":first,
            "smallest_tested_passing_parameters":expected_parameter_count(first) if first else None,
            "adjacent_tested_bracket": [expected_parameter_count(PRIMARY_SIZES[index-1]),expected_parameter_count(first)] if index is not None and index > 0 else None,
            "nonmonotone_successes":any(passing[a] and not passing[b] for i,a in enumerate(PRIMARY_SIZES) for b in PRIMARY_SIZES[i+1:])}


def summarize(reports):
    expected = jobs(30)
    if len(reports) != len(expected) or {r["run"] for r in reports} != {job["name"] for job in expected}:
        raise ValueError("No final onsets/statistics until all 45 unique conditions are verified")
    by_key = {(r["size"],r["source"],r["seed"]):r for r in reports}
    required = {(j["size"],j["representation"],j["seed"]) for j in expected}
    if len(by_key) != 45 or set(by_key) != required:
        raise ValueError("Missing or duplicate size/source/seed condition")
    for report in reports:
        expected_pairs = {(report["source"],target,split) for target in REPRESENTATIONS if target != report["source"] for split in TESTS}
        actual_pairs = {(c["source"],c["target"],c["split"]) for c in report["contrasts"]}
        if len(report["contrasts"]) != 14 or actual_pairs != expected_pairs or any(
            any(c[key] != report[key] for key in ("run","size","seed")) for c in report["contrasts"]):
            raise ValueError("Incomplete or duplicate contrasts within a primary condition")
        if set(report["measurements"]) != {f"{rep}/{split}" for rep in REPRESENTATIONS for split in TESTS}:
            raise ValueError("Missing primary test criteria")
    contrasts = [dict(c,source_gate_with_iid=r["source_gate_with_iid"]) for r in reports for c in r["contrasts"]]
    if len(contrasts) != 630:
        raise ValueError("Expected the fixed 630-contrast baseline family")
    for row,adjusted in zip(contrasts,holm([c["p_exact_mcnemar"] for c in contrasts])):
        row["p_holm_full_baseline_family"] = adjusted
    criteria = {}
    for source in REPRESENTATIONS:
        criteria[f"{source}/source_learning"] = onset({size:all(by_key[size,source,seed]["source_gate_with_iid"] for seed in (1,2,3)) for size in PRIMARY_SIZES})
        for target in REPRESENTATIONS:
            for split in TESTS:
                criteria[f"{source}->{target}/{split}"] = onset({size:all(
                    by_key[size,source,seed]["source_gate_with_iid"] and
                    by_key[size,source,seed]["measurements"][f"{target}/{split}"]["criterion_passed"]
                    for seed in (1,2,3)) for size in PRIMARY_SIZES})
    return {"status":"complete_primary_matrix_analysis","family_size":630,"contrasts":contrasts,"behavioral_criteria":criteria,
            "runs":reports,"thresholds":{"accuracy":.95,"every_class_recall":.90,"required_seeds":[1,2,3]},
            "interpretation":"Descriptive smallest tested capacity under this architecture/data/budget, not a universal parameter threshold or proof of an internal reasoning algorithm. Every transfer/holdout criterion also requires validation plus source IID mastery in every seed. Brackets compare adjacent tested capacities; monotonicity is not assumed. Holm covers only the 630 predefined baseline paired contrasts, not calibration, warmups, robustness or all scientific claims. P-values are conditional on checkpoints and test records, not training-seed uncertainty. This family definition is documented before matrix completion, not claimed as external preregistration. Robustness/mechanistic validation remains separate."}


def export(data, runs, output):
    manifest_path = data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["train_count"] != 100000 or manifest["config"]["evaluation_count"] != 10000 or set(manifest["files"]) != {split+".jsonl" for split in SPLITS}:
        raise ValueError("Final baseline analysis requires the full primary corpus")
    fingerprint = file_checksum(manifest_path)
    audit = json.loads((data/"audit.json").read_text())
    if not audit.get("passed") or not audit.get("classical_solver_verified") or audit["manifest_sha256"] != fingerprint:
        raise ValueError("Requires the passing parent-corpus audit")
    planned = jobs(30)
    ready = [job for job in planned if completed(runs/job["name"],job,fingerprint)]
    missing = [job["name"] for job in planned if job not in ready]
    provenance = {"dataset_manifest_sha256":fingerprint,"analysis_code_sha256":file_checksum(Path(__file__)),
                  "dependency_code_sha256":{name:file_checksum(Path(__file__).with_name(name))
                                             for name in ("suite.py","paired_statistics.py","transformer.py","representations.py")}}
    if missing:
        report = dict(provenance,status="incomplete_matrix_no_final_statistics",aggregate_complete_conditions=len(ready),
                      required_conditions=45,planned_paired_family_size=630,missing_or_incomplete=missing,
                      interpretation="Aggregate/config coverage only. Retained-prediction validation is deferred until the entire matrix is present. No global corrected statistics, capacity onset, final conclusion or completed experiment is claimed.")
    else:
        targets = {}
        for name,info in manifest["files"].items():
            if file_checksum(data/name) != info["sha256"]:
                raise ValueError("Parent corpus changed since audit")
            split = name.removesuffix(".jsonl")
            if split in TESTS:
                targets[split] = {}
                count = 0
                with (data/name).open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        targets[split][row["id"]] = row["target"]
                        count += 1
                if len(targets[split]) != info["count"] or count != info["count"]:
                    raise ValueError("Incorrect or duplicate source IDs")
        report = dict(summarize([verified_run(runs/job["name"],job,fingerprint,manifest,targets) for job in planned]),**provenance)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--runs",type=Path,default=Path("models/runs"))
    parser.add_argument("--output",type=Path,default=Path("graphs/final_baseline_analysis.json"))
    args = parser.parse_args()
    print(export(args.data,args.runs,args.output)["status"])
