"""Instance-paired representation comparisons; no independent-sample tests.

Run with ``python -m models.paired_statistics`` after evaluations finish.
Holm correction is local to the 14 source-vs-target contrasts in each run;
it is NOT a correction across model sizes, seeds, or future interventions.
"""

import argparse
import json
import math
from pathlib import Path

from synthetic_dataset_generator.generator import SPLITS, file_checksum, write_json
from .representations import LABELS, REPRESENTATIONS


def exact_mcnemar(source_only, target_only):
    """Two-sided conditional Binomial(n_discordant, .5), without scipy."""
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0
           for x in (source_only, target_only)):
        raise ValueError("Discordance counts must be nonnegative integers")
    n = source_only + target_only
    k = min(source_only, target_only)
    if n == 0 or 2 * k == n:
        return 1.0
    # Sum relative to the largest term, avoiding underflow near n=10,000.
    log_peak = math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)
    relative = total = 1.0
    for j in range(k, 0, -1):
        relative *= j/(n-j+1)
        total += relative
    return min(1.0, math.exp(math.log(2)+log_peak+math.log(total)))


def holm(p_values):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("Invalid p-value")
    adjusted = [None] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        previous = max(previous, min(1.0, (len(p_values)-rank)*p_values[index]))
        adjusted[index] = previous
    return adjusted


def read_predictions(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in rows:
                raise ValueError(f"Missing/duplicate instance ID in {path}")
            if row["target"] not in LABELS or row["prediction"] not in LABELS:
                raise ValueError(f"Invalid prediction label in {path}")
            rows[identifier] = row
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    return rows


def validate_metrics(rows, metric):
    confusion = [[0]*3 for _ in LABELS]
    for row in rows.values():
        confusion[LABELS.index(row["target"])][LABELS.index(row["prediction"])] += 1
    if len(rows) != metric["count"] or confusion != metric["confusion"]:
        raise ValueError("Prediction file does not match saved evaluation metrics")


def compare(source, target):
    if not source:
        raise ValueError("Paired comparisons require nonempty predictions")
    if source.keys() != target.keys():
        raise ValueError("Paired comparisons require exactly identical instance IDs")
    table = {"both_correct": 0, "source_only_correct": 0,
             "target_only_correct": 0, "both_wrong": 0}
    for identifier, left in source.items():
        right = target[identifier]
        if left["target"] != right["target"]:
            raise ValueError("Paired instances have inconsistent gold targets")
        a, b = left["prediction"] == left["target"], right["prediction"] == right["target"]
        key = "both_correct" if a and b else "source_only_correct" if a else "target_only_correct" if b else "both_wrong"
        table[key] += 1
    return dict(table, count=len(source),
                target_minus_source_accuracy=(table["target_only_correct"]-table["source_only_correct"])/len(source),
                p_exact_mcnemar=exact_mcnemar(table["source_only_correct"], table["target_only_correct"]))


def analyze_run(directory):
    result = json.loads((directory/"results.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("Only completed evaluation runs can be compared")
    source = result["config"]["representation"]
    rows, hashes = [], {}
    for split in sorted(set(SPLITS)-{"train", "validation"}):
        left_path = directory/f"{source}_{split}.predictions.jsonl"
        left = read_predictions(left_path)
        validate_metrics(left, result["evaluations"][source][split])
        hashes[left_path.name] = file_checksum(left_path)
        for target in REPRESENTATIONS:
            if target == source:
                continue
            right_path = directory/f"{target}_{split}.predictions.jsonl"
            right = read_predictions(right_path)
            metric = result["evaluations"][target][split]
            validate_metrics(right, metric)
            hashes[right_path.name] = file_checksum(right_path)
            rows.append(dict(compare(left, right), split=split, source=source, target=target,
                             unseen_source_token_ids=metric.get("unseen_source_token_ids"),
                             unseen_all_training_token_ids=metric.get("unseen_all_training_token_ids")))
    for row, adjusted in zip(rows, holm([row["p_exact_mcnemar"] for row in rows])):
        row["p_holm_within_run"] = adjusted
    return {"run": directory.name, "scope": result["scope"],
            "source_learning_gate_passed": result.get("source_learning_gate_passed"),
            "result_sha256": file_checksum(directory/"results.json"),
            "prediction_sha256": hashes, "family_size": len(rows), "comparisons": rows}


def export(runs, output):
    reports, unavailable = [], []
    for directory in sorted(runs.iterdir()):
        path = directory/"results.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "completed":
            missing = [f"{rep}_{split}.predictions.jsonl" for rep in REPRESENTATIONS
                       for split in sorted(set(SPLITS)-{"train","validation"})
                       if not (directory/f"{rep}_{split}.predictions.jsonl").exists()]
            if missing:
                unavailable.append({"run":directory.name, "reason":"Missing retained instance predictions; aggregate accuracies cannot reconstruct pairs", "missing_files":missing})
                continue
            reports.append(analyze_run(directory))
    write_json(output, {"interpretation": "Exploratory paired accuracy differences; failed source gates do not establish procedure-transfer failure. P-values condition on this checkpoint and test set, not training seeds.",
                        "correction": "Holm within each run's 14 contrasts only; not across runs or the full final confirmatory family",
                        "unavailable": unavailable, "runs": reports})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("models/runs"))
    parser.add_argument("--output", type=Path, default=Path("graphs/paired_statistics.json"))
    args = parser.parse_args()
    reports = export(args.runs, args.output)
    print(f"Verified paired predictions for {len(reports)} completed runs")


if __name__ == "__main__":
    main()
