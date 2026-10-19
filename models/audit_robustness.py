"""Audit retained intervention predictions, metrics, pairing, and provenance.

This does not rerun neural inference or independently reconstruct modified texts.
Transformation semantics were solver-checked during probe construction.
"""

import argparse
import json
import math
from pathlib import Path

from synthetic_dataset_generator.generator import file_checksum, write_json
from .representations import LABELS

NAMES = ("original","rename","statement_order","irrelevant_components","query_negation","support_fact_ablation")


def verify_predictions(report, predictions, gold):
    if not gold or set(report["variants"]) != set(NAMES):
        raise ValueError("Missing gold examples or intervention categories")
    groups = {name:{} for name in NAMES}
    for row in predictions:
        name, identifier = row["variant"],row["id"]
        if name not in groups or identifier not in gold or identifier in groups[name]:
            raise ValueError("Invalid/duplicate variant-instance pair")
        if row["target"] not in LABELS or row["prediction"] not in LABELS:
            raise ValueError("Invalid prediction or target label")
        groups[name][identifier] = row
    original = groups["original"]
    decisive = {i for i,g in gold.items() if g != "unknown"}
    flip = {"entailed":"contradicted","contradicted":"entailed","unknown":"unknown"}
    for name,rows in groups.items():
        expected_ids = decisive if name == "support_fact_ablation" else set(gold)
        if set(rows) != expected_ids:
            raise ValueError(f"Incorrect applicability/IDs: {name}")
        for identifier,row in rows.items():
            expected = "unknown" if name == "support_fact_ablation" else flip[gold[identifier]] if name == "query_negation" else gold[identifier]
            if row["target"] != expected:
                raise ValueError(f"Incorrect intervention gold label: {name}/{identifier}")
        correct = {i for i,r in rows.items() if r["prediction"] == r["target"]}
        before_correct = {i for i in rows if original[i]["prediction"] == gold[i]}
        changed = {i for i,r in rows.items() if r["target"] != gold[i]}
        confusion = [[sum(r["target"] == a and r["prediction"] == b for r in rows.values()) for b in LABELS] for a in LABELS]
        expected = {"count":len(rows),"not_applicable":len(gold)-len(rows),"confusion":confusion,
                    "original_correct_on_applicable":len(before_correct),"jointly_correct":len(correct & before_correct),
                    "unchanged_predictions":sum(r["prediction"] == original[i]["prediction"] for i,r in rows.items()),
                    "gold_label_changed":len(changed),"correctly_followed_gold_change":len(changed & correct & before_correct)}
        metric = report["variants"][name]
        if any(metric.get(k) != v for k,v in expected.items()):
            raise ValueError(f"Saved metrics disagree with predictions: {name}")
        if rows:
            if not math.isclose(metric["accuracy"],len(correct)/len(rows),rel_tol=0,abs_tol=1e-12):
                raise ValueError(f"Incorrect accuracy: {name}")
        elif metric["accuracy"] is not None:
            raise ValueError("Inapplicable accuracy must be null")
    return sum(len(rows) for rows in groups.values())


def audit(directory, data):
    report_path = directory/"robustness.json"
    report = json.loads(report_path.read_text())
    checks = {directory/"results.json":"result_sha256",directory/"best.pt":"checkpoint_sha256",
              directory/"robustness.predictions.jsonl":"prediction_sha256",
              data/"manifest.json":"dataset_manifest_sha256",data/"test_iid.jsonl":"iid_sha256"}
    for path,key in checks.items():
        if file_checksum(path) != report[key]:
            raise ValueError(f"Provenance checksum mismatch: {path}")
    gold = {}
    with (data/"test_iid.jsonl").open() as handle:
        for index,line in enumerate(handle):
            if report["limit"] is not None and index >= report["limit"]:
                break
            row = json.loads(line)
            if row["id"] in gold:
                raise ValueError("Duplicate IID source ID")
            gold[row["id"]] = row["target"]
    with (directory/"robustness.predictions.jsonl").open() as handle:
        total = verify_predictions(report,(json.loads(line) for line in handle),gold)
    output = {"passed":True,"run":report["run"],"original_cases":len(gold),"prediction_records":total,
              "report_sha256":file_checksum(report_path),
              "probe_source_matches_current":file_checksum(Path(__file__).with_name("robustness.py")) == report["probe_code_sha256"],
              "scope":"Retained prediction counts, labels, pairings, confusions, accuracy, correctness/change metrics and file provenance; not an independent inference or transformed-text reconstruction. Input-token means and irredundant-cut sizes are not rederived here."}
    write_json(directory/"robustness_audit.json",output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory,args.data)))
