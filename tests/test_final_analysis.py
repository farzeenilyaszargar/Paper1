import copy
import json
import tempfile
import unittest
from pathlib import Path

from models.final_analysis import TESTS, export, metric_pass, onset, summarize, verified_run
from models.paired_statistics import exact_mcnemar
from models.representations import LABELS, REPRESENTATIONS
from models.suite import jobs
from synthetic_dataset_generator.generator import SPLITS, file_checksum, write_json


def metric(matrix):
    totals = [sum(row) for row in matrix]
    count = sum(totals)
    correct = sum(matrix[i][i] for i in range(3))
    recall = {label:matrix[i][i]/totals[i] for i,label in enumerate(LABELS)}
    return {"confusion":matrix,"count":count,"correct":correct,"accuracy":correct/count,
            "recall":recall,"balanced_accuracy":sum(recall.values())/3}


def software_reports():
    # Synthetic pure-function fixtures only, never exported as scientific data.
    reports = []
    for job in jobs(30):
        source = job["representation"]
        passed = job["size"] in ("500k","1m") and not (job["size"] == "500k" and job["seed"] == 3)
        contrasts = [dict(run=job["name"],source=source,target=target,split=split,size=job["size"],seed=job["seed"],p_exact_mcnemar=1.0)
                     for target in REPRESENTATIONS if target != source for split in TESTS]
        reports.append({"run":job["name"],"source":source,"size":job["size"],"seed":job["seed"],
                        "source_gate_with_iid":passed,"contrasts":contrasts,
                        "measurements":{f"{rep}/{split}":{"criterion_passed":True} for rep in REPRESENTATIONS for split in TESTS}})
    return reports


class FinalAnalysisTests(unittest.TestCase):
    def test_confusion_based_criteria_reject_inconsistent_or_nonfinite_metrics(self):
        counts = dict.fromkeys(LABELS,100)
        actual = metric([[95,5,0],[0,95,5],[5,0,95]])
        self.assertTrue(metric_pass(actual,counts))
        self.assertFalse(metric_pass(metric([[100,0,0],[100,0,0],[100,0,0]]),counts))
        for value in (.96,float("nan")):
            changed = copy.deepcopy(actual)
            changed["accuracy"] = value
            with self.assertRaisesRegex(ValueError,"disagrees"):
                metric_pass(changed,counts)
        with self.assertRaisesRegex(ValueError,"gold counts"):
            metric_pass(actual,dict.fromkeys(LABELS,101))

    def test_descriptive_onset_does_not_assume_monotonicity(self):
        result = onset({"10k":False,"50k":False,"100k":False,"500k":True,"1m":False})
        self.assertEqual(result["smallest_tested_passing_parameters"],457251)
        self.assertEqual(result["adjacent_tested_bracket"],[99075,457251])
        self.assertTrue(result["nonmonotone_successes"])
        self.assertIsNone(onset(dict.fromkeys(("10k","50k","100k","500k","1m"),False))["smallest_tested_passing_size"])

    def test_all_630_contrasts_and_all_three_seeds_are_required(self):
        reports = software_reports()
        p = exact_mcnemar(0,15)
        reports[0]["contrasts"][0]["p_exact_mcnemar"] = p
        result = summarize(reports)
        self.assertEqual(result["family_size"],630)
        self.assertAlmostEqual(result["contrasts"][0]["p_holm_full_baseline_family"],630*p)
        self.assertEqual(len(result["behavioral_criteria"]),66)
        self.assertEqual(result["behavioral_criteria"]["symbolic/source_learning"]["smallest_tested_passing_size"],"1m")
        # A passing holdout cannot rescue failed source mastery in seed 3.
        self.assertEqual(result["behavioral_criteria"]["symbolic->english/test_symbols"]["smallest_tested_passing_size"],"1m")
        with self.assertRaisesRegex(ValueError,"45 unique"):
            summarize(reports[:-1])
        altered = copy.deepcopy(reports)
        altered[0]["contrasts"][0] = altered[0]["contrasts"][1]
        with self.assertRaisesRegex(ValueError,"duplicate contrasts"):
            summarize(altered)

    def test_partial_matrix_exports_coverage_not_statistics_or_onsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"config":{"train_count":100000,"evaluation_count":10000},"files":{s+".jsonl":{} for s in SPLITS}}
            write_json(root/"manifest.json",manifest)
            write_json(root/"audit.json",{"passed":True,"classical_solver_verified":True,"manifest_sha256":file_checksum(root/"manifest.json")})
            result = export(root,root/"absent_runs",root/"coverage.json")
            self.assertEqual(result["status"],"incomplete_matrix_no_final_statistics")
            self.assertEqual(result["aggregate_complete_conditions"],0)
            self.assertEqual(len(result["missing_or_incomplete"]),45)
            self.assertNotIn("contrasts",result)
            self.assertNotIn("behavioral_criteria",result)

    def test_retained_predictions_match_actual_split_ids_and_golds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = jobs(30)[0]
            golds = {label:n for label,n in zip(LABELS,(3334,3333,3333))}
            score = metric([[3334,0,0],[3333,0,0],[3333,0,0]])
            config = dict(job,lr=.001,train_examples=100000,validation_examples=10000,dataset_manifest_sha256="fixture")
            config["source_checksums"] = {name:file_checksum(Path(__file__).parent.parent/"models"/name) for name in ("transformer.py","representations.py")}
            result = {"status":"completed","scope":"full corpus","config":config,"selected_validation":score,
                      "source_learning_gate_passed":False,"evaluations":{rep:{split:score for split in TESTS} for rep in REPRESENTATIONS}}
            write_json(root/"results.json",result)
            history = [{"epoch":epoch,"step":epoch*3125,"training_examples_seen":epoch*100000,"validation":score} for epoch in range(1,31)]
            (root/"history.jsonl").write_text("\n".join(json.dumps(row) for row in history)+"\n")
            (root/"best.pt").write_bytes(b"software fixture: only checkpoint hashing is exercised")
            targets = {}
            for split in TESTS:
                targets[split] = {}
                rows = []
                for label,count in golds.items():
                    for index in range(count):
                        identifier = f"{split}-{label}-{index}"
                        targets[split][identifier] = label
                        rows.append(json.dumps({"id":identifier,"target":label,"prediction":"entailed"}))
                text = "\n".join(rows)+"\n"
                for rep in REPRESENTATIONS:
                    (root/f"{rep}_{split}.predictions.jsonl").write_text(text)
            manifest = {"files":{name+".jsonl":{"labels":golds} for name in (*TESTS,"validation")}}
            checked = verified_run(root,job,"fixture",manifest,targets)
            self.assertFalse(checked["source_gate_with_iid"])
            self.assertEqual(len(checked["contrasts"]),14)
            altered = copy.deepcopy(targets)
            first = next(iter(altered["test_iid"]))
            altered["test_iid"][first] = "unknown"
            with self.assertRaisesRegex(ValueError,"actual held-out"):
                verified_run(root,job,"fixture",manifest,altered)
            (root/"history.jsonl").write_text("\n".join(json.dumps(row) for row in history[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"30-epoch"):
                verified_run(root,job,"fixture",manifest,targets)


if __name__ == "__main__":
    unittest.main()
