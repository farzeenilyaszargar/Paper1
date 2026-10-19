import copy
import unittest

from models.audit_robustness import NAMES, verify_predictions


class RobustnessAuditTests(unittest.TestCase):
    def fixture(self):
        gold = {"a":"entailed","b":"contradicted","c":"unknown"}
        predictions = []
        metrics = {}
        flip = {"entailed":"contradicted","contradicted":"entailed","unknown":"unknown"}
        for name in NAMES:
            ablated = name == "support_fact_ablation"
            for identifier,label in gold.items():
                if ablated and label == "unknown":
                    continue
                predictions.append({"id":identifier,"variant":name,"prediction":"contradicted",
                    "target":"unknown" if ablated else flip[label] if name == "query_negation" else label})
            metrics[name] = {"count":2 if ablated else 3,"not_applicable":1 if ablated else 0,
                "accuracy":0 if ablated else 1/3,"original_correct_on_applicable":1,
                "jointly_correct":0 if name in ("query_negation","support_fact_ablation") else 1,
                "unchanged_predictions":2 if ablated else 3,"gold_label_changed":2 if name in ("query_negation","support_fact_ablation") else 0,
                "correctly_followed_gold_change":0,
                "confusion":[[0,0,0],[0,0,0],[0,2,0]] if ablated else [[0,1,0],[0,1,0],[0,1,0]]}
        return {"variants":metrics},predictions,gold

    def test_complete_pairing_and_metrics(self):
        report,rows,gold = self.fixture()
        self.assertEqual(verify_predictions(report,reversed(rows),gold),17)

    def test_reject_corruptions(self):
        report,rows,gold = self.fixture()
        for changed in (rows+rows[:1],rows[1:],rows+[{"id":"extra","variant":"original","target":"unknown","prediction":"unknown"}]):
            with self.assertRaises(ValueError):
                verify_predictions(report,changed,gold)
        changed = copy.deepcopy(rows)
        changed[-1]["target"] = "entailed"
        with self.assertRaises(ValueError):
            verify_predictions(report,changed,gold)
        report["variants"]["query_negation"]["jointly_correct"] = 1
        with self.assertRaises(ValueError):
            verify_predictions(report,rows,gold)


if __name__ == "__main__":
    unittest.main()
