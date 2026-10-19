import math
import json
import tempfile
import unittest
from pathlib import Path

from models.paired_statistics import compare, exact_mcnemar, export, holm, read_predictions, validate_metrics


class PairedStatisticsTests(unittest.TestCase):
    def test_exact_against_enumeration(self):
        for n in range(31):
            for k in range(n+1):
                expected = min(1, 2*sum(math.comb(n,j) for j in range(min(k,n-k)+1))/2**n)
                self.assertAlmostEqual(exact_mcnemar(k,n-k),expected,places=12)
        self.assertAlmostEqual(exact_mcnemar(5000,5000),1)
        self.assertAlmostEqual(exact_mcnemar(0,10),2/1024)
        self.assertEqual(exact_mcnemar(0,10000),0)
        with self.assertRaises(ValueError):
            exact_mcnemar(-1,3)

    def test_holm_order_and_monotonicity(self):
        self.assertEqual(holm([]),[])
        actual = holm([0.04,0.001,0.03])
        for a,b in zip(actual,[0.06,0.003,0.06]):
            self.assertAlmostEqual(a,b)
        with self.assertRaises(ValueError):
            holm([float("nan")])

    def test_pairing_by_id_not_file_order(self):
        left = {str(i): {"target":"entailed", "prediction":pred} for i,pred in enumerate(
            ["entailed","entailed","unknown","unknown"])}
        right = {str(i): {"target":"entailed", "prediction":pred} for i,pred in reversed(list(enumerate(
            ["entailed","unknown","entailed","unknown"])))}
        result = compare(left,right)
        self.assertEqual([result[k] for k in ("both_correct","source_only_correct","target_only_correct","both_wrong")],[1,1,1,1])
        self.assertEqual(result["target_minus_source_accuracy"],0)
        with self.assertRaises(ValueError):
            compare(left,{"0":right["0"]})
        right["0"]["target"] = "unknown"
        with self.assertRaises(ValueError):
            compare(left,right)

    def test_prediction_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.jsonl"
            line = '{"id":"a","target":"unknown","prediction":"entailed"}\n'
            path.write_text(line)
            rows = read_predictions(path)
            validate_metrics(rows,{"count":1,"confusion":[[0,0,0],[0,0,0],[1,0,0]]})
            with self.assertRaises(ValueError):
                validate_metrics(rows,{"count":2,"confusion":[[0]*3]*3})
            path.write_text(line+line)
            with self.assertRaises(ValueError):
                read_predictions(path)

    def test_missing_predictions_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/"legacy"
            run.mkdir()
            (run/"results.json").write_text('{"status":"completed"}')
            report_path = root/"summary.json"
            self.assertEqual(export(root,report_path),[])
            report = json.loads(report_path.read_text())
            self.assertEqual(len(report["unavailable"][0]["missing_files"]),21)
            self.assertEqual(report["unavailable"][0]["run"],"legacy")


if __name__ == "__main__":
    unittest.main()
