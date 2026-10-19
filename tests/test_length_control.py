import random
import copy
import tempfile
import unittest
from pathlib import Path

from models.length_control import build, make_pair, padding_plan
from models.audit_length_control import audit as audit_control, verify_member
from models.evaluate_length_control import run as evaluate_control
from models.plot_length_control import export as plot_control, verified_report
from models.transformer import TinyTransformer
import torch
from models.parsing import parse
from models.representations import REPRESENTATIONS, encode, format_input
from synthetic_dataset_generator.generator import Config, LABELS, generate, make_problem, reserved_pair, write_json, file_checksum
from synthetic_dataset_generator.audit import audit as audit_parent
from synthetic_dataset_generator.logic import Rule, classical_label, dependency_edges


class LengthControlTests(unittest.TestCase):
    def test_exact_lengths_labels_depths_and_no_reserved_edges(self):
        config = Config(train_count=300,evaluation_count=3)
        for label in LABELS:
            short = make_problem(config,"test_iid",label,random.Random(10))
            for depth in (3,4,5):
                long = make_problem(config,f"test_depth_{depth}",label,random.Random(20+depth))
                for rep in REPRESENTATIONS:
                    row = make_pair(short,long,rep,depth,142)
                    self.assertEqual(row,make_pair(short,long,rep,depth,142))
                    self.assertEqual(row["short"]["input_tokens"],row["long"]["input_tokens"])
                    self.assertEqual(row["short"]["proof_depth"],None if label == "unknown" else 2)
                    self.assertEqual(row["long"]["proof_depth"],None if label == "unknown" else depth)
                    for side in ("short","long"):
                        meta = parse(row[side]["input"],rep)["metadata"]
                        rules = [Rule.from_dict(r) for r in meta["rules"]]
                        self.assertEqual(classical_label(meta["facts"],rules,meta["query"]),label)
                        self.assertEqual(len(encode(row[side]["input"])),row[side]["input_tokens"])
                        self.assertTrue(all(not reserved_pair(42,a,b) for a,b in dependency_edges(rules)))
                        self.assertEqual(format_input({"metadata":meta},rep),row[side]["input"])

    def test_invalid_gaps_rejected(self):
        for rep in REPRESENTATIONS:
            self.assertEqual(padding_plan(0,rep),[])
            for gap in (-1,1,4095):
                with self.assertRaises(ValueError):
                    padding_plan(gap,rep)

    def test_member_audit_rejects_tampered_metadata(self):
        config = Config(train_count=300,evaluation_count=3)
        short = make_problem(config,"test_iid","entailed",random.Random(10))
        long = make_problem(config,"test_depth_5","entailed",random.Random(25))
        row = make_pair(short,long,"english",5,142)
        verify_member(row["short"],short,"english","entailed",2,42,512)
        changed = copy.deepcopy(row["short"])
        changed["added_rules"] += 1
        with self.assertRaisesRegex(ValueError,"distractor counts"):
            verify_member(changed,short,"english","entailed",2,42,512)

    def test_complete_small_corpus_is_reproducible_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"source"
            generate(Config(train_count=300,evaluation_count=3),data,progress=lambda _:None)
            write_json(data/"audit.json",audit_parent(data,progress=lambda _:None))
            first,second = root/"first",root/"second"
            a,b = build(data,first),build(data,second)
            self.assertEqual(a,b)
            self.assertEqual(len(a["files"]),9)
            for name in a["files"]:
                self.assertEqual((first/name).read_bytes(),(second/name).read_bytes())
            checked = audit_control(first,data)
            self.assertTrue(checked["passed"])
            self.assertEqual(checked["pair_count"],27)
            self.assertEqual(checked["logical_input_count"],54)
            # Untrained constant-head fixture: software validation, not paper data.
            model = TinyTransformer("10k")
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.zero_()
            run_directory = root/"model_fixture"
            run_directory.mkdir()
            torch.save(model.state_dict(),run_directory/"best.pt")
            write_json(run_directory/"results.json",{"status":"completed","source_learning_gate_passed":False,
                "config":{"size":"10k","parameter_count":model.parameter_count,"representation":"symbolic",
                          "encoding":"characters","dataset_manifest_sha256":file_checksum(data/"manifest.json")}})
            measured = evaluate_control(run_directory,first)
            self.assertEqual(measured["status"],"completed")
            for result in measured["depths"].values():
                self.assertEqual(result["metrics"]["short"]["count"],3)
                self.assertEqual(result["metrics"]["short"]["accuracy"],1/3)
                self.assertEqual(result["paired_comparison"]["long_minus_short_accuracy"],0)
            report_path = run_directory/"length_control_results.json"
            self.assertEqual(verified_report(report_path),measured)
            plot_control(report_path,root/"fixture_graphs")
            for suffix in ("png","svg","md"):
                self.assertTrue((root/"fixture_graphs"/f"model_fixture_length_control.{suffix}").is_file())
            altered = copy.deepcopy(measured)
            altered["status"] = "evaluation_in_progress"
            write_json(report_path,altered)
            with self.assertRaisesRegex(ValueError,"completed"):
                plot_control(report_path,root/"must_not_exist")
            self.assertFalse((root/"must_not_exist").exists())
            altered = copy.deepcopy(measured)
            altered["depths"]["3"]["metrics"]["short"]["accuracy"] = 1.0
            write_json(report_path,altered)
            with self.assertRaisesRegex(ValueError,"accuracy"):
                verified_report(report_path)
            altered = copy.deepcopy(measured)
            altered["depths"]["4"]["paired_comparison"]["p_holm_within_run_three_depths"] = 0
            write_json(report_path,altered)
            with self.assertRaisesRegex(ValueError,"multiplicity"):
                verified_report(report_path)
            write_json(report_path,measured)
            prediction_path = run_directory/"length_control_depth5_short.predictions.jsonl"
            prediction_path.write_text(prediction_path.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"checksum"):
                verified_report(report_path)


if __name__ == "__main__":
    unittest.main()
