import copy
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

from models.auxiliary_queue import (
    argument, command_for, completed, drain, matching_config, planned_jobs,
    verify_protocol,
)
from models.transformer import expected_parameter_count
from models.representations import REPRESENTATIONS
from synthetic_dataset_generator.generator import SPLITS, file_checksum, write_json


def config_for(job):
    return dict({k:v for k,v in job.items() if k != "name"},
                dataset_manifest_sha256="software-fixture",
                train_examples=job["train_limit"] or 100000,
                validation_examples=job["eval_limit"] or 10000,
                parameter_count=expected_parameter_count(job["size"]),
                source_checksums={"transformer.py":file_checksum(Path("models/transformer.py"))})


class AuxiliaryQueueTests(unittest.TestCase):
    def test_lane_budgets_match_recorded_protocols(self):
        for lane,count in (("calibration_1m",3),("syntax_supervision",9)):
            jobs = planned_jobs(lane)
            self.assertEqual(len(jobs),count)
            self.assertEqual(len({j["name"] for j in jobs}),count)
            verify_protocol(lane)
            self.assertTrue(all(j["device"] == "cpu" and j["threads"] == 1 for j in jobs))
            self.assertTrue(all(j["batch_size"]*j["accumulation"] == 32 for j in jobs))
        jobs = planned_jobs("syntax_supervision")
        self.assertEqual([j["seed"] for j in jobs],[1,1,1,2,2,2,3,3,3])
        self.assertIsNone(jobs[0]["initialization"])
        self.assertIn("source_only",jobs[1]["initialization"])
        self.assertIn("all_formats",jobs[2]["initialization"])

    def test_completion_requires_budget_selection_and_no_calibration_tests(self):
        # Metadata/checkpoint placeholders exercise queue coverage only. No model
        # training or scientific prediction result is manufactured or exported.
        job = planned_jobs("calibration_1m")[0]
        config = config_for(job)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = dict(config=config,status="validation_only_completed",evaluations={},selected_validation={"loss":1.0})
            history = [dict(epoch=e,step=e*625,training_examples_seen=e*20000,
                            validation={"loss":1.0}) for e in range(1,11)]
            (root/"history.jsonl").write_text("\n".join(json.dumps(row) for row in history)+"\n")
            (root/"best.pt").write_bytes(b"software fixture, not model weights")
            write_json(root/"results.json",report)
            self.assertTrue(completed(root,job,"software-fixture"))
            changed = copy.deepcopy(report)
            changed["selected_validation"]["loss"] = 1.1
            write_json(root/"results.json",changed)
            with self.assertRaisesRegex(ValueError,"minimum validation"):
                completed(root,job,"software-fixture")
            changed = copy.deepcopy(report)
            changed["evaluations"] = {"symbolic":{}}
            write_json(root/"results.json",changed)
            with self.assertRaisesRegex(ValueError,"test evaluations"):
                completed(root,job,"software-fixture")
            write_json(root/"results.json",report)
            (root/"history.jsonl").write_text("\n".join(json.dumps(row) for row in history[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"retained training budget"):
                completed(root,job,"software-fixture")

    def test_configuration_and_commands_preserve_separation(self):
        job = planned_jobs("syntax_supervision")[0]
        config = config_for(job)
        self.assertTrue(matching_config(config,job,"software-fixture"))
        config["purpose"] = "experiment"
        self.assertFalse(matching_config(config,job,"software-fixture"))
        with tempfile.TemporaryDirectory() as folder, patch("models.auxiliary_queue.live_training_commands",return_value=[]):
            directory = Path(folder)/"new"
            command = command_for(job,directory,"software-fixture")
            self.assertEqual(argument(command,"--purpose"),"control")
            self.assertNotIn("--validation-only",command)
            directory.mkdir()
            write_json(directory/"config.json",config_for(job))
            with self.assertRaisesRegex(ValueError,"cannot safely resume"):
                command_for(job,directory,"software-fixture")
            (directory/"last.pt").write_bytes(b"software fixture")
            self.assertIn("--resume",command_for(job,directory,"software-fixture"))
            with patch("models.auxiliary_queue.live_training_commands",return_value=[["python","-m","models.train","--output",str(directory)]]):
                with self.assertRaisesRegex(RuntimeError,"already live"):
                    command_for(job,directory,"software-fixture")

    def test_drain_accepts_only_same_planned_cpu_command(self):
        job = planned_jobs("calibration_1m")[1]
        command = f"python -m models.train --device cpu --output models/runs/{job['name']}"
        live = CompletedProcess([],0,"S "+command+"\n","")
        dead = CompletedProcess([],1,"","")
        with patch("models.auxiliary_queue.subprocess.run",side_effect=[live,dead]), \
                patch("models.auxiliary_queue.time.sleep") as sleep, \
                patch("pathlib.Path.read_text",return_value=json.dumps(config_for(job))):
            drain(123,[job],"software-fixture")
            sleep.assert_called_once()
        wrong = CompletedProcess([],0,"S "+command.replace("cpu","mps"),"")
        with patch("models.auxiliary_queue.subprocess.run",return_value=wrong):
            with self.assertRaisesRegex(RuntimeError,"same planned trainer"):
                drain(123,[job],"software-fixture")

    def test_incomplete_reports_do_not_count_as_completion(self):
        job = planned_jobs("calibration_1m")[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(completed(root,job,"software-fixture"))
            write_json(root/"results.json",dict(config=config_for(job),status="evaluation_in_progress"))
            self.assertFalse(completed(root,job,"software-fixture"))

    def test_full_control_requires_all_test_groups_and_retained_files(self):
        job = planned_jobs("syntax_supervision")[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = set(SPLITS)-{"train","validation"}
            report = dict(config=config_for(job),status="completed",selected_validation={"loss":1.0},
                          evaluations={rep:{split:{"count":10000} for split in splits} for rep in REPRESENTATIONS})
            write_json(root/"results.json",report)
            history = [dict(epoch=e,step=e*3125,training_examples_seen=e*100000,
                            validation={"loss":1.0}) for e in range(1,31)]
            (root/"history.jsonl").write_text("\n".join(json.dumps(row) for row in history)+"\n")
            (root/"best.pt").write_bytes(b"software fixture")
            with self.assertRaisesRegex(ValueError,"retained prediction"):
                completed(root,job,"software-fixture")
            # Empty placeholders test file coverage, not statistical validation;
            # the scientific retained-prediction audit remains a separate step.
            for rep in REPRESENTATIONS:
                for split in splits:
                    (root/f"{rep}_{split}.predictions.jsonl").write_text("")
            self.assertTrue(completed(root,job,"software-fixture"))
            report["evaluations"]["symbolic"]["test_iid"]["count"] = 9
            write_json(root/"results.json",report)
            with self.assertRaisesRegex(ValueError,"21 full test"):
                completed(root,job,"software-fixture")


if __name__ == "__main__":
    unittest.main()
