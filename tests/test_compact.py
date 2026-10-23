import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from models.compact import command, passes, planned
from synthetic_dataset_generator.generator import Config, make_problem


class CompactTests(unittest.TestCase):
    def test_budget_and_gated_stages(self):
        p = json.loads(Path("models/compact_protocol.json").read_text())
        self.assertEqual(p["screen_sizes"], ["10k", "50k", "100k", "500k"])
        self.assertEqual(len(planned("pilot", p)), 1)
        with patch("models.compact.report_for", return_value={"config":{"size":"100k"}, "status":"validation_only_completed", "selected_validation":{"count":1000,"accuracy":.34}}):
            with self.assertRaisesRegex(ValueError, "Pilot failed"):
                planned("screen", p)
        metric = {"count":1000,"accuracy":.96,"recall":dict(entailed=.96,contradicted=.95,unknown=.97)}
        self.assertTrue(passes(metric))
        with patch("models.compact.report_for", return_value={"config":{"size":"100k"}, "status":"validation_only_completed", "selected_validation":metric}):
            jobs = planned("screen", p)
            self.assertEqual(len(jobs), 4)
            self.assertNotIn("--validation-only", command(jobs[0], p))
        self.assertIn("--validation-only", command(planned("pilot", p)[0], p))

    def test_compact_is_shorter_and_preserves_logic(self):
        c = Config.load(Path("synthetic_dataset_generator/configs/compact.json"))
        lengths = []
        for config in (Config(), c):
            records = [make_problem(config, "train", label, random.Random(i))
                       for i in range(10) for label in ("entailed", "contradicted", "unknown")]
            lengths.append(sum(len(r["input"]) for r in records))
            self.assertTrue(all(r["metadata"]["proof_depth"] == 2 for r in records if r["target"] != "unknown"))
        self.assertLess(lengths[1], lengths[0])
