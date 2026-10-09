import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from synthetic_dataset_generator.audit import audit
from synthetic_dataset_generator.generator import (Config, LABELS, SPLITS, digest,
    file_checksum, generate, make_problem, write_json)
from synthetic_dataset_generator.logic import (Rule, canonical_problem, classical_label,
    dependency_edges, forward_label, has_reconvergence, infer, negate)
from synthetic_dataset_generator.encoding import encode, decode
from synthetic_dataset_generator.verification import SolverVerifier


class LogicTests(unittest.TestCase):
    def test_shared_encoding(self):
        for name in ("p000", "p511", "p512", "p767"):
            self.assertEqual(decode(encode(name)), name)
        with self.assertRaises(ValueError):
            encode("unseen alphabet!")
        with self.assertRaises(ValueError):
            decode([-1])

    def test_two_hops_and_three_labels(self):
        rules = [Rule((("p000",),), "p001"), Rule((("p001",),), "p002")]
        depths, _ = infer(["p000"], rules)
        self.assertEqual(depths["p002"], 2)
        for query, expected in [("p002", "entailed"), ("~p002", "contradicted"), ("p003", "unknown")]:
            self.assertEqual(forward_label(depths, query), expected)
            self.assertEqual(classical_label(["p000"], rules, query), expected)

    def test_and_or_and_negated_literals(self):
        rules = [Rule((("p000", "~p001"), ("p002",)), "~p003")]
        for facts, expected in [(["p000"], "unknown"), (["p000", "~p001"], "entailed"), (["p002"], "entailed")]:
            depths, _ = infer(facts, rules)
            self.assertEqual(forward_label(depths, "~p003"), expected)
            self.assertEqual(classical_label(facts, rules, "~p003"), expected)

    def test_shortest_proof_and_unfounded_cycle(self):
        rules = [Rule((("p000",),), "p001"), Rule((("p001",),), "p002"),
                 Rule((("p000",),), "p002"), Rule((("p003",),), "p004"), Rule((("p004",),), "p003")]
        depths, _ = infer(["p000"], list(reversed(rules)))
        self.assertEqual(depths["p002"], 1)
        self.assertNotIn("p003", depths)

    def test_classical_contraposition_is_independent(self):
        rules = [Rule((("p000",),), "p001")]
        depths, _ = infer(["~p001"], rules)
        self.assertEqual(forward_label(depths, "~p000"), "unknown")
        self.assertEqual(classical_label(["~p001"], rules, "~p000"), "entailed")

    def test_inconsistency_rejected(self):
        with self.assertRaises(ValueError):
            classical_label(["p000", "~p000"], [], "p001")

    def test_normalization_and_absorption(self):
        a = canonical_problem(["p001", "p000"], [Rule((("p000",), ("p000", "p001")), "p002")], "p002")
        b = canonical_problem(["p000", "p001", "p000"], [Rule((("p000",),), "p002")], "p002")
        self.assertEqual(digest(a), digest(b))

    def test_graph_shapes(self):
        self.assertFalse(has_reconvergence({("a", "c"), ("b", "c"), ("c", "d")}))
        self.assertTrue(has_reconvergence({("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}))
        with self.assertRaises(ValueError):
            has_reconvergence({("a", "b"), ("b", "a")})


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = Config(name="test", train_count=300, evaluation_count=6)
        cls.first, cls.second = cls.root / "first", cls.root / "second"
        generate(cls.config, cls.first, progress=lambda _: None)
        generate(cls.config, cls.second, progress=lambda _: None, workers=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_byte_identical_and_fully_audited(self):
        for path in self.first.iterdir():
            self.assertEqual(path.read_bytes(), (self.second / path.name).read_bytes(), path.name)
        report = audit(self.first, progress=lambda _: None, workers=2)
        self.assertTrue(report["classical_solver_verified"])
        self.assertEqual(report["total_examples"], 348)

    def test_seed_changes_problem(self):
        first = make_problem(self.config, "train", "entailed", random.Random(1))
        second = make_problem(self.config, "train", "entailed", random.Random(2))
        self.assertNotEqual(first["id"], second["id"])

    def test_counts_not_label_shortcuts(self):
        for split in SPLITS:
            records = [make_problem(self.config, split, label, random.Random(77)) for label in LABELS]
            shapes = [(len(r["metadata"]["facts"]), len(r["metadata"]["rules"]),
                       len(r["metadata"]["symbols"])) for r in records]
            self.assertEqual(len(set(shapes)), 1, split)
            for record in records:
                query = record["metadata"]["query"]
                conclusions = {r["conclusion"] for r in record["metadata"]["rules"]}
                self.assertTrue({query, negate(query)} <= conclusions)
            self.assertEqual(len({r["metadata"]["query"] for r in records}), 1)

    def test_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            generate(self.config, self.first, progress=lambda _: None)

    def test_parallel_solver_failure_is_propagated(self):
        with self.assertRaisesRegex(ValueError, "Classical solver disagrees"):
            with SolverVerifier(2) as verifier:
                verifier.check(["p000"], [], "p000", "unknown")

    def test_attempt_limit_leaves_no_manifest(self):
        from unittest.mock import patch
        output = self.root / "exhausted"
        with patch("synthetic_dataset_generator.generator.make_problem", side_effect=ValueError("Reject candidate")):
            with self.assertRaisesRegex(RuntimeError, "constraints were not relaxed"):
                generate(replace(self.config, max_attempts_per_example=2), output, progress=lambda _: None)
        self.assertFalse((output / "manifest.json").exists())

    def test_audit_detects_tampering_beyond_checksums(self):
        import shutil
        broken = self.root / "broken"
        shutil.copytree(self.first, broken)
        path = broken / "train.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["metadata"]["proof_depth"] = 99
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        manifest = json.loads((broken / "manifest.json").read_text())
        manifest["files"][path.name]["sha256"] = file_checksum(path)
        write_json(broken / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "proof depth"):
            audit(broken, progress=lambda _: None)

    def test_invalid_configs(self):
        for overrides in ({"train_count": 0}, {"evaluation_count": -1}, {"name": "../bad"},
                          {"seed": True}, {"heldout_symbols": 10}):
            with self.assertRaises(ValueError):
                replace(self.config, **overrides)


if __name__ == "__main__":
    unittest.main()
