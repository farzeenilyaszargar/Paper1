import unittest

from models.task_illustration import example
from models.representations import REPRESENTATIONS
from synthetic_dataset_generator.logic import Rule, classical_label, infer


class TaskIllustrationTests(unittest.TestCase):
    def test_solver_verified_lossless_two_hop_example(self):
        report = example()  # Also reparses and compares all three latent KBs.
        self.assertEqual(set(report["inputs"]),set(REPRESENTATIONS))
        self.assertEqual(report["minimum_rule_depth"],2)
        self.assertEqual(report["solver_label"],"entailed")
        meta = report["record"]["metadata"]
        rules = [Rule.from_dict(r) for r in meta["rules"]]
        self.assertEqual(infer(meta["facts"],rules)[0][meta["query"]],2)
        # The AND node must not be mistaken for two alternative unary rules.
        for missing_fact in meta["facts"]:
            remaining = [f for f in meta["facts"] if f != missing_fact]
            self.assertEqual(classical_label(remaining,rules,meta["query"]),"unknown")


if __name__ == "__main__":
    unittest.main()
