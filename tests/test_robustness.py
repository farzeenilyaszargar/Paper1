import json
import random
import unittest
from pathlib import Path

from models.parsing import parse
from models.representations import REPRESENTATIONS, format_input
from models.robustness import NAMES, variants
from synthetic_dataset_generator.generator import Config, LABELS, make_problem
from synthetic_dataset_generator.logic import Rule, atom, canonical_problem, classical_label


class RobustnessTests(unittest.TestCase):
    def test_all_labels_and_surface_forms(self):
        config = Config(train_count=300,evaluation_count=6)
        for label in LABELS:
            record = make_problem(config,"test_iid",label,random.Random(42))
            for rep in REPRESENTATIONS:
                text = format_input(record,rep)
                first = variants(text,rep,123)
                self.assertEqual(first,variants(text,rep,123))
                self.assertEqual(set(first),set(NAMES)-({"support_fact_ablation"} if label == "unknown" else set()))
                for name,example in first.items():
                    parsed = parse(format_input(example,rep),rep)["metadata"]
                    rules = [Rule.from_dict(r) for r in parsed["rules"]]
                    self.assertEqual(classical_label(parsed["facts"],rules,parsed["query"]),example["target"])
                self.assertEqual(first["original"]["target"],label)
                for name in ("rename","statement_order","irrelevant_components"):
                    self.assertEqual(first[name]["target"],label)
                if label != "unknown":
                    self.assertEqual(first["support_fact_ablation"]["target"],"unknown")
                    self.assertGreater(first["support_fact_ablation"]["removed_fact_count"],0)

    def test_ablation_can_cut_multiple_or_roots(self):
        text = "FACTS: p000; p001. RULES: (p000 | p001) -> p002. QUERY: p002."
        result = variants(text,"symbolic",123)
        self.assertEqual(result["support_fact_ablation"]["removed_fact_count"],2)
        self.assertEqual(result["support_fact_ablation"]["metadata"]["facts"],[])

    def test_statement_order_preserves_canonical_knowledge_base(self):
        record = json.loads(Path("datasets/example/records.jsonl").read_text().splitlines()[0])
        changed = variants(format_input(record,"symbolic"),"symbolic",7)
        canonical = []
        for name in ("original","statement_order"):
            meta = changed[name]["metadata"]
            canonical.append(canonical_problem(meta["facts"],[Rule.from_dict(r) for r in meta["rules"]],meta["query"]))
        self.assertEqual(*canonical)
        old = changed["original"]["metadata"]
        new = changed["irrelevant_components"]["metadata"]
        self.assertEqual(len(new["facts"])-len(old["facts"]),3)
        self.assertEqual(len(new["rules"])-len(old["rules"]),6)


if __name__ == "__main__":
    unittest.main()
