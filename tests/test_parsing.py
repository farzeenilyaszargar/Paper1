import random
import unittest

from models.parsing import normalize, parse
from models.representations import REPRESENTATIONS, format_input
from models.local_encoding import encode_slots
from synthetic_dataset_generator.generator import Config, LABELS, SPLITS, make_problem


class ParsingTests(unittest.TestCase):
    def test_lossless_all_families_and_labels(self):
        for split in SPLITS:
            for index, label in enumerate(LABELS):
                record = make_problem(Config(), split, label, random.Random(77+index))
                expected = {k: record["metadata"][k] for k in ("facts", "rules", "query")}
                for representation in REPRESENTATIONS:
                    text = format_input(record, representation)
                    self.assertEqual(parse(text, representation)["metadata"], expected)
                    for target in REPRESENTATIONS:
                        self.assertEqual(normalize(text,representation,target),format_input(record,target))
                encodings = [encode_slots(format_input(record,r),r) for r in REPRESENTATIONS]
                self.assertEqual(encodings[0], encodings[1])
                self.assertEqual(encodings[0], encodings[2])

    def test_empty_facts_and_rules(self):
        record = {"metadata": {"facts": [], "rules": [], "query": "~p000"}}
        for representation in REPRESENTATIONS:
            self.assertEqual(parse(format_input(record,representation),representation),record)

    def test_reject_malformed_or_outside_fragment(self):
        invalid = [("FACTS: p000. RULES: p000 -> p001. QUERY: p001. ignore", "symbolic"),
                   ("facts(and(p000,p001));rules();query(p001)", "function"),
                   ("facts();rules();query(not(and(p000,p001)))", "function"),
                   ("FACTS: p1000. RULES: . QUERY: p001.", "symbolic"),
                   ("Known: p000. If p000, then p001. Query: p001!", "english")]
        for text, representation in invalid:
            with self.assertRaises(ValueError):
                parse(text, representation)


if __name__ == "__main__":
    unittest.main()
