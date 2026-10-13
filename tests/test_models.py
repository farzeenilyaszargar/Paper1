import copy
import json
import unittest
from pathlib import Path

import torch

from models.representations import LABELS, PAD, REPRESENTATIONS, encode, format_input
from models.train import batches, evaluate, optimizer_step
from models.suite import jobs
from models.transformer import SIZES, TinyTransformer, expected_parameter_count
from models.baselines import features


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.record = json.loads(Path("datasets/example/records.jsonl").read_text().splitlines()[0])

    def test_formats_never_read_answers_or_proofs(self):
        other = copy.deepcopy(self.record)
        other["target"] = "changed"
        for key in ("derivation", "proof_depth", "matched_depth", "family", "decisive_literal"):
            other["metadata"][key] = "DO NOT READ THIS"
        for representation in REPRESENTATIONS:
            self.assertEqual(format_input(self.record, representation), format_input(other, representation))
            self.assertNotIn("DO NOT READ", format_input(other, representation))
            self.assertTrue(len(encode(format_input(other, representation))) > 0)
        self.assertEqual(features(other), features(self.record))

    def test_causality_padding_and_gradient(self):
        torch.manual_seed(3)
        model = TinyTransformer("10k")
        tokens = torch.tensor([encode("p000 -> p001")])
        length = torch.tensor([tokens.shape[1]])
        model.eval()
        first = model(tokens, length)
        padded = torch.cat([tokens, torch.full((1, 8), PAD)], dim=1)
        self.assertTrue(torch.allclose(first, model(padded, length), atol=1e-6))
        first.sum().backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())

    def test_context_never_truncated(self):
        model = TinyTransformer("10k", max_context=4)
        with self.assertRaisesRegex(ValueError, "truncation"):
            model(torch.ones((1, 5), dtype=torch.long), torch.tensor([5]))

    def test_parameter_scaling(self):
        previous = 0
        for size in ("10k", "100k", "500k", "1m", "3m", "5m", "10m"):
            count = TinyTransformer(size).parameter_count
            self.assertEqual(count,expected_parameter_count(size))
            self.assertGreater(count, previous)
            previous = count

    def test_accumulation_matches_full_batch(self):
        torch.manual_seed(7)
        full, micro = TinyTransformer("10k"), TinyTransformer("10k")
        micro.load_state_dict(full.state_dict())
        samples = [(encode("p000"),i%3,str(i)) for i in range(5)]
        first = torch.optim.SGD(full.parameters(),lr=0.01)
        second = torch.optim.SGD(micro.parameters(),lr=0.01)
        optimizer_step(full,first,list(batches(samples,5)),"cpu")
        optimizer_step(micro,second,list(batches(samples,2)),"cpu")
        self.assertTrue(all(torch.allclose(a,b,atol=1e-6) for a,b in zip(full.parameters(),micro.parameters())))

    def test_full_queue_covers_five_sizes_three_formats_three_seeds(self):
        planned = jobs(30)
        self.assertEqual(len(planned),45)
        self.assertEqual(len({j["name"] for j in planned}),45)
        self.assertTrue(all(j["batch_size"]*j["accumulation"] == 32 for j in planned))

    def test_batches_keep_identifiers_and_lengths(self):
        samples = [(encode("p000"), 0, "a"), (encode("p000 -> p001"), 1, "b")]
        tokens, lengths, targets, ids = next(batches(samples, 2))
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(targets.tolist(), [0, 1])
        self.assertEqual(lengths.tolist(), [len(s[0]) for s in samples])
        self.assertTrue(torch.all(tokens[0, lengths[0]:] == PAD))
        self.assertEqual(tokens.shape[1],128)


if __name__ == "__main__":
    unittest.main()
