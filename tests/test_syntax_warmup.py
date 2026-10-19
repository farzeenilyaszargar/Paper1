import argparse
import copy
import tempfile
import unittest
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.representations import PAD,REPRESENTATIONS,VOCAB_SIZE,encode
from models.syntax_warmup import TrainingStream,initialization,language_logits,run
from models.plot_warmup import load_pair,export as plot_warmup
from models.train import run as supervise
from models.transformer import TinyTransformer
from synthetic_dataset_generator.audit import audit
from synthetic_dataset_generator.generator import Config,file_checksum,generate,write_json


class SyntaxWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stream_is_deterministic_and_ignores_labels_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"train.jsonl"
            row = {"target":"SECRET_ANSWER", "metadata":{"facts":["p000"],"query":"p001",
                "rules":[{"antecedent":[["p000"]],"conclusion":"p001"}],"derivation":"SECRET_PROOF"}}
            import json
            path.write_text("\n".join(json.dumps(row) for _ in range(6))+"\n")
            first = TrainingStream(path,"symbolic","all_formats",7,buffer_records=2)
            second = TrainingStream(path,"symbolic","all_formats",7,buffer_records=2)
            a,b = list(islice(first.tokens(),2000)),list(islice(second.tokens(),2000))
            self.assertEqual(a,b)
            self.assertTrue(all(first.form_counts[rep] > 0 for rep in REPRESENTATIONS))
            self.assertLessEqual(max(first.form_counts.values())-min(first.form_counts.values()),1)
            # Unsupported secret letters/underscores cannot enter this grammar.
            from models.representations import CHARACTERS
            rendered = "".join(CHARACTERS[t] for t in a if t < len(CHARACTERS))
            self.assertNotIn("SECRET",rendered)

    def test_tied_projection_preserves_parameter_count_and_classifier(self):
        model = TinyTransformer("10k")
        count = model.parameter_count
        head = copy.deepcopy(model.classifier.state_dict())
        tokens = torch.tensor([encode("p000 -> p001")])
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        logits = language_logits(model,tokens[:,:-1])
        self.assertEqual(logits.shape,(*tokens[:,:-1].shape,VOCAB_SIZE))
        F.cross_entropy(logits.flatten(0,1),tokens[:,1:].flatten()).backward()
        self.assertIsNone(model.classifier.weight.grad)
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        optimizer.step()
        self.assertEqual(count,model.parameter_count)
        self.assertTrue(torch.equal(head["weight"],model.classifier.weight))
        self.assertTrue(torch.equal(head["bias"],model.classifier.bias))
        self.assertTrue(torch.equal(model.embedding.weight[PAD],torch.zeros_like(model.embedding.weight[PAD])))

    def test_train_only_provenance_and_supervised_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            generate(Config(train_count=300,evaluation_count=3),data,progress=lambda _:None)
            write_json(data/"audit.json",audit(data,progress=lambda _:None))
            reports = []
            for treatment in ("source_only","all_formats"):
                args = argparse.Namespace(data=data,output=root/treatment,size="10k",source="symbolic",
                    treatment=treatment,seed=1,steps=10,batch_size=2,context=128,lr=.001,device="cpu",threads=1,resume=False)
                normal_open = Path.open

                def checked_open(path,*positional,**kwargs):
                    if path.suffix == ".jsonl" and path.name not in ("train.jsonl","history.jsonl"):
                        raise AssertionError("Warmup accessed non-training examples")
                    return normal_open(path,*positional,**kwargs)

                with patch.object(Path,"open",checked_open):
                    report = run(args)
                reports.append(report)
                self.assertTrue(report["all_requested_formats_observed"])
                state,loaded = initialization(args.output/"initialization.pt","10k",1,"symbolic",file_checksum(data/"manifest.json"))
                self.assertEqual(loaded["config"]["parameter_count"],TinyTransformer("10k").parameter_count)
                with self.assertRaisesRegex(ValueError,"mismatch"):
                    initialization(args.output/"initialization.pt","10k",2,"symbolic",file_checksum(data/"manifest.json"))
            self.assertEqual(reports[0]["config"]["token_budget"],reports[1]["config"]["token_budget"])
            self.assertEqual(reports[0]["stream_tokens_consumed"],reports[1]["stream_tokens_consumed"])
            arms = (root/"source_only",root/"all_formats")
            self.assertEqual(len(load_pair(arms)),2)
            plot_warmup(arms,root/"fixture_graphs")
            self.assertTrue((root/"fixture_graphs"/"warmup_10k_symbolic_seed1.png").exists())
            with self.assertRaisesRegex(ValueError,"one source-only"):
                load_pair((arms[0],arms[0]))
            history_path = arms[1]/"history.jsonl"
            original = history_path.read_text()
            import json
            altered = json.loads(original)
            altered["predicted_tokens"] += 1
            history_path.write_text(json.dumps(altered)+"\n")
            with self.assertRaisesRegex(ValueError,"token budget"):
                load_pair(arms)
            history_path.write_text(original)
            fit = argparse.Namespace(data=data,output=root/"supervised",size="10k",representation="symbolic",
                seed=1,epochs=1,batch_size=3,lr=.001,train_limit=3,eval_limit=3,device="cpu",threads=1,
                purpose="pilot",encoding="characters",accumulation=1,eval_batch_size=3,validation_only=True,
                initialization=root/"all_formats"/"initialization.pt",resume=False)
            result = supervise(fit)
            self.assertEqual(result["status"],"validation_only_completed")
            self.assertEqual(result["config"]["warmup"]["config"]["treatment"],"all_formats")
            self.assertEqual(result["evaluations"],{})

    def test_interrupted_warmup_resumes_identically_on_cpu(self):
        # Tiny software fixture only; no scientific result is published.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            generate(Config(train_count=300,evaluation_count=3),data,progress=lambda _:None)
            write_json(data/"audit.json",audit(data,progress=lambda _:None))
            args = argparse.Namespace(data=data,output=root/"full",size="10k",source="symbolic",
                treatment="source_only",seed=1,steps=101,batch_size=1,context=8,lr=.001,device="cpu",threads=1,resume=False)
            run(args)
            from models.syntax_warmup import save_checkpoint
            args.output = root/"resumed"

            def stop_after_checkpoint(value,path):
                save_checkpoint(value,path)
                if path.name == "last.pt" and value["step"] == 100:
                    raise RuntimeError("Test interruption after atomic checkpoint")

            with patch("models.syntax_warmup.save_checkpoint",side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,"Test interruption"):
                    run(args)
            args.resume = True
            run(args)
            a = torch.load(root/"full"/"initialization.pt",weights_only=True)["model"]
            b = torch.load(root/"resumed"/"initialization.pt",weights_only=True)["model"]
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))


if __name__ == "__main__":
    unittest.main()
