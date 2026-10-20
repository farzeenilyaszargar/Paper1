import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.train import load, run
from synthetic_dataset_generator.audit import audit
from synthetic_dataset_generator.generator import Config, generate, write_json


class CalibrationTests(unittest.TestCase):
    def test_no_warmup_controls_keep_separate_scope_and_resume_purpose(self):
        # Small software fixtures, not reported as scientific control results.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            generate(Config(name="unit",train_count=300,evaluation_count=3),data,progress=lambda _:None)
            write_json(data/"audit.json",audit(data,progress=lambda _:None))
            for limit in (3,None):
                args = argparse.Namespace(data=data,output=root/str(limit),size="10k",representation="symbolic",
                    seed=1,epochs=1,batch_size=3,lr=.001,train_limit=limit,eval_limit=3,device="cpu",
                    threads=1,purpose="control",encoding="characters",accumulation=1,eval_batch_size=3,
                    validation_only=False,resume=False)
                if limit is None:
                    args.eval_limit = None
                result = run(args)
                expected = "no-warmup control pilot" if limit else "no-warmup full control"
                self.assertEqual(result["scope"],expected)
                self.assertEqual(result["status"],"completed")
                self.assertEqual(sum(len(splits) for splits in result["evaluations"].values()),21)
                args.resume = True
                args.purpose = "experiment"
                with self.assertRaisesRegex(ValueError,"purpose mismatch"):
                    run(args)

    def test_validation_only_never_loads_or_evaluates_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            generate(Config(name="unit",train_count=300,evaluation_count=3),data,progress=lambda _:None)
            write_json(data/"audit.json",audit(data,progress=lambda _:None))
            args = argparse.Namespace(data=data,output=root/"run",size="10k",representation="symbolic",
                seed=1,epochs=1,batch_size=3,lr=.001,train_limit=3,eval_limit=3,device="cpu",
                threads=1,purpose="pilot",encoding="characters",accumulation=1,eval_batch_size=3,
                validation_only=True,resume=False)
            paths = []

            def tracked_load(path,*positional,**kwargs):
                paths.append(path.name)
                if path.name not in ("train.jsonl","validation.jsonl"):
                    raise AssertionError("Calibration attempted to load test supervision")
                return load(path,*positional,**kwargs)

            with patch("models.train.load",side_effect=tracked_load):
                result = run(args)
            self.assertEqual(paths,["train.jsonl","validation.jsonl"])
            self.assertEqual(result["status"],"validation_only_completed")
            self.assertEqual(result["evaluations"],{})
            self.assertEqual(result["selected_validation"]["count"],3)
            self.assertEqual(list(args.output.glob("*.predictions.jsonl")),[])
            args.resume = True
            args.validation_only = False
            with self.assertRaisesRegex(ValueError,"validation-only mode mismatch"):
                run(args)


if __name__ == "__main__":
    unittest.main()
