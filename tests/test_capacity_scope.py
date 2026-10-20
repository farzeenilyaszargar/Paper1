import copy
import unittest
from unittest.mock import patch
from pathlib import Path

from models.architecture_provenance import (
    HISTORICAL_COMPUTATION, HISTORICAL_SOURCE, computation_digest,
    verify_transformer_provenance,
)
from models.suite import accelerator_trainer, jobs, wait_for_trainer
from models.transformer import PRIMARY_SIZES, TinyTransformer, expected_parameter_count


class CapacityScopeTests(unittest.TestCase):
    def test_requested_matrix_and_small_presets(self):
        self.assertEqual(PRIMARY_SIZES, ("10k", "50k", "100k", "500k", "1m"))
        self.assertEqual(TinyTransformer("50k").parameter_count, 50771)
        self.assertEqual(TinyTransformer("10k").parameter_count, 11491)
        planned = jobs(30)
        self.assertEqual(len({job["name"] for job in planned}), 45)
        self.assertEqual({job["size"] for job in planned}, set(PRIMARY_SIZES))
        self.assertTrue(all(job["batch_size"]*job["accumulation"] == 32 for job in planned))
        self.assertTrue(all(job["parameter_count"] == expected_parameter_count(job["size"]) for job in planned))

    def test_only_explicit_cpu_trainers_can_coexist(self):
        self.assertFalse(accelerator_trainer("python -m models.train --device cpu"))
        self.assertFalse(accelerator_trainer("python -m models.train --device=cpu"))
        self.assertTrue(accelerator_trainer("python -m models.train --device mps"))
        self.assertTrue(accelerator_trainer("python -m models.train"))
        self.assertFalse(accelerator_trainer("python -m models.suite --drain-pid 123"))
        self.assertFalse(accelerator_trainer("unrelated-app --title user's document"))
        self.assertTrue(accelerator_trainer("python -m models.train --output 'unclosed"))

    def test_historical_registry_requires_unchanged_computation_and_preset(self):
        path = Path(__file__).parent.parent/"models"/"transformer.py"
        source = path.read_text()
        self.assertEqual(computation_digest(source), HISTORICAL_COMPUTATION)
        config = {"size":"100k", "source_checksums":{"transformer.py":HISTORICAL_SOURCE},
                  "architecture":{"width":48, "layers":3, "heads":4, "ff_width":224}}
        verify_transformer_provenance(config)
        bad = copy.deepcopy(config)
        bad["architecture"]["heads"] = 2
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_transformer_provenance(bad)
        bad = copy.deepcopy(config)
        bad["source_checksums"]["transformer.py"] = "unapproved"
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_transformer_provenance(bad)
        with patch("models.architecture_provenance.computation_digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_transformer_provenance(config)

    def test_drain_checks_identity_and_waits_for_terminal_process(self):
        from subprocess import CompletedProcess
        live = CompletedProcess([],0,"S python -m models.train --device mps\n","")
        dead = CompletedProcess([],1,"","")
        with patch("models.suite.subprocess.run",side_effect=[live,live,dead]), patch("models.suite.time.sleep") as sleep:
            wait_for_trainer(123)
            self.assertEqual(sleep.call_count,2)
        reused = CompletedProcess([],0,"S python -m models.train --device mps --seed 2\n","")
        with patch("models.suite.subprocess.run",side_effect=[live,reused]), patch("models.suite.time.sleep"):
            with self.assertRaisesRegex(RuntimeError,"same accelerator"):
                wait_for_trainer(123)
        with patch("models.suite.subprocess.run",return_value=CompletedProcess([],0,"S python -m models.train --device cpu", "")):
            with self.assertRaisesRegex(RuntimeError,"same accelerator"):
                wait_for_trainer(123)
