import errno
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.train import save_checkpoint,tensor_bytes,training_space_preflight


class CheckpointStorageTests(unittest.TestCase):
    def test_low_space_does_not_touch_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"last.pt"
            path.write_bytes(b"existing checkpoint sentinel")
            with patch("models.train.shutil.disk_usage",return_value=SimpleNamespace(free=0)):
                with self.assertRaises(OSError) as caught:
                    save_checkpoint({"model":torch.ones(10)},path)
                self.assertEqual(caught.exception.errno,errno.ENOSPC)
                self.assertEqual(path.read_bytes(),b"existing checkpoint sentinel")
                self.assertFalse(path.with_name("last.pt.tmp").exists())
                with self.assertRaises(OSError):
                    training_space_preflight(Path(directory)/"new_run",99075)
                self.assertFalse((Path(directory)/"new_run").exists())

    def test_tensor_size_and_successful_atomic_write(self):
        value = {"model":torch.ones(10),"optimizer":[torch.zeros(2,dtype=torch.float64)],"epoch":1}
        self.assertEqual(tensor_bytes(value),56)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"last.pt"
            with patch("models.train.shutil.disk_usage",return_value=SimpleNamespace(free=10*1024**3)):
                save_checkpoint(value,path)
            loaded = torch.load(path,weights_only=True)
            self.assertTrue(torch.equal(loaded["model"],value["model"]))
            self.assertFalse(path.with_name("last.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
