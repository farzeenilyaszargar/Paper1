import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.calibration_selection import main, select


class CalibrationSelectionTests(unittest.TestCase):
    def test_cli_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            output = root / "selection.json"
            output.write_text("frozen dependency")
            with patch("sys.argv", ["selection", "--protocol", str(path), "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(output.read_text(), "frozen dependency")

    def fixture(self, root):
        protocol = json.loads(Path("models/calibration_1m_protocol.json").read_text())
        path = root / "protocol.json"
        path.write_text(json.dumps(protocol))
        (root / "manifest.json").write_text("software fixture")
        for tag, loss in (("001", 1.2), ("0003", 1.0), ("003", 1.0)):
            directory = root / f"calibration_1m_symbolic_seed101_lr{tag}"
            directory.mkdir()
            (directory / "results.json").write_text(json.dumps({"selected_validation":{"loss":loss}}))
            (directory / "history.jsonl").write_text("software fixture")
            (directory / "best.pt").write_bytes(b"not model weights")
        return path

    def test_partial_grid_never_selects_and_ties_keep_declared_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            with patch("models.calibration_selection.completed", side_effect=[True, False, False]):
                result = select(path, root, root)
                self.assertEqual(result["status"], "incomplete_grid_no_selection")
                self.assertNotIn("selected_learning_rate", result)
            with patch("models.calibration_selection.completed", return_value=True):
                result = select(path, root, root)
                self.assertEqual(result["selected_learning_rate"], .0003)

    def test_rejects_test_selection_and_propagates_invalid_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root)
            with patch("models.calibration_selection.completed", side_effect=ValueError("bad budget")):
                with self.assertRaisesRegex(ValueError, "bad budget"):
                    select(path, root, root)
            p = json.loads(path.read_text())
            p["test_supervision_for_selection"] = True
            path.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError, "validation only"):
                select(path, root, root)
