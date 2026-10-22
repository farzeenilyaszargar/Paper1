"""Audit a declared calibration grid without consulting test data or training."""

import argparse
import json
import math
from pathlib import Path

from .auxiliary_queue import completed
from synthetic_dataset_generator.generator import file_checksum, write_json


def select(protocol_path, data=Path("datasets/default"), runs=Path("models/runs")):
    protocol_path, data, runs = map(Path, (protocol_path, data, runs))
    p = json.loads(protocol_path.read_text())
    if p["validation_only"] is not True or p["test_supervision_for_selection"] is not False:
        raise ValueError("Selection must use validation only")
    rates = p["learning_rates"]
    if not rates or len(set(rates)) != len(rates) or any(
            not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Invalid or duplicate declared rates")
    fingerprint = file_checksum(data / "manifest.json")
    trials = []
    for rate in rates:
        tag = format(rate, "f").rstrip("0").split(".")[1]
        name = f"calibration_{p['size']}_{p['representation']}_seed{p['seed']}_lr{tag}"
        job = dict(name=name, size=p["size"], representation=p["representation"],
                   seed=p["seed"], epochs=p["epochs"], batch_size=p["microbatch"],
                   accumulation=p["accumulation"], lr=rate, device=p["device"],
                   threads=p["threads"], purpose="pilot", encoding=p["encoding"],
                   initialization=None, train_limit=p["train_limit"],
                   eval_limit=p["validation_limit"], validation_only=True)
        directory = runs / name
        trial = dict(run=name, learning_rate=rate, complete=completed(directory, job, fingerprint))
        if trial["complete"]:
            report = json.loads((directory / "results.json").read_text())
            trial.update(selected_validation=report["selected_validation"],
                         results_sha256=file_checksum(directory / "results.json"),
                         history_sha256=file_checksum(directory / "history.jsonl"),
                         checkpoint_sha256=file_checksum(directory / "best.pt"))
        trials.append(trial)
    result = dict(status="incomplete_grid_no_selection", trials=trials,
                  protocol_sha256=file_checksum(protocol_path),
                  dataset_manifest_sha256=fingerprint,
                  interpretation="Coverage and saved minimum-validation-loss audit; not independent inference replay or evidence of source mastery.",
                  selection_rule="Minimum selected source-validation loss after all declared trials complete; exact ties retain declared rate order")
    if all(t["complete"] for t in trials):
        winner = min(trials, key=lambda t: t["selected_validation"]["loss"])
        result.update(status="completed_subset_calibration_selection",
                      selected_learning_rate=winner["learning_rate"], selected_run=winner["run"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/default"))
    parser.add_argument("--runs", type=Path, default=Path("models/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Frozen selection records can be protocol dependencies. Never replace one.
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing selection records are preserved")
    result = select(args.protocol, args.data, args.runs)
    write_json(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
