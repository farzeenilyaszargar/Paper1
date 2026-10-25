"""Replay train/validation only to distinguish fitting failure from transfer."""

import argparse
import json
import math
from pathlib import Path

import torch

from synthetic_dataset_generator.generator import file_checksum, write_json
from .compact import passes, report_for
from .architecture_provenance import verify_transformer_provenance
from .train import evaluate, load
from .transformer import TinyTransformer


def diagnose(device="cpu"):
    torch.set_num_threads(1)
    protocol = json.loads(Path("models/compact_protocol.json").read_text())
    pilot = report_for("compact_pilot_100k_symbolic_seed1", protocol)
    data = Path(protocol["dataset"])
    manifest = json.loads((data / "manifest.json").read_text())
    # Integrity checks do not read test labels or use test scores for diagnosis.
    for filename in ("train.jsonl", "validation.jsonl"):
        if file_checksum(data / filename) != manifest["files"][filename]["sha256"]:
            raise ValueError("Changed train/validation data")
    reports = {}
    for name in ("compact_pilot_100k_symbolic_seed1", "compact_diagnostic_overfit32_100k_seed1"):
        directory = Path("models/runs") / name
        if not (directory / "results.json").exists():
            if name.startswith("compact_diagnostic"):
                continue
            raise ValueError("Missing completed pilot")
        r = json.loads((directory / "results.json").read_text())
        c = r["config"]
        verify_transformer_provenance(c)
        if (r["status"] != "validation_only_completed" or r["evaluations"] or
                c["dataset_manifest_sha256"] != file_checksum(data / "manifest.json") or
                c["representation"] != "symbolic" or c["encoding"] != "characters"):
            raise ValueError("Diagnostic identity or validation-only provenance mismatch")
        train = load(data / "train.jsonl", "symbolic", c["train_limit"])
        validation = load(data / "validation.jsonl", "symbolic", c["eval_limit"])
        history = [json.loads(s) for s in (directory / "history.jsonl").read_text().splitlines()]
        steps = math.ceil(len(train) / (c["batch_size"] * c["accumulation"]))
        if (len(history) != c["epochs"] or len(train) != c["train_examples"]
                or len(validation) != c["validation_examples"]
                or any(x["epoch"] != i or x["step"] != i * steps or
                       x["training_examples_seen"] != i * len(train)
                       for i, x in enumerate(history, 1))):
            raise ValueError("Incomplete diagnostic budget")
        model = TinyTransformer(c["size"]).to(device)
        model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=True))
        selected_train = evaluate(model, train, 4, device)
        replay = evaluate(model, validation, 4, device)
        if replay["confusion"] != r["selected_validation"]["confusion"]:
            raise ValueError("Selected validation predictions differ on replay")
        last = torch.load(directory / "last.pt", map_location=device, weights_only=False)
        if last["epoch"] != c["epochs"]:
            raise ValueError("Final checkpoint epoch mismatch")
        model.load_state_dict(last["model"])
        final_train = evaluate(model, train, 4, device)
        reports[name] = dict(selected_checkpoint_train=selected_train,
                             selected_validation_replay=replay, final_checkpoint_train=final_train,
                             selected_checkpoint_sha256=file_checksum(directory / "best.pt"),
                             last_checkpoint_sha256=file_checksum(directory / "last.pt"),
                             results_sha256=file_checksum(directory / "results.json"),
                             epochs=len(history), train_examples=len(train),
                             first_online_perfect_train_epoch=next((x["epoch"] for x in history if x["train_accuracy"] == 1), None))
    return dict(status="pilot_passed" if passes(pilot["selected_validation"]) else "pilot_failed_no_screen",
                dataset_manifest_sha256=file_checksum(data / "manifest.json"), device=device,
                diagnostic_code_sha256=file_checksum(Path(__file__)), runs=reports,
                interpretation="Training-set replay and validation-only checks, not independent test generalization. Memorizing 32 examples does not establish learning the full task. No capacity screen or transfer run is authorized by a failed pilot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("models/compact_diagnostic_results.json"))
    args = parser.parse_args()
    result = diagnose(args.device)
    write_json(args.output, result)
    print(result["status"])
