"""Gated, explicitly invoked compact stages. No automatic 45-run sweep."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from synthetic_dataset_generator.generator import file_checksum
from .auxiliary_queue import live_training_commands

PROTOCOL = Path("models/compact_protocol.json")


def passes(metric):
    return (metric.get("count") == 1000 and metric.get("accuracy", 0) >= .95
            and set(metric.get("recall", {})) == {"entailed", "contradicted", "unknown"}
            and all(v is not None and v >= .9 for v in metric["recall"].values()))


def report_for(name, p):
    path = Path("models/runs") / name / "results.json"
    if not path.exists():
        raise ValueError(f"Required completed run missing: {name}")
    r = json.loads(path.read_text())
    c = r["config"]
    if (r["status"] not in ("completed", "validation_only_completed")
            or c["dataset_manifest_sha256"] != file_checksum(Path(p["dataset"]) / "manifest.json")
            or c["train_examples"] != 5000 or c["validation_examples"] != 1000
            or c["epochs"] != 20 or c["seed"] != 1 or c["encoding"] != "characters"
            or c["lr"] != .003 or c["batch_size"] != 16 or c["accumulation"] != 2
            or c["representation"] != "symbolic" or c["initialization"] is not None):
        raise ValueError("Required run has wrong budget/provenance")
    history = [json.loads(s) for s in (path.parent / "history.jsonl").read_text().splitlines()]
    if ([x["epoch"] for x in history] != list(range(1, 21)) or any(
            x["step"] != x["epoch"] * 157 or x["training_examples_seen"] != x["epoch"] * 5000
            for x in history) or
            r["selected_validation"]["loss"] != min(x["validation"]["loss"] for x in history)):
        raise ValueError("Required run lacks complete budget or minimum-loss selection")
    return r


def planned(stage, p):
    if stage == "pilot":
        return [("compact_pilot_100k_symbolic_seed1", "100k", "symbolic", True)]
    pilot = report_for("compact_pilot_100k_symbolic_seed1", p)
    if pilot["config"]["size"] != "100k" or pilot["status"] != "validation_only_completed":
        raise ValueError("Wrong pilot identity/status")
    if not passes(pilot["selected_validation"]):
        raise ValueError("Pilot failed learning gate: diagnose before launching more models")
    if stage == "screen":
        return [(f"compact_screen_{s}_symbolic_seed1", s, "symbolic", False) for s in p["screen_sizes"]]
    successful = []
    for size in p["screen_sizes"]:
        r = report_for(f"compact_screen_{size}_symbolic_seed1", p)
        if r["config"]["size"] != size or r["status"] != "completed":
            raise ValueError("Wrong screen identity/status")
        if (passes(r["selected_validation"]) and
                passes(r.get("evaluations", {}).get("symbolic", {}).get("test_iid", {}))):
            successful.append(size)
    if not successful:
        raise ValueError("No screened model passes validation and source IID mastery")
    size = successful[0]
    return [(f"compact_transfer_{size}_{rep}_seed1", size, rep, False) for rep in ("function", "english")]


def command(job, p):
    name, size, rep, validation_only = job
    args = [sys.executable, "-u", "-m", "models.train", "--data", p["dataset"],
            "--output", f"models/runs/{name}", "--size", size, "--representation", rep,
            "--seed", "1", "--epochs", "20", "--batch-size", "16", "--accumulation", "2",
            "--lr", "0.003", "--device", "mps", "--threads", "1", "--purpose", "pilot"]
    if validation_only:
        args.append("--validation-only")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "screen", "transfer"))
    parser.add_argument("--execute", action="store_true", help="Without this flag, print gated commands only")
    args = parser.parse_args()
    p = json.loads(PROTOCOL.read_text())
    jobs = planned(args.stage, p)
    with Path(".compact.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.execute and live_training_commands():
            raise RuntimeError("A trainer is already live; do not overlap compact and legacy jobs")
        for job in jobs:
            cmd = command(job, p)
            print(" ".join(cmd), flush=True)
            if args.execute:
                # Existing output is intentionally not overwritten or silently resumed.
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
