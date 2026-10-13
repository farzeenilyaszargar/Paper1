"""Train on train/validation only; evaluate all representations after selection.

Example: python -m models.train --size 100k --representation symbolic --epochs 10
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from synthetic_dataset_generator.generator import file_checksum, write_json
from .representations import CHARACTERS, LABELS, PAD, REPRESENTATIONS, encode, format_input
from .transformer import FF_WIDTH, SIZES, TinyTransformer


def load(path, representation, limit=None):
    samples = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            samples.append((encode(format_input(record, representation)), LABELS.index(record["target"]), record["id"]))
            if limit and len(samples) >= limit:
                break
    return samples


def batches(samples, batch_size, rng=None):
    indices = list(range(len(samples)))
    if rng:
        rng.shuffle(indices)
    # Sort only within shuffled windows; avoids learning labels through batch order.
    ordered = []
    for start in range(0, len(indices), 32 * batch_size):
        ordered.extend(sorted(indices[start:start + 32 * batch_size], key=lambda i: len(samples[i][0])))
    groups = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    if rng:
        rng.shuffle(groups)
    for group in groups:
        selected = [samples[i] for i in group]
        lengths = torch.tensor([len(s[0]) for s in selected])
        tokens = torch.full((len(group), int(lengths.max())), PAD, dtype=torch.long)
        for i, sample in enumerate(selected):
            tokens[i, :len(sample[0])] = torch.tensor(sample[0])
        yield tokens, lengths, torch.tensor([s[1] for s in selected]), [s[2] for s in selected]


@torch.no_grad()
def evaluate(model, samples, batch_size, device, prediction_path=None):
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.int64)
    loss = 0.0
    handle = prediction_path.open("w") if prediction_path else None
    try:
        for tokens, lengths, targets, ids in batches(samples, batch_size):
            logits = model(tokens.to(device), lengths.to(device))
            loss += float(F.cross_entropy(logits, targets.to(device), reduction="sum").cpu())
            predictions = logits.argmax(-1).cpu().tolist()
            for target, predicted, identifier in zip(targets.tolist(), predictions, ids):
                confusion[target, predicted] += 1
                if handle:
                    handle.write(json.dumps({"id": identifier, "target": LABELS[target], "prediction": LABELS[predicted]}) + "\n")
    finally:
        if handle:
            handle.close()
    count = int(confusion.sum())
    correct = int(np.trace(confusion))
    p = correct / count
    z = 1.96
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    recalls = [float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None for i in range(3)]
    return {"accuracy": p, "count": count, "correct": correct, "loss": loss / count,
            "wilson95": [center - half, center + half], "confusion": confusion.tolist(),
            "recall": dict(zip(LABELS, recalls)), "balanced_accuracy": float(np.mean([r for r in recalls if r is not None]))}


def run(args):
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    manifest_path = args.data / "manifest.json"
    audit_path = args.data / "audit.json"
    if not audit_path.exists():
        raise ValueError("Dataset must have a passing full audit")
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or not audit.get("classical_solver_verified") or audit["manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Missing, stale or incomplete dataset audit")
    manifest = json.loads(manifest_path.read_text())
    # Check every file before training so saved audit cannot mask changed data.
    for name, info in manifest["files"].items():
        if file_checksum(args.data / name) != info["sha256"]:
            raise ValueError(f"Dataset changed since audit: {name}")
    train = load(args.data / "train.jsonl", args.representation, args.train_limit)
    validation = load(args.data / "validation.jsonl", args.representation, args.eval_limit)
    model = TinyTransformer(args.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    config = vars(args).copy()
    config["data"], config["output"] = str(args.data), str(args.output)
    config.update(device=device, parameter_count=model.parameter_count, torch_version=torch.__version__,
                  dataset_name=manifest["config"]["name"],
                  architecture={"width": SIZES[args.size][0], "layers": SIZES[args.size][1],
                                "heads": SIZES[args.size][2], "ff_width": FF_WIDTH.get(args.size, 4 * SIZES[args.size][0])},
                  source_checksums={name: file_checksum(Path(__file__).parent / name)
                                    for name in ("train.py", "transformer.py", "representations.py")},
                  dataset_manifest_sha256=file_checksum(manifest_path), characters=CHARACTERS,
                  train_examples=len(train), validation_examples=len(validation),
                  determinism="seeded; MPS kernels may be nondeterministic")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        checkpoint = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        old = json.loads((args.output / "config.json").read_text())
        for key in ("size", "representation", "parameter_count", "seed", "dataset_manifest_sha256", "train_limit", "eval_limit", "batch_size", "lr"):
            if config[key] != old[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start_epoch, best_loss, step = checkpoint["epoch"] + 1, checkpoint["best_loss"], checkpoint["step"]
    else:
        write_json(args.output / "config.json", config)
        start_epoch, best_loss, step = 1, float("inf"), 0
    tokens_seen = sum(len(s[0]) for s in train) * (start_epoch - 1)
    started = time.monotonic()
    with (args.output / "history.jsonl").open("a") as history:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = count = 0
            for tokens, lengths, targets, _ in batches(train, args.batch_size, random.Random(args.seed * 100000 + epoch)):
                optimizer.zero_grad(set_to_none=True)
                logits = model(tokens.to(device), lengths.to(device))
                loss = F.cross_entropy(logits, targets.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(targets)
                correct += int((logits.argmax(-1).cpu() == targets).sum())
                count += len(targets)
                tokens_seen += int(lengths.sum())
                step += 1
                if step % 50 == 0:
                    print(f"epoch={epoch} step={step} train_loss={total_loss/count:.4f} train_accuracy={correct/count:.4f}", flush=True)
            metrics = evaluate(model, validation, args.batch_size, device)
            row = {"epoch": epoch, "step": step, "train_loss": total_loss / count,
                   "train_accuracy": correct / count, "validation": metrics,
                   "training_examples_seen": epoch * len(train), "nonpadding_training_tokens_seen": tokens_seen,
                   "elapsed_seconds_this_invocation": time.monotonic() - started}
            history.write(json.dumps(row) + "\n")
            history.flush()
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save(model.state_dict(), args.output / "best.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "best_loss": best_loss, "step": step, "torch_rng": torch.get_rng_state()}, args.output / "last.pt")
            write_json(args.output / "state.json", row)
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
    results = {"status": "completed", "config": config, "selection": "minimum source-representation validation loss",
               "scope": "smoke" if manifest["config"]["name"] == "smoke" else "pilot subset" if args.train_limit or args.eval_limit else "full corpus", "evaluations": {}}
    for representation in REPRESENTATIONS:
        results["evaluations"][representation] = {}
        for filename in manifest["files"]:
            if filename in ("train.jsonl", "validation.jsonl"):
                continue
            split = filename.removesuffix(".jsonl")
            samples = load(args.data / filename, representation, args.eval_limit)
            metrics = evaluate(model, samples, args.batch_size, device,
                               args.output / f"{representation}_{split}.predictions.jsonl")
            training_chars = set(c for sequence, _, _ in train for c in sequence)
            test_chars = set(c for sequence, _, _ in samples for c in sequence)
            metrics["unseen_source_character_ids"] = sorted(test_chars - training_chars)
            results["evaluations"][representation][split] = metrics
            write_json(args.output / "results.json", results | {"status": "evaluation_in_progress"})
            print(f"test {representation}/{split}: {metrics['accuracy']:.4f}", flush=True)
    write_json(args.output / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/default"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=SIZES, default="100k")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="symbolic")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0 or any(v is not None and v < 1 for v in (args.train_limit, args.eval_limit)):
        parser.error("Epochs, batch size, learning rate and optional limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
