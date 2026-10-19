"""Train on train/validation only; evaluate all representations after selection.

Example: python -m models.train --size 100k --representation symbolic --epochs 10
"""

import argparse
import errno
import hashlib
import json
import math
import random
import shutil
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from synthetic_dataset_generator.generator import file_checksum, write_json
from .representations import CHARACTERS, LABELS, PAD, REPRESENTATIONS, encode, format_input
from .transformer import FF_WIDTH, SIZES, TinyTransformer, expected_parameter_count
from .local_encoding import encode_slots


def load(path, representation, limit=None, encoding="characters"):
    samples = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            text = format_input(record,representation)
            tokens = encode_slots(text,representation) if encoding == "local_slots" else encode(text)
            samples.append((tokens, LABELS.index(record["target"]), record["id"]))
            if limit and len(samples) >= limit:
                break
    return samples


def batches(samples, batch_size, rng=None, padding_multiple=128):
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
        maximum = int(lengths.max())
        padded_length = ((maximum+padding_multiple-1)//padding_multiple)*padding_multiple
        tokens = torch.full((len(group), padded_length), PAD, dtype=torch.long)
        for i, sample in enumerate(selected):
            tokens[i, :len(sample[0])] = torch.tensor(sample[0])
        yield tokens, lengths, torch.tensor([s[1] for s in selected]), [s[2] for s in selected]


def optimizer_step(model, optimizer, group, device):
    """Weight microbatch losses by examples, including partial final batches."""
    count = sum(len(batch[2]) for batch in group)
    optimizer.zero_grad(set_to_none=True)
    total_loss = correct = token_count = 0
    for tokens, lengths, targets, _ in group:
        logits = model(tokens.to(device), lengths.to(device))
        loss_sum = F.cross_entropy(logits, targets.to(device), reduction="sum")
        (loss_sum / count).backward()
        total_loss += float(loss_sum.detach().cpu())
        correct += int((logits.argmax(-1).cpu() == targets).sum())
        token_count += int(lengths.sum())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss, correct, count, token_count


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value,dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def require_space(path,required):
    ancestor = path.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < required:
        raise OSError(errno.ENOSPC,f"Insufficient checkpoint space: {free:,} bytes free; {required:,} required; existing checkpoints preserved",str(path))


def training_space_preflight(output,parameters):
    # FP32 model + Adam moments + atomic copies + predictions, plus OS reserve.
    require_space(output,1024**3+32*parameters+64*1024**2)


def save_checkpoint(value, path):
    require_space(path.parent,tensor_bytes(value)+512*1024**2+1024**2)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


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
    training_space_preflight(args.output,expected_parameter_count(args.size))
    torch.set_num_threads(args.threads)
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
    train = load(args.data / "train.jsonl", args.representation, args.train_limit, args.encoding)
    validation = load(args.data / "validation.jsonl", args.representation, args.eval_limit, args.encoding)
    model = TinyTransformer(args.size).to(device)
    initialization_path = getattr(args,"initialization",None)
    warmup = None
    if initialization_path is not None:
        if args.encoding != "characters":
            raise ValueError("Raw syntax warmup cannot initialize parser-normalized encoding")
        from .syntax_warmup import initialization as load_initialization
        initial_state,warmup = load_initialization(initialization_path,args.size,args.seed,args.representation,file_checksum(manifest_path))
        if warmup["config"]["parameter_count"] != model.parameter_count:
            raise ValueError("Warmup parameter count differs from current model")
        if not args.resume:
            model.load_state_dict(initial_state)
    # Supervised optimization always starts with fresh moments after warmup.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    config = vars(args).copy()
    config["data"], config["output"] = str(args.data), str(args.output)
    config["initialization"] = str(initialization_path) if initialization_path is not None else None
    config["initialization_sha256"] = file_checksum(initialization_path) if initialization_path is not None else None
    config["warmup"] = warmup
    config.update(device=device, parameter_count=model.parameter_count, torch_version=torch.__version__,
                  dataset_name=manifest["config"]["name"],
                  effective_batch_size=args.batch_size * args.accumulation,
                  padding_bucket_tokens=128,
                  evaluation_batch_size=args.eval_batch_size or min(args.batch_size,4),
                  architecture={"width": SIZES[args.size][0], "layers": SIZES[args.size][1],
                                "heads": SIZES[args.size][2], "ff_width": FF_WIDTH.get(args.size, 4 * SIZES[args.size][0])},
                  source_checksums={name: file_checksum(Path(__file__).parent / name)
                                    for name in ("train.py", "transformer.py", "representations.py", "parsing.py", "local_encoding.py")},
                  dataset_manifest_sha256=file_checksum(manifest_path), characters=CHARACTERS,
                  train_examples=len(train), validation_examples=len(validation),
                  determinism="seeded; MPS kernels may be nondeterministic")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        checkpoint = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        old = json.loads((args.output / "config.json").read_text())
        if old.get("encoding","characters") != args.encoding:
            raise ValueError("Resume encoding mismatch")
        if old.get("accumulation",1) != args.accumulation:
            raise ValueError("Resume accumulation mismatch")
        if old.get("validation_only",False) != args.validation_only:
            raise ValueError("Resume validation-only mode mismatch")
        if old.get("initialization_sha256") != config["initialization_sha256"]:
            raise ValueError("Resume initialization mismatch")
        for key in ("size", "representation", "parameter_count", "seed", "dataset_manifest_sha256", "train_limit", "eval_limit", "batch_size", "lr"):
            if config[key] != old[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        start_epoch, best_loss, step = checkpoint["epoch"] + 1, checkpoint["best_loss"], checkpoint["step"]
    else:
        write_json(args.output / "config.json", config)
        start_epoch, best_loss, step = 1, float("inf"), 0
    tokens_seen = sum(len(s[0]) for s in train) * (start_epoch - 1)
    with (args.output / "invocations.jsonl").open("a") as invocations:
        invocations.write(json.dumps({"start_epoch": start_epoch, "config": config}) + "\n")
    started = time.monotonic()
    with (args.output / "history.jsonl").open("a") as history:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = count = 0
            iterator = iter(batches(train, args.batch_size, random.Random(args.seed * 100000 + epoch)))
            while group := list(islice(iterator, args.accumulation)):
                loss_sum, batch_correct, batch_count, batch_tokens = optimizer_step(model,optimizer,group,device)
                total_loss += loss_sum
                correct += batch_correct
                count += batch_count
                tokens_seen += batch_tokens
                step += 1
                if step % 50 == 0:
                    print(f"epoch={epoch} step={step} train_loss={total_loss/count:.4f} train_accuracy={correct/count:.4f}", flush=True)
            metrics = evaluate(model, validation, config["evaluation_batch_size"], device)
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            row = {"epoch": epoch, "step": step, "train_loss": total_loss / count,
                   "train_accuracy": correct / count, "validation": metrics,
                   "training_examples_seen": epoch * len(train), "nonpadding_training_tokens_seen": tokens_seen,
                   "elapsed_seconds_this_invocation": time.monotonic() - started}
            history.write(json.dumps(row) + "\n")
            history.flush()
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(model.state_dict(), args.output / "best.pt")
            save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "best_loss": best_loss, "step": step, "torch_rng": torch.get_rng_state()}, args.output / "last.pt")
            write_json(args.output / "state.json", row)
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True))
    selected_validation = evaluate(model,validation,config["evaluation_batch_size"],device)
    results = {"status": "completed", "config": config, "selection": "minimum source-representation validation loss",
               "scope": "memorization diagnostic" if args.purpose == "diagnostic" else "smoke" if manifest["config"]["name"] == "smoke" else "pilot subset" if args.train_limit or args.eval_limit else "full corpus", "evaluations": {}}
    results["selected_validation"] = selected_validation
    results["source_learning_gate_passed"] = selected_validation["accuracy"] >= 0.95 and all(
        value is not None and value >= 0.90 for value in selected_validation["recall"].values())
    if warmup is not None:
        results["scope"] = warmup["config"]["treatment"]+" warmup "+("pilot" if args.train_limit or args.eval_limit else "full control")
        results["interpretation"] = "Separate unlabeled-exposure treatment; compare matched source-only warmup, not just a no-warmup baseline. Not zero-exposure transfer."
    if args.encoding == "local_slots":
        results["scope"] = "binding-normalized pilot" if args.train_limit or args.eval_limit else "binding-normalized full control"
        results["interpretation"] = "External parsing and alpha-renaming; identical normalized inputs are not evidence of learned raw-format transfer"
    if args.validation_only:
        results.update(status="validation_only_completed",scope="validation-only calibration",
                       interpretation="No test examples were loaded for model evaluation. Dataset file checksums were verified for integrity only. Select hyperparameters using source validation, not test diagnostics.")
        write_json(args.output / "results.json",results)
        return results
    training_chars = set(c for sequence, _, _ in train for c in sequence)
    all_training_chars = training_chars | (set(warmup["input_token_ids_observed"]) if warmup else set())
    for representation in REPRESENTATIONS:
        results["evaluations"][representation] = {}
        for filename in manifest["files"]:
            if filename in ("train.jsonl", "validation.jsonl"):
                continue
            split = filename.removesuffix(".jsonl")
            samples = load(args.data / filename, representation, args.eval_limit, args.encoding)
            metrics = evaluate(model, samples, config["evaluation_batch_size"], device,
                               args.output / f"{representation}_{split}.predictions.jsonl")
            test_chars = set(c for sequence, _, _ in samples for c in sequence)
            metrics["unseen_source_token_ids"] = sorted(test_chars - training_chars)
            metrics["unseen_all_training_token_ids"] = sorted(test_chars - all_training_chars)
            if args.encoding == "characters":
                metrics["unseen_source_character_ids"] = metrics["unseen_source_token_ids"]
            results["evaluations"][representation][split] = metrics
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
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
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--purpose", choices=("experiment", "pilot", "diagnostic", "smoke"), default="experiment")
    parser.add_argument("--encoding", choices=("characters", "local_slots"), default="characters")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--eval-batch-size",type=int)
    parser.add_argument("--initialization",type=Path,help="Verified TRAIN-only LM warmup initialization.pt")
    parser.add_argument("--validation-only",action="store_true",
                        help="Finish after source validation; never evaluate test splits (for calibration)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.accumulation) < 1 or args.lr <= 0 or any(v is not None and v < 1 for v in (args.train_limit, args.eval_limit, args.eval_batch_size)):
        parser.error("Epochs, batch size, learning rate and optional limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
