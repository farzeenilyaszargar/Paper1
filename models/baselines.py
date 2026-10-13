"""Exact inference and a non-reasoning surface-statistic baseline."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from synthetic_dataset_generator.generator import file_checksum, write_json
from synthetic_dataset_generator.logic import Rule, forward_label, infer
from .representations import LABELS

FEATURE_NAMES = ("facts", "rules", "symbols", "or_rules", "and_clauses", "negative_facts",
                 "negative_premises", "negative_conclusions", "negative_query", "input_characters")


def features(record):
    meta = record["metadata"]
    rules = meta["rules"]
    literals = meta["facts"] + [meta["query"]]
    for rule in rules:
        literals.append(rule["conclusion"])
        literals.extend(p for c in rule["antecedent"] for p in c)
    return [len(meta["facts"]), len(rules), len({p.removeprefix("~") for p in literals}),
            sum(len(r["antecedent"]) > 1 for r in rules),
            sum(len(c) > 1 for r in rules for c in r["antecedent"]),
            sum(p.startswith("~") for p in meta["facts"]),
            sum(p.startswith("~") for r in rules for c in r["antecedent"] for p in c),
            sum(r["conclusion"].startswith("~") for r in rules), int(meta["query"].startswith("~")),
            len(record["input"])]


def load(path):
    x, y, oracle = [], [], []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            x.append(features(record))
            y.append(LABELS.index(record["target"]))
            meta = record["metadata"]
            depths, _ = infer(meta["facts"], [Rule.from_dict(r) for r in meta["rules"]])
            oracle.append(LABELS.index(forward_label(depths, meta["query"])))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y), np.array(oracle)


def metrics(y, predictions):
    y = y.numpy()
    confusion = np.zeros((3,3), dtype=int)
    for target, prediction in zip(y, predictions):
        confusion[target, prediction] += 1
    recalls = [float(confusion[i,i]/confusion[i].sum()) for i in range(3)]
    return {"accuracy": float(np.mean(y == predictions)), "balanced_accuracy": float(np.mean(recalls)),
            "count": len(y), "confusion": confusion.tolist(), "recall": dict(zip(LABELS,recalls))}


def run(data, output):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    manifest = json.loads((data/"manifest.json").read_text())
    audit = json.loads((data/"audit.json").read_text())
    if not audit["passed"] or not audit["classical_solver_verified"] or audit["manifest_sha256"] != file_checksum(data/"manifest.json"):
        raise ValueError("Dataset audit is missing/stale")
    for name, info in manifest["files"].items():
        if file_checksum(data/name) != info["sha256"]:
            raise ValueError("Dataset file changed")
    x, y, _ = load(data/"train.jsonl")
    vx, vy, _ = load(data/"validation.jsonl")
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-5)
    x, vx = (x-mean)/scale, (vx-mean)/scale
    classifier = nn.Linear(len(FEATURE_NAMES), 3)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.05)
    best, best_loss = None, float("inf")
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(classifier(x), y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(F.cross_entropy(classifier(vx), vy))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best = copy.deepcopy(classifier.state_dict())
    classifier.load_state_dict(best)
    majority = int(y.bincount(minlength=3).argmax())
    result = {"dataset_manifest_sha256": file_checksum(data/"manifest.json"), "features": list(FEATURE_NAMES),
              "feature_weights_standardized": classifier.weight.detach().tolist(),
              "feature_bias": classifier.bias.detach().tolist(), "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
              "seed": 1, "optimizer": "AdamW", "learning_rate": 0.05, "weight_decay": 0.01,
              "optimizer_steps": 200, "class_order": list(LABELS),
              "selection": "validation loss only; 200 predefined optimizer steps", "splits": {}}
    for name in manifest["files"]:
        if name == "train.jsonl":
            continue
        tx, ty, oracle = load(data/name)
        with torch.no_grad():
            prediction = classifier((tx-mean)/scale).argmax(-1).numpy()
        split = name.removesuffix(".jsonl")
        result["splits"][split] = {"oracle": metrics(ty, oracle), "surface_features": metrics(ty, prediction),
                                   "training_majority": metrics(ty, np.full(len(ty),majority)),
                                   "uniform_random_expected_accuracy": 1/3}
        print(f"{split}: oracle={result['splits'][split]['oracle']['accuracy']:.4f}; surface={result['splits'][split]['surface_features']['accuracy']:.4f}",flush=True)
    write_json(output,result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/default"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline_results.json"))
    args = parser.parse_args()
    run(args.data,args.output)
