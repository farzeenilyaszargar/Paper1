"""Streaming audits of stored data; solver checks are enabled by default."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .generator import (Config, LABELS, SPLITS, VOCABULARY, digest, file_checksum,
                        group_id, reserved_pair)
from .logic import (Rule, atom, canonical_problem, classical_label, dependency_edges,
                    forward_label, has_reconvergence, infer, negate, render)
from .verification import SolverVerifier


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit(directory: Path, verify_solver: bool, progress, verifier: SolverVerifier) -> dict:
    config = Config.load(directory / "config.json")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["config"] == json.loads((directory / "config.json").read_text()), "Config/manifest mismatch")
    require(manifest["config_sha256"] == file_checksum(directory / "config.json"), "Config checksum mismatch")
    require(manifest["vocabulary_sha256"] == file_checksum(directory / "vocabulary.json"), "Vocabulary checksum mismatch")
    vocabulary = json.loads((directory / "vocabulary.json").read_text())
    require(vocabulary == {"type": "character", "characters": VOCABULARY,
                           "special_tokens": ["<pad>", "<bos>", "<eos>"], "targets": list(LABELS)}, "Invalid vocabulary")
    require(set(manifest["files"]) == {f"{s}.jsonl" for s in SPLITS}, "Invalid split inventory")
    ids, groups = set(), set()
    training_symbols, training_chars = set(), set()
    report = {"passed": True, "classical_solver_verified": verify_solver,
              "manifest_sha256": file_checksum(directory / "manifest.json"), "splits": {}}
    for split, expected_depth in SPLITS.items():
        path = directory / f"{split}.jsonl"
        require(file_checksum(path) == manifest["files"][path.name]["sha256"], f"{split}: checksum mismatch")
        labels = Counter()
        stats = {label: {"count": 0, "input_characters": 0, "symbols": 0, "rules": 0,
                         "facts": 0, "or_rules": 0, "negative_query": 0} for label in LABELS}
        length_histogram = Counter()
        families = Counter()
        allowed = set(config.pool(split))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                prefix = f"{split}:{line_number}: "
                record = json.loads(line)
                meta = record["metadata"]
                facts = meta["facts"]
                rules = [Rule.from_dict(r) for r in meta["rules"]]
                require(all(1 <= len(r.clauses) <= 4 and all(1 <= len(c) <= 3 for c in r.clauses)
                            for r in rules), prefix + "Antecedent bounds violated")
                query = meta["query"]
                target = record["target"]
                conclusions = {r.conclusion for r in rules}
                require({query, negate(query)} <= conclusions, prefix + "Missing opposite-polarity candidate path")
                require(target in LABELS and record["split"] == split, prefix + "Invalid label/split")
                expected_id = digest(canonical_problem(facts, rules, query))
                expected_group = group_id(facts, rules)
                require(record["id"] == expected_id and expected_id not in ids, prefix + "Duplicate or invalid instance ID")
                require(meta["group_id"] == expected_group and expected_group not in groups, prefix + "Duplicate underlying knowledge base")
                ids.add(expected_id)
                groups.add(expected_group)
                symbols = {atom(f) for f in facts} | {atom(query)}
                for rule in rules:
                    symbols.add(atom(rule.conclusion))
                    symbols.update(atom(p) for c in rule.clauses for p in c)
                require(symbols == set(meta["symbols"]) and symbols <= allowed, prefix + "Invalid symbol inventory")
                require(all(len(s) == 4 and s[0] == "p" and s[1:].isdigit() for s in symbols), prefix + "Invalid symbol spelling")
                if split == "train":
                    training_symbols.update(symbols)
                    training_chars.update(record["input"])
                elif split != "test_symbols":
                    require(symbols <= training_symbols, prefix + "Unseen symbol in familiar-symbol test")
                else:
                    require(not symbols & training_symbols, prefix + "Held-out symbol leakage")
                edges = dependency_edges(rules)
                require(meta["edges"] == [list(e) for e in sorted(edges)], prefix + "Invalid graph metadata")
                require(all(reserved_pair(config.seed, a, b) == (split == "test_combinations") for a, b in edges), prefix + "Reserved dependency pair violation")
                reconverges = has_reconvergence(edges)
                require(reconverges == (split == "test_structures"), prefix + "Graph family leakage")
                if split == "test_structures":
                    require(meta["family"] == "diamond", prefix + "Incorrect held-out family")
                    # Require an actual two-hop conjunctive diamond, not just a family tag.
                    for candidate in (query, negate(query)):
                        require(any(len(c) == 2 and any((root, atom(c[0])) in edges and (root, atom(c[1])) in edges
                                    for root in symbols) for r in rules if r.conclusion == candidate
                                    for c in r.clauses), prefix + "Missing AND diamond for query polarity")
                depths, proof = infer(facts, rules)
                require(forward_label(depths, query) == target, prefix + "Incorrect forward label")
                if verify_solver:
                    verifier.check(facts, rules, query, target)
                decisive = query if target == "entailed" else negate(query) if target == "contradicted" else None
                require(meta["decisive_literal"] == decisive, prefix + "Incorrect decisive literal")
                actual_depth = depths.get(decisive) if decisive else None
                require(meta["proof_depth"] == actual_depth and (decisive is None or actual_depth == expected_depth), prefix + "Incorrect proof depth")
                require(meta["matched_depth"] == expected_depth, prefix + "Incorrect matching depth")
                require(meta["derivation"] == proof, prefix + "Incorrect derivation metadata")
                require(record["input"] == render(facts, rules, query), prefix + "Input/metadata mismatch")
                require(set(record["input"]) <= set(VOCABULARY), prefix + "Unknown input character")
                labels[target] += 1
                families[meta["family"]] += 1
                length_histogram[str(len(record["input"]))] += 1
                values = stats[target]
                for key, value in {"count": 1, "input_characters": len(record["input"]), "symbols": len(symbols),
                                   "rules": len(rules), "facts": len(facts),
                                   "or_rules": sum(len(r.clauses) > 1 for r in rules),
                                   "negative_query": int(query.startswith("~"))}.items():
                    values[key] += value
                if line_number % 10000 == 0:
                    progress(f"Audited {split}: {line_number}")
        require(sum(labels.values()) == config.count(split), f"{split}: incorrect count")
        require(max(labels.get(k, 0) for k in LABELS) - min(labels.get(k, 0) for k in LABELS) <= 1, f"{split}: label imbalance")
        require(dict(labels) == manifest["files"][path.name]["labels"], f"{split}: manifest label count mismatch")
        require(sum(labels.values()) == manifest["files"][path.name]["count"], f"{split}: manifest total mismatch")
        for values in stats.values():
            for key in list(values):
                if key != "count":
                    values[key] = round(values[key] / values["count"], 4) if values["count"] else None
        verifier.finish()
        report["splits"][split] = {"count": sum(labels.values()), "labels": dict(labels),
                                  "families": dict(families), "means_by_label": stats,
                                  "input_length_histogram": dict(sorted(length_histogram.items(), key=lambda x: int(x[0])))}
        progress(f"Passed {split}: {sum(labels.values())} examples")
    require(training_chars == set(VOCABULARY), "Training does not cover every character token")
    report["total_examples"] = len(ids)
    report["training_symbol_count"] = len(training_symbols)
    return report


def audit(directory: Path, verify_solver: bool = True, progress=print, *, workers: int = 1) -> dict:
    with SolverVerifier(workers) as verifier:
        return _audit(directory, verify_solver, progress, verifier)
