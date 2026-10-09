"""Deterministic generation of controlled propositional reasoning problems."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import z3

from . import __version__
from .logic import (Rule, atom, canonical_problem, classical_label, dependency_edges,
                    forward_label, infer, negate, render)
from .verification import SolverVerifier

LABELS = ("entailed", "contradicted", "unknown")
SPLITS = {
    "train": 2, "validation": 2, "test_iid": 2,
    "test_combinations": 2, "test_structures": 2, "test_symbols": 2,
    "test_depth_3": 3, "test_depth_4": 4, "test_depth_5": 5,
}
# Sorted Unicode characters, not individually learned symbol-name embeddings.
VOCABULARY = sorted(set("FACTS: RULES: QUERY: p0123456789;().~&|->"))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def file_checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class Config:
    name: str = "default"
    seed: int = 42
    train_count: int = 100000
    evaluation_count: int = 10000
    familiar_symbols: int = 512
    heldout_symbols: int = 256
    max_attempts_per_example: int = 100

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or Path(self.name).name != self.name or self.name in (".", ".."):
            raise ValueError("name must be a nonempty directory basename")
        for name, value in asdict(self).items():
            if name != "name" and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{name} must be an integer")
        if min(self.train_count, self.evaluation_count, self.max_attempts_per_example) < 1:
            raise ValueError("Counts and attempt limit must be positive")
        if min(self.familiar_symbols, self.heldout_symbols) < 128:
            raise ValueError("Use at least 128 symbols per pool for constrained embeddings")
        if self.familiar_symbols + self.heldout_symbols > 1000:
            raise ValueError("Three-digit symbol encoding supports at most 1000 names")

    @classmethod
    def load(cls, path: Path) -> Config:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def count(self, split: str) -> int:
        return self.train_count if split == "train" else self.evaluation_count

    def pool(self, split: str) -> list[str]:
        start, count = (self.familiar_symbols, self.heldout_symbols) if split == "test_symbols" else (0, self.familiar_symbols)
        return [f"p{i:03}" for i in range(start, start + count)]


def reserved_pair(seed: int, a: str, b: str) -> bool:
    # Stable across processes/platforms; deliberately never use Python hash().
    return int(digest([seed, "symbol-pair", a, b])[:8], 16) % 5 == 0


@lru_cache(maxsize=4096)
def permitted_targets(seed: int, source: str, start: int, count: int, reserved: bool) -> frozenset[str]:
    return frozenset(f"p{i:03}" for i in range(start, start + count)
                     if reserved_pair(seed, source, f"p{i:03}") == reserved)


def group_id(facts: list[str], rules: list[Rule]) -> str:
    problem = canonical_problem(facts, rules, "")
    del problem["query"]
    return digest(problem)


def make_problem(config: Config, split: str, label: str, rng: random.Random, *, verify_solver: bool = True) -> dict:
    depth = SPLITS[split]
    clauses: list[tuple[list[list[int]], int]] = []
    roots: list[int] = []
    node_count = 0

    def node() -> int:
        nonlocal node_count
        n = node_count
        node_count += 1
        return n

    def chain(length: int) -> int:
        current = node()
        roots.append(current)
        for _ in range(length):
            target = node()
            clauses.append(([[current]], target))
            current = target
        return current

    if split == "test_structures":
        root = chain(0)
        left, right, terminal = node(), node(), node()
        clauses.extend([([[root]], left), ([[root]], right), ([[left, right]], terminal)])
        family = "diamond"
    else:
        current = chain(0)
        uses_merge = False
        for stage in range(depth):
            antecedent = [[current]]
            if rng.random() < 0.45:
                # Separate ancestry prevents reconvergence in familiar families.
                side = chain(stage)
                antecedent = [[current, side]] if rng.random() < 0.5 else [[current], [side]]
                uses_merge = True
            target = node()
            clauses.append((antecedent, target))
            current = target
        terminal = current
        family = "merge_tree" if uses_merge else "chain"

    core_nodes = node_count
    core_roots = list(roots)
    # Dangling branches require no extra fact and cannot shorten the query proof.
    for _ in range(rng.randint(1, 3)):
        parent = rng.randrange(core_nodes)
        child = node()
        clauses.append(([[parent]], child))
    for _ in range(rng.randint(1, 3)):
        chain(rng.randint(1, depth))

    # Both polarities of the queried atom have a proof path. Which roots are
    # facts, not a sign comparison with the final rule, determines the answer.
    previous_root_count = len(roots)
    if split == "test_structures":
        alternate_root = chain(0)
        alternate_left, alternate_right = node(), node()
        clauses.extend([([[alternate_root]], alternate_left), ([[alternate_root]], alternate_right),
                        ([[alternate_left, alternate_right]], terminal)])
    else:
        alternative = chain(depth - 1)
        clauses.append(([[alternative]], terminal))
    alternative_root = roots[previous_root_count]
    opposite_rule_index = len(clauses) - 1
    query_opposite = rng.random() < 0.5
    if label == "unknown":
        inactive_roots = set(core_roots + [alternative_root])
    else:
        use_alternative = query_opposite == (label == "entailed")
        inactive_roots = set(core_roots if use_alternative else [alternative_root])

    fact_count = rng.randint(max(8, len(roots)), 12)
    fact_nodes = list(roots)
    while len(fact_nodes) < fact_count:
        fact_nodes.append(node())
    if len(fact_nodes) > fact_count:
        raise ValueError("Fact budget exceeded")
    # Every label gets the same backup components. Unknown removes core facts
    # and activates these instead, preserving fact, rule and symbol counts.
    backups = []
    for _ in core_roots + [alternative_root]:
        root, leaf = node(), node()
        backups.append(root)
        clauses.append(([[root]], leaf))
    backup_for = dict(zip(core_roots + [alternative_root], backups))
    fact_nodes = [backup_for[r] if r in inactive_roots else r for r in fact_nodes]

    parents = {n: set() for n in range(node_count)}
    for terms, conclusion in clauses:
        parents[conclusion].update(p for term in terms for p in term)
    names = {}
    available = set(config.pool(split))
    start = config.familiar_symbols if split == "test_symbols" else 0
    count = config.heldout_symbols if split == "test_symbols" else config.familiar_symbols
    # The opposite-polarity branch points back to an already allocated terminal.
    # Assign symbols in dependency order, not allocation order.
    pending = set(range(node_count))
    while pending:
        ready = sorted(n for n in pending if parents[n] <= names.keys())
        if not ready:
            raise ValueError("Cyclic generation scaffold")
        n = ready[0]
        pending.remove(n)
        candidates = available.copy()
        for p in sorted(parents[n]):
            candidates.intersection_update(permitted_targets(
                config.seed, names[p], start, count, split == "test_combinations"))
        if not candidates:
            raise ValueError("No symbol assignment satisfies dependency partition")
        chosen = rng.choice(sorted(candidates))
        names[n] = chosen
        available.remove(chosen)
    signed = {n: ("~" if rng.random() < 0.5 else "") + names[n] for n in range(node_count)}
    facts = [signed[n] for n in fact_nodes]
    rules = [Rule(tuple(tuple(signed[n] for n in term) for term in terms),
                  negate(signed[c]) if index == opposite_rule_index else signed[c])
             for index, (terms, c) in enumerate(clauses)]
    query = negate(signed[terminal]) if query_opposite else signed[terminal]
    rng.shuffle(facts)
    # Randomize all semantically irrelevant presentation order.
    rng.shuffle(rules)
    shuffled_rules = []
    for rule in rules:
        terms = [list(c) for c in rule.clauses]
        for term in terms:
            rng.shuffle(term)
        rng.shuffle(terms)
        shuffled_rules.append(Rule(tuple(tuple(c) for c in terms), rule.conclusion))
    rules = shuffled_rules
    depths, derivation = infer(facts, rules)
    if forward_label(depths, query) != label or (verify_solver and classical_label(facts, rules, query) != label):
        raise ValueError("Inference/solver label mismatch")
    decisive = query if label == "entailed" else negate(query) if label == "contradicted" else None
    actual_depth = depths.get(decisive) if decisive is not None else None
    if decisive is not None and actual_depth != depth:
        raise ValueError("Unexpected minimum inference depth")
    problem_id = digest(canonical_problem(facts, rules, query))
    return {
        "id": problem_id, "split": split, "input": render(facts, rules, query), "target": label,
        "metadata": {
            "facts": facts, "rules": [r.to_dict() for r in rules], "query": query,
            "group_id": group_id(facts, rules),
            "symbols": sorted(names.values()), "edges": [list(e) for e in sorted(dependency_edges(rules))],
            "family": family, "matched_depth": depth, "proof_depth": actual_depth,
            "decisive_literal": decisive, "derivation": derivation,
        },
    }


def _generate(config: Config, output: Path, progress, verifier: SolverVerifier) -> dict:
    # Refuse replacement, including an existing partial run.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(output / "vocabulary.json", {"type": "character", "characters": VOCABULARY,
               "special_tokens": ["<pad>", "<bos>", "<eos>"], "targets": list(LABELS)})
    seen: set[str] = set()
    files = {}
    train_symbols = set()
    train_characters = set()
    for split in SPLITS:
        rng = random.Random(int(digest([config.seed, split]), 16))
        labels = [LABELS[i % 3] for i in range(config.count(split))]
        rng.shuffle(labels)
        counts = dict.fromkeys(LABELS, 0)
        attempts = 0
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, label in enumerate(labels):
                for _ in range(config.max_attempts_per_example):
                    attempts += 1
                    try:
                        record = make_problem(config, split, label, rng, verify_solver=False)
                    except ValueError:
                        continue
                    if record["metadata"]["group_id"] not in seen:
                        break
                else:
                    raise RuntimeError(f"Cannot generate unique valid {split} example {index}; constraints were not relaxed")
                seen.add(record["metadata"]["group_id"])
                meta = record["metadata"]
                verifier.check(meta["facts"], [Rule.from_dict(r) for r in meta["rules"]],
                               meta["query"], record["target"])
                if split == "train":
                    train_symbols.update(record["metadata"]["symbols"])
                    train_characters.update(record["input"])
                elif split == "test_combinations" and not set(record["metadata"]["symbols"]) <= train_symbols:
                    raise RuntimeError("Combination test contains symbols absent from training")
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                counts[label] += 1
                if (index + 1) % 10000 == 0:
                    progress(f"{split}: {index + 1}/{len(labels)}")
        verifier.finish()
        files[path.name] = {"count": len(labels), "labels": counts, "attempts": attempts,
                            "sha256": file_checksum(path)}
        progress(f"Completed {split}: {len(labels)} examples")
    if train_characters != set(VOCABULARY):
        raise RuntimeError(f"Training does not cover vocabulary: {set(VOCABULARY) - train_characters}")
    verifier.finish()
    manifest = {
        "schema_version": 1, "generator_version": __version__, "solver_version": z3.get_version_string(),
        "config": asdict(config), "config_sha256": file_checksum(output / "config.json"),
        "vocabulary_sha256": file_checksum(output / "vocabulary.json"),
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def generate(config: Config, output: Path, progress=print, *, workers: int = 1) -> dict:
    with SolverVerifier(workers) as verifier:
        return _generate(config, output, progress, verifier)
