"""Logic semantics and a classical solver independent of forward inference.

Literals are pNNN or ~pNNN. Antecedents are bounded DNF: a tuple of
conjunctions joined by OR. Each conclusion is a single literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import z3


def negate(literal: str) -> str:
    return literal[1:] if literal.startswith("~") else "~" + literal


def atom(literal: str) -> str:
    return literal.removeprefix("~")


@dataclass(frozen=True)
class Rule:
    clauses: tuple[tuple[str, ...], ...]
    conclusion: str

    def to_dict(self) -> dict:
        return {"antecedent": [list(c) for c in self.clauses],
                "conclusion": self.conclusion}

    @classmethod
    def from_dict(cls, value: dict) -> Rule:
        return cls(tuple(tuple(c) for c in value["antecedent"]), value["conclusion"])


def infer(facts: Iterable[str], rules: list[Rule]) -> tuple[dict, dict]:
    """Minimum parallel rule depth; conjunction introduction has no extra cost."""
    depths = dict.fromkeys(facts, 0)
    proofs = {f: {"kind": "fact"} for f in depths}
    changed = True
    while changed:
        changed = False
        for index, rule in enumerate(rules):
            for clause in rule.clauses:
                if not all(p in depths for p in clause):
                    continue
                depth = 1 + max(depths[p] for p in clause)
                if depth < depths.get(rule.conclusion, float("inf")):
                    depths[rule.conclusion] = depth
                    proofs[rule.conclusion] = {"kind": "rule", "rule_index": index,
                                              "premises": list(clause)}
                    changed = True
    return depths, proofs


def forward_label(depths: dict, query: str) -> str:
    if query in depths and negate(query) in depths:
        raise ValueError("Inconsistent forward closure")
    if query in depths:
        return "entailed"
    if negate(query) in depths:
        return "contradicted"
    return "unknown"


@lru_cache(maxsize=4096)
def _solver_literal(value: str):
    symbol = z3.Bool(atom(value))
    return z3.Not(symbol) if value.startswith("~") else symbol


def classical_label(facts: list[str], rules: list[Rule], query: str) -> str:
    literal = _solver_literal

    solver = z3.Solver()
    assertions = [literal(f) for f in facts]
    for rule in rules:
        terms = [literal(c[0]) if len(c) == 1 else z3.And(*[literal(p) for p in c])
                 for c in rule.clauses]
        antecedent = terms[0] if len(terms) == 1 else z3.Or(*terms)
        assertions.append(z3.Implies(antecedent, literal(rule.conclusion)))
    solver.add(*assertions)
    status = solver.check()
    if status != z3.sat:
        raise ValueError(f"Knowledge base is inconsistent or solver returned {status}")
    q = literal(query)
    positive = solver.check(z3.Not(q))
    negative = solver.check(q)
    if z3.unknown in (positive, negative):
        raise ValueError("Solver returned unknown")
    return "entailed" if positive == z3.unsat else "contradicted" if negative == z3.unsat else "unknown"


def canonical_problem(facts: list[str], rules: list[Rule], query: str) -> dict:
    """Normalizes ordering, duplicate clauses and DNF absorption."""
    normalized = set()
    for rule in rules:
        clauses = {frozenset(c) for c in rule.clauses}
        clauses = {c for c in clauses if not any(other < c for other in clauses)}
        normalized.add((tuple(sorted(tuple(sorted(c)) for c in clauses)), rule.conclusion))
    return {"facts": sorted(set(facts)), "rules": sorted(normalized), "query": query}


def render(facts: list[str], rules: list[Rule], query: str) -> str:
    def antecedent(rule: Rule) -> str:
        terms = [" & ".join(c) if len(c) == 1 else "(" + " & ".join(c) + ")"
                 for c in rule.clauses]
        return terms[0] if len(terms) == 1 else "(" + " | ".join(terms) + ")"

    expressions = [f"{antecedent(r)} -> {r.conclusion}" for r in rules]
    return f"FACTS: {'; '.join(facts)}. RULES: {'; '.join(expressions)}. QUERY: {query}."


def dependency_edges(rules: list[Rule]) -> set[tuple[str, str]]:
    return {(atom(p), atom(r.conclusion)) for r in rules for c in r.clauses for p in c}


def has_reconvergence(edges: set[tuple[str, str]]) -> bool:
    """True if any ancestor has two distinct directed paths to one descendant."""
    nodes = {n for edge in edges for n in edge}
    children = {n: [] for n in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for a, b in edges:
        children[a].append(b)
        indegree[b] += 1
    ready = sorted(n for n in nodes if indegree[n] == 0)
    order = []
    while ready:
        n = ready.pop()
        order.append(n)
        for child in children[n]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("Cyclic dependency graph")
    for root in nodes:
        paths = dict.fromkeys(nodes, 0)
        paths[root] = 1
        for n in order:
            for child in children[n]:
                paths[child] += paths[n]
                if paths[child] > 1:
                    return True
    return False
