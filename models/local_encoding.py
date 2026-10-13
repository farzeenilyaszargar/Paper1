"""Binding-normalized control, NOT raw cross-representation understanding.

A text-only grammar parser names atoms with shared local slot IDs. It does not
derive a single fact. Randomized slots cover the same vocabulary at every depth.
"""

import random

from synthetic_dataset_generator.generator import digest
from synthetic_dataset_generator.logic import Rule, atom, canonical_problem
from .parsing import parse
from .representations import BOS, EOS

SLOTS = 64
NEG, FACTS, RULES, QUERY, LP, RP, AND, OR, IMPLIES, SEP = range(64,74)


def encode_slots(text, representation):
    meta = parse(text, representation)["metadata"]
    rules = [Rule.from_dict(r) for r in meta["rules"]]
    literals = meta["facts"] + [meta["query"]]
    for rule in rules:
        literals.extend([rule.conclusion, *[p for c in rule.clauses for p in c]])
    symbols = sorted({atom(p) for p in literals})
    if len(symbols) > SLOTS:
        raise ValueError("Local slot capacity exceeded; never merge distinct atoms")
    rng = random.Random(int(digest(canonical_problem(meta["facts"],rules,meta["query"])),16))
    mapping = dict(zip(symbols,rng.sample(range(SLOTS),len(symbols))))

    def literal(value):
        return ([NEG] if value.startswith("~") else []) + [mapping[atom(value)]]

    tokens = [BOS, FACTS]
    for fact in meta["facts"]:
        tokens.extend([*literal(fact), SEP])
    tokens.append(RULES)
    for rule in rules:
        tokens.append(LP)
        for i, clause in enumerate(rule.clauses):
            if i:
                tokens.append(OR)
            tokens.append(LP)
            for j, premise in enumerate(clause):
                if j:
                    tokens.append(AND)
                tokens.extend(literal(premise))
            tokens.append(RP)
        tokens.extend([RP, IMPLIES, *literal(rule.conclusion), SEP])
    return tokens + [QUERY, *literal(meta["query"]), EOS]
