"""Lossless alternative renderings of the same signed-literal reasoning task."""

import string

from synthetic_dataset_generator.logic import Rule, render

REPRESENTATIONS = ("symbolic", "function", "english")
# Shared fixed alphabet: novelty in token spellings is not novelty in token IDs.
CHARACTERS = list(string.printable[:95])
CHAR_IDS = {c: i for i, c in enumerate(CHARACTERS)}
PAD, BOS, EOS = range(len(CHARACTERS), len(CHARACTERS) + 3)
VOCAB_SIZE = len(CHARACTERS) + 3
LABELS = ("entailed", "contradicted", "unknown")


def encode(text):
    try:
        return [BOS, *[CHAR_IDS[c] for c in text], EOS]
    except KeyError as exc:
        raise ValueError(f"Unsupported character {exc.args[0]!r}") from exc


def format_input(record, representation):
    """Read only facts/rules/query; never use proof or target fields."""
    meta = record["metadata"]
    facts, query = meta["facts"], meta["query"]
    rules = [Rule.from_dict(r) for r in meta["rules"]]
    if representation == "symbolic":
        return render(facts, rules, query)
    if representation == "function":
        def lit(value):
            return f"not({value[1:]})" if value.startswith("~") else value

        expressions = []
        for rule in rules:
            terms = [lit(c[0]) if len(c) == 1 else f"and({','.join(map(lit, c))})" for c in rule.clauses]
            premise = terms[0] if len(terms) == 1 else f"or({','.join(terms)})"
            expressions.append(f"implies({premise},{lit(rule.conclusion)})")
        return f"facts({','.join(map(lit, facts))});rules({';'.join(expressions)});query({lit(query)})"
    if representation == "english":
        def lit(value):
            return f"not {value[1:]}" if value.startswith("~") else value

        expressions = []
        for rule in rules:
            terms = [" and ".join(map(lit, c)) for c in rule.clauses]
            premise = terms[0] if len(terms) == 1 else " or ".join(f"({t})" for t in terms)
            expressions.append(f"If {premise}, then {lit(rule.conclusion)}.")
        return f"Known: {', '.join(map(lit, facts))}. {' '.join(expressions)} Query: {lit(query)}."
    raise ValueError(f"Unknown representation: {representation}")
