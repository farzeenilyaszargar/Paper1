"""Generate exactly length-matched, solver-checked depth diagnostics.

Only held-out IID/depth records are used. Each pair shares a gold label and
surface grammar, not a knowledge base. Disconnected logical components pad BOTH
members to a common length. This controls token count, not all distractor cues.
"""

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

from synthetic_dataset_generator.generator import (LABELS, digest, file_checksum,
    group_id, permitted_targets, write_json)
from synthetic_dataset_generator.logic import (Rule, atom, classical_label,
    forward_label, has_reconvergence, dependency_edges, infer, negate)
from .parsing import parse
from .representations import REPRESENTATIONS, encode, format_input


def templates():
    result = [(["p990"],[]),(["~p990"],[])]
    for kind in ("unary","and","or"):
        for root_negative in (False,True):
            for conclusion_negative in (False,True):
                a = ("~" if root_negative else "")+"p990"
                b = ("~" if root_negative else "")+"p991"
                c = ("~" if conclusion_negative else "")+("p991" if kind == "unary" else "p992")
                facts = [a] if kind == "unary" else [a,b]
                clauses = ((a,),) if kind == "unary" else ((a,b),) if kind == "and" else ((a,),(b,))
                result.append((facts,[Rule(clauses,c)]))
    return tuple(result)


TEMPLATES = templates()


def record(facts,rules,query):
    return {"metadata":{"facts":facts,"rules":[r.to_dict() for r in rules],"query":query}}


@lru_cache(maxsize=3)
def padding_table(representation):
    # All source corpora have nonempty facts AND rules; insertion costs are fixed.
    base = record(["p000"],[Rule((("p000",),),"p001")],"p001")
    length = len(format_input(base,representation))
    costs,atoms = [],[]
    for facts,rules in TEMPLATES:
        costs.append(len(format_input(record(["p000"]+facts,[Rule((("p000",),),"p001")]+rules,"p001"),representation))-length)
        atoms.append(len({atom(f) for f in facts}|{atom(r.conclusion) for r in rules}))
    # Minimize fresh symbols, then component count, with deterministic tie breaks.
    scores,previous = [(0,0)]+[None]*4094,[None]*4095
    for gap in range(1,4095):
        for i,cost in enumerate(costs):
            if cost <= gap and scores[gap-cost] is not None:
                candidate = (scores[gap-cost][0]+atoms[i],scores[gap-cost][1]+1)
                if scores[gap] is None or candidate < scores[gap]:
                    scores[gap],previous[gap] = candidate,i
    return tuple(costs),tuple(scores),tuple(previous)


def padding_plan(gap,representation):
    costs,scores,previous = padding_table(representation)
    if not 0 <= gap < len(scores) or scores[gap] is None:
        raise ValueError("Exact logical padding is unavailable within context")
    result = []
    while gap:
        index = previous[gap]
        result.append(index)
        gap -= costs[index]
    return result


def pad(text,representation,length,seed,dataset_seed=42,familiar_symbols=512):
    parsed = parse(text,representation)["metadata"]
    facts,query = list(parsed["facts"]),parsed["query"]
    rules = [Rule.from_dict(r) for r in parsed["rules"]]
    if not facts or not rules or format_input(record(facts,rules,query),representation) != text:
        raise ValueError("Padding requires canonical nonempty source input")
    used = {atom(f) for f in facts+[query]}|{atom(v) for r in rules for c in r.clauses for v in c}|{atom(r.conclusion) for r in rules}
    available = {f"p{i:03}" for i in range(familiar_symbols)}-used
    if not used <= {f"p{i:03}" for i in range(familiar_symbols)}:
        raise ValueError("Depth control must use familiar names only")
    before = infer(facts,rules)[0]
    target = classical_label(facts,rules,query)
    decisive = query if target == "entailed" else negate(query) if target == "contradicted" else None
    depth = before.get(decisive) if decisive is not None else None
    rng = random.Random(seed)
    fact_count,rule_count = len(facts),len(rules)
    for index in padding_plan(length-len(text),representation):
        new_facts,new_rules = TEMPLATES[index]
        mapping = {}
        for f in new_facts:
            if not available:
                raise ValueError("Padding exhausted familiar symbols")
            chosen = rng.choice(sorted(available))
            available.remove(chosen)
            mapping[atom(f)] = chosen
        for rule in new_rules:
            candidates = available.copy()
            for clause in rule.clauses:
                for premise in clause:
                    candidates.intersection_update(permitted_targets(dataset_seed,mapping[atom(premise)],0,familiar_symbols,False))
            if not candidates:
                raise ValueError("Cannot allocate nonreserved distractor dependencies")
            chosen = rng.choice(sorted(candidates))
            available.remove(chosen)
            mapping[atom(rule.conclusion)] = chosen

        def rename(v):
            return ("~" if v.startswith("~") else "")+mapping[atom(v)]

        facts.extend(map(rename,new_facts))
        rules.extend(Rule(tuple(tuple(rename(v) for v in c) for c in r.clauses),rename(r.conclusion)) for r in new_rules)
    rng.shuffle(facts)
    rng.shuffle(rules)
    changed = record(facts,rules,query)
    rendered = format_input(changed,representation)
    after = infer(facts,rules)[0]
    if len(rendered) != length or forward_label(after,query) != target or classical_label(facts,rules,query) != target:
        raise ValueError("Length/label invariant failed")
    if decisive is not None and after[decisive] != depth:
        raise ValueError("Distractors changed minimum decisive proof depth")
    if has_reconvergence(dependency_edges(rules)):
        raise ValueError("Depth control unexpectedly contains a held-out graph family")
    return {"input":rendered,"proof_depth":depth,"group_id":group_id(facts,rules),
            "added_facts":len(facts)-fact_count,"added_rules":len(rules)-rule_count,
            "target":target,"input_tokens":len(encode(rendered))}


def make_pair(short,long,representation,depth,seed,dataset_seed=42,familiar_symbols=512):
    if short["target"] != long["target"]:
        raise ValueError("Length-control pairs must share the same gold label")
    texts = [format_input(r,representation) for r in (short,long)]
    costs,scores,_ = padding_table(representation)
    length = max(map(len,texts))+2*max(costs)
    while length <= 4094 and not all(scores[length-len(t)] is not None for t in texts):
        length += 1
    if length > 4094:
        raise ValueError("No common exact length within the 4096-token context")
    identity = digest(["length-control-v1",seed,short["id"],long["id"],representation,depth])
    padded = [pad(text,representation,length,int(digest([identity,side])[:16],16),dataset_seed,familiar_symbols)
              for side,text in zip(("short","long"),texts)]
    for item,expected in zip(padded,(2,depth)):
        if item["target"] != short["target"] or item["proof_depth"] != (None if item["target"] == "unknown" else expected):
            raise ValueError("Incorrect original depth or label")
    if padded[0]["input_tokens"] != padded[1]["input_tokens"]:
        raise ValueError("Character-token lengths differ")
    return {"id":identity,"representation":representation,"target":short["target"],
            "matched_scaffold_depths":[2,depth],"source_ids":[short["id"],long["id"]],
            "short":padded[0],"long":padded[1]}


def build(data,output,limit=None,seed=142):
    if output.exists():
        raise FileExistsError("Choose a new directory; control data is never overwritten")
    manifest_path = data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((data/"audit.json").read_text())
    if not audit.get("passed") or not audit.get("classical_solver_verified") or audit["manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Requires a passing parent-corpus audit")
    source_files = ("train.jsonl","test_iid.jsonl","test_depth_3.jsonl","test_depth_4.jsonl","test_depth_5.jsonl")
    for filename in source_files:
        if file_checksum(data/filename) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Changed parent file: {filename}")
    # Original training KBs are forbidden, including alternate queries/orderings.
    with (data/"train.jsonl").open() as handle:
        training_groups = {json.loads(line)["metadata"]["group_id"] for line in handle}

    def load_grouped(filename):
        grouped = {label:[] for label in LABELS}
        with (data/filename).open() as handle:
            for line in handle:
                row = json.loads(line)
                grouped[row["target"]].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda r:(len(r["input"]),r["id"]))
        return grouped

    iid = load_grouped("test_iid.jsonl")
    output.mkdir(parents=True)
    files = {}
    for depth in (3,4,5):
        deeper = load_grouped(f"test_depth_{depth}.jsonl")
        if any(len(iid[label]) != len(deeper[label]) for label in LABELS):
            raise ValueError("Parent label counts must match for complete within-label pairing")
        # Pair once, before choosing any evaluation representation or accuracy.
        pairs = [(a,b) for label in LABELS for a,b in zip(iid[label],deeper[label])]
        random.Random(seed+depth).shuffle(pairs)
        if limit is not None:
            pairs = pairs[:limit]
        for rep in REPRESENTATIONS:
            path = output/f"depth_{depth}_{rep}.jsonl"
            counts = dict.fromkeys(LABELS,0)
            min_length,max_length = 4096,0
            with path.open("w") as handle:
                for index,(short,long) in enumerate(pairs):
                    row = make_pair(short,long,rep,depth,seed,manifest["config"]["seed"],manifest["config"]["familiar_symbols"])
                    if any(row[side]["group_id"] in training_groups for side in ("short","long")):
                        raise ValueError("Padded control KB leaked from training")
                    handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
                    counts[row["target"]] += 1
                    length = row["short"]["input_tokens"]
                    min_length,max_length = min(min_length,length),max(max_length,length)
                    if (index+1)%1000 == 0:
                        print(f"Length control {depth}/{rep}: {index+1} pairs",flush=True)
            files[path.name] = {"count":len(pairs),"labels":counts,"minimum_tokens":min_length,
                                "maximum_tokens":max_length,"sha256":file_checksum(path)}
    report = {"status":"generated_and_solver_checked","scope":"full held-out length control" if limit is None else "held-out length-control subset",
              "parent_manifest_sha256":file_checksum(manifest_path),"parent_files":{name:manifest["files"][name]["sha256"] for name in source_files},
              "code_sha256":file_checksum(Path(__file__)),"seed":seed,"limit":limit,"files":files,
              "pairing":"Within gold label, sorted by original symbolic length then ID; seeded pair shuffle; pairing independent of model predictions",
              "interpretation":"Same label and exact character-token length within each pair/grammar; different core KBs. Both sides padded with disconnected familiar-name, nonreserved-pair forest components. Controls length, not all graph, fact-count, operator, position or distractor cues. Unknown proof depths remain null. Derived test controls must never be used for training or tuning."}
    write_json(output/"manifest.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--output",type=Path,default=Path("datasets/length_matched"))
    parser.add_argument("--limit",type=int)
    parser.add_argument("--seed",type=int,default=142)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Optional pair limit must be positive")
    result = build(args.data,args.output,args.limit,args.seed)
    print(f"Generated {sum(v['count'] for v in result['files'].values())} solver-checked pairs")
