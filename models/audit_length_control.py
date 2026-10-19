"""Independently parse and verify the stored length-matched control corpus."""

import argparse
import json
from pathlib import Path

from synthetic_dataset_generator.generator import (LABELS,digest,file_checksum,group_id,
    permitted_targets,write_json)
from synthetic_dataset_generator.logic import (Rule,atom,canonical_problem,classical_label,
    dependency_edges,forward_label,has_reconvergence,infer,negate)
from .parsing import parse
from .representations import REPRESENTATIONS,encode,format_input


def literals(facts,rules,query):
    return {atom(v) for v in facts+[query]}|{atom(v) for r in rules for c in r.clauses for v in c}|{atom(r.conclusion) for r in rules}


def verify_member(member,parent,representation,label,depth,seed,familiar_symbols):
    meta = parse(member["input"],representation)["metadata"]
    facts,query = meta["facts"],meta["query"]
    rules = [Rule.from_dict(r) for r in meta["rules"]]
    if member["target"] != label or len(encode(member["input"])) != member["input_tokens"] or member["input_tokens"] > 4096:
        raise ValueError("Incorrect stored target or token length")
    if format_input({"metadata":meta},representation) != member["input"]:
        raise ValueError("Control text must be canonical, not padded with filler whitespace")
    base = parent["metadata"]
    base_rules = [Rule.from_dict(r) for r in base["rules"]]
    original = canonical_problem(base["facts"],base_rules,base["query"])
    changed = canonical_problem(facts,rules,query)
    if query != base["query"] or not set(original["facts"]) <= set(changed["facts"]) or not set(original["rules"]) <= set(changed["rules"]):
        raise ValueError("Padding removed or changed the original core")
    old_atoms = literals(base["facts"],base_rules,base["query"])
    for f in set(facts)-set(base["facts"]):
        if atom(f) in old_atoms:
            raise ValueError("Added fact touches the original core")
    original_rules = set(original["rules"])
    for rule in rules:
        if canonical_problem([], [rule], "")["rules"][0] not in original_rules:
            if literals([], [rule], rule.conclusion) & old_atoms:
                raise ValueError("Added rule is not disconnected from the core")
    if not literals(facts,rules,query) <= {f"p{i:03}" for i in range(familiar_symbols)}:
        raise ValueError("Control includes held-out symbol names")
    edges = dependency_edges(rules)
    if has_reconvergence(edges) or any(b not in permitted_targets(seed,a,0,familiar_symbols,False) for a,b in edges):
        raise ValueError("Unexpected graph-family or reserved-pair novelty")
    closure = infer(facts,rules)[0]
    if forward_label(closure,query) != label or classical_label(facts,rules,query) != label:
        raise ValueError("Forward/classical control label mismatch")
    decisive = query if label == "entailed" else negate(query) if label == "contradicted" else None
    actual_depth = closure[decisive] if decisive is not None else None
    if actual_depth != member["proof_depth"] or actual_depth != (None if label == "unknown" else depth):
        raise ValueError("Incorrect minimum decisive depth")
    if parent["target"] != label or base["matched_depth"] != depth:
        raise ValueError("Incorrect source label/scaffold depth")
    if member["group_id"] != group_id(facts,rules):
        raise ValueError("Incorrect padded knowledge-base group ID")
    if member["added_facts"] != len(facts)-len(base["facts"]) or member["added_rules"] != len(rules)-len(base_rules):
        raise ValueError("Incorrect distractor counts")
    return member["group_id"]


def audit(directory,data):
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parent_path = data/"manifest.json"
    parent = json.loads(parent_path.read_text())
    if file_checksum(parent_path) != manifest["parent_manifest_sha256"]:
        raise ValueError("Parent fingerprint changed")
    if set(manifest["parent_files"]) != {"train.jsonl","test_iid.jsonl","test_depth_3.jsonl","test_depth_4.jsonl","test_depth_5.jsonl"}:
        raise ValueError("Missing parent provenance conditions")
    for name,checksum in manifest["parent_files"].items():
        if file_checksum(data/name) != checksum or parent["files"][name]["sha256"] != checksum:
            raise ValueError("Parent corpus checksum mismatch")
    expected = {f"depth_{depth}_{rep}.jsonl" for depth in (3,4,5) for rep in REPRESENTATIONS}
    if set(manifest["files"]) != expected:
        raise ValueError("Missing or extra control conditions")
    with (data/"train.jsonl").open() as handle:
        training_groups = {json.loads(line)["metadata"]["group_id"] for line in handle}

    def load(name):
        with (data/name).open() as handle:
            return {row["id"]:row for row in map(json.loads,handle)}

    iid = load("test_iid.jsonl")
    summaries = {}
    all_ids = set()
    for depth in (3,4,5):
        deeper = load(f"test_depth_{depth}.jsonl")
        for rep in REPRESENTATIONS:
            name = f"depth_{depth}_{rep}.jsonl"
            info = manifest["files"][name]
            if file_checksum(directory/name) != info["sha256"]:
                raise ValueError("Control file checksum mismatch")
            seen_short,seen_long = set(),set()
            counts = dict.fromkeys(LABELS,0)
            lengths,added_facts,added_rules = [],[0,0],[0,0]
            with (directory/name).open() as handle:
                for index,line in enumerate(handle):
                    row = json.loads(line)
                    a,b = row["source_ids"]
                    identity = digest(["length-control-v1",manifest["seed"],a,b,rep,depth])
                    if row["id"] != identity or identity in all_ids or a in seen_short or b in seen_long:
                        raise ValueError("Invalid/duplicate pair or source ID")
                    all_ids.add(identity)
                    seen_short.add(a)
                    seen_long.add(b)
                    if row["representation"] != rep or row["matched_scaffold_depths"] != [2,depth]:
                        raise ValueError("Incorrect condition metadata")
                    if row["short"]["input_tokens"] != row["long"]["input_tokens"]:
                        raise ValueError("Pair is not exactly token-length matched")
                    for i,(side,source,d) in enumerate((("short",iid[a],2),("long",deeper[b],depth))):
                        group = verify_member(row[side],source,rep,row["target"],d,parent["config"]["seed"],parent["config"]["familiar_symbols"])
                        if group in training_groups:
                            raise ValueError("Control knowledge base leaked from training")
                        added_facts[i] += row[side]["added_facts"]
                        added_rules[i] += row[side]["added_rules"]
                    counts[row["target"]] += 1
                    lengths.append(row["short"]["input_tokens"])
                    if (index+1)%1000 == 0:
                        print(f"Audited length control {depth}/{rep}: {index+1} pairs",flush=True)
            if len(lengths) != info["count"] or counts != info["labels"] or not lengths:
                raise ValueError("Incorrect corpus counts")
            if min(lengths) != info["minimum_tokens"] or max(lengths) != info["maximum_tokens"]:
                raise ValueError("Incorrect length range")
            if manifest["limit"] is None and (seen_short != iid.keys() or seen_long != deeper.keys()):
                raise ValueError("Incomplete full-control source coverage")
            summaries[name] = {"count":len(lengths),"labels":counts,"minimum_tokens":min(lengths),"maximum_tokens":max(lengths),
                               "mean_tokens":sum(lengths)/len(lengths),"mean_added_facts_short_long":[v/len(lengths) for v in added_facts],
                               "mean_added_rules_short_long":[v/len(lengths) for v in added_rules]}
    report = {"passed":True,"classical_solver_verified":True,"scope":manifest["scope"],
              "manifest_sha256":file_checksum(manifest_path),"parent_manifest_sha256":file_checksum(parent_path),
              "pair_count":len(all_ids),"logical_input_count":2*len(all_ids),"files":summaries,
              "audit_code_sha256":file_checksum(Path(__file__)),
              "generator_source_matches_current":file_checksum(Path(__file__).with_name("length_control.py")) == manifest["code_sha256"],
              "interpretation":manifest["interpretation"]}
    write_json(directory/"audit.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path,nargs="?",default=Path("datasets/length_matched"))
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--summary",type=Path,default=Path("datasets/length_control_summary.json"))
    args = parser.parse_args()
    result = audit(args.directory,args.data)
    write_json(args.summary,result)
    print(f"PASS: {result['pair_count']} pairs / {result['logical_input_count']} logical inputs")
