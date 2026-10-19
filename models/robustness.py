"""Solver-checked raw-input interventions, independent of proof metadata.

Evaluate on CPU alongside the sequential accelerator queue. These are diagnostic
test interventions, never training examples or hyperparameter-selection data.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from synthetic_dataset_generator.generator import file_checksum, write_json
from synthetic_dataset_generator.logic import Rule, atom, classical_label, forward_label, infer, negate
from .parsing import parse
from .representations import LABELS, encode, format_input
from .train import batches
from .transformer import TinyTransformer

NAMES = ("original", "rename", "statement_order", "irrelevant_components", "query_negation", "support_fact_ablation")


def checked_record(facts, rules, query):
    target = classical_label(facts,rules,query)
    if forward_label(infer(facts,rules)[0],query) != target:
        raise ValueError("Intervention left the forward/classical agreement fragment")
    return {"target":target, "metadata":{"facts":facts,"rules":[r.to_dict() for r in rules],"query":query}}


def variants(text, representation, seed, familiar_symbols=512):
    """Only parse input text. Gold/proof/depth metadata is never consulted."""
    meta = parse(text,representation)["metadata"]
    facts, query = list(meta["facts"]), meta["query"]
    rules = [Rule.from_dict(r) for r in meta["rules"]]
    original = checked_record(facts,rules,query)
    result = {"original":original}
    rng = random.Random(seed)
    used = sorted({atom(v) for v in facts+[query]+[v for r in rules for c in r.clauses for v in c]+[r.conclusion for r in rules]})
    pool = [f"p{i:03}" for i in range(familiar_symbols)]
    mapping = dict(zip(used,rng.sample(pool,len(used))))

    def rename(v):
        return ("~" if v.startswith("~") else "")+mapping[atom(v)]

    renamed_rules = [Rule(tuple(tuple(rename(v) for v in c) for c in r.clauses),rename(r.conclusion)) for r in rules]
    result["rename"] = checked_record(list(map(rename,facts)),renamed_rules,rename(query))
    shuffled_facts, shuffled_rules = list(facts), []
    rng.shuffle(shuffled_facts)
    for rule in rules:
        clauses = [list(c) for c in rule.clauses]
        for clause in clauses:
            rng.shuffle(clause)
        rng.shuffle(clauses)
        shuffled_rules.append(Rule(tuple(tuple(c) for c in clauses),rule.conclusion))
    rng.shuffle(shuffled_rules)
    result["statement_order"] = checked_record(shuffled_facts,shuffled_rules,query)
    fresh = rng.sample(sorted(set(pool)-set(used)),9)
    added_rules, added_facts = list(rules), list(facts)
    for start in range(0,9,3):
        a,b,c = fresh[start:start+3]
        added_facts.append(a)
        added_rules.extend([Rule(((a,),),b),Rule(((b,),),c)])
    rng.shuffle(added_facts)
    rng.shuffle(added_rules)
    result["irrelevant_components"] = checked_record(added_facts,added_rules,query)
    result["query_negation"] = checked_record(facts,rules,negate(query))
    # Find an irredundant fact cut. Multiple OR branches may require >1 removal.
    # Unknown originals have no decisive support to ablate and are not scored.
    if original["target"] != "unknown":
        order = list(range(len(facts)))
        rng.shuffle(order)
        removed = set()
        for i in order:
            removed.add(i)
            if classical_label([f for j,f in enumerate(facts) if j not in removed],rules,query) != original["target"]:
                break
        for i in sorted(removed):
            trial = removed-{i}
            if classical_label([f for j,f in enumerate(facts) if j not in trial],rules,query) != original["target"]:
                removed = trial
        changed = checked_record([f for j,f in enumerate(facts) if j not in removed],rules,query)
        if changed["target"] != "unknown":
            raise ValueError("Removing supporting facts must turn a decisive answer into unknown")
        changed["removed_fact_count"] = len(removed)
        result["support_fact_ablation"] = changed
    for name in ("rename","statement_order","irrelevant_components"):
        if result[name]["target"] != original["target"]:
            raise ValueError(f"Label-preserving intervention changed semantics: {name}")
    expected = {"entailed":"contradicted","contradicted":"entailed","unknown":"unknown"}
    if result["query_negation"]["target"] != expected[original["target"]]:
        raise ValueError("Query negation violated three-way semantics")
    return result


@torch.no_grad()
def predict(model, samples, batch_size):
    model.eval()
    predictions = {}
    for tokens,lengths,targets,ids in batches(samples,batch_size):
        for identifier,prediction in zip(ids,model(tokens,lengths).argmax(-1).tolist()):
            predictions[identifier] = LABELS[prediction]
    return predictions


def run(directory, data, limit=None, batch_size=8, threads=1):
    torch.set_num_threads(threads)
    result_path = directory/"results.json"
    saved = json.loads(result_path.read_text())
    if saved["status"] != "completed" or saved["config"].get("encoding","characters") != "characters":
        raise ValueError("Requires a completed raw-character run, not a parser-normalized control")
    config = saved["config"]
    manifest_path = data/"manifest.json"
    if config["dataset_manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Dataset fingerprint does not match the checkpoint's experiment")
    manifest = json.loads(manifest_path.read_text())
    corpus_path = data/"test_iid.jsonl"
    if file_checksum(corpus_path) != manifest["files"]["test_iid.jsonl"]["sha256"]:
        raise ValueError("IID test corpus checksum mismatch")
    output = directory/"robustness.json"
    if output.exists():
        raise FileExistsError("Robustness report already exists; preserve prior measurements")
    checkpoint = directory/"best.pt"
    model = TinyTransformer(config["size"])
    if model.parameter_count != config["parameter_count"]:
        raise ValueError("Checkpoint architecture differs from current model")
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    samples = {name:[] for name in NAMES}
    labels, removed_counts = {}, []
    source = config["representation"]
    with corpus_path.open() as handle:
        for index,line in enumerate(handle):
            if limit and index >= limit:
                break
            record = json.loads(line)
            seed = int(hashlib.sha256((record["id"]+":robustness-v1").encode()).hexdigest()[:16],16)
            changed = variants(format_input(record,source),source,seed,manifest["config"]["familiar_symbols"])
            if changed["original"]["target"] != record["target"]:
                raise ValueError("Original input gold label differs from solver")
            for name,example in changed.items():
                identifier = record["id"]
                samples[name].append((encode(format_input(example,source)),LABELS.index(example["target"]),identifier))
                labels[name,identifier] = example["target"]
                if name == "support_fact_ablation":
                    removed_counts.append(example["removed_fact_count"])
            if (index+1)%100 == 0:
                print(f"Solver-checked interventions: {index+1}",flush=True)
    predictions, reports = {}, {}
    prediction_path = directory/"robustness.predictions.jsonl"
    with prediction_path.open("w") as handle:
        for name in NAMES:
            predictions[name] = predict(model,samples[name],batch_size)
            confusion = [[0]*3 for _ in LABELS]
            unchanged = joint_correct = original_correct = gold_changed = followed_gold_change = 0
            for tokens,_,identifier in samples[name]:
                target, prediction = labels[name,identifier],predictions[name][identifier]
                before = predictions["original"][identifier]
                before_gold = labels["original",identifier]
                confusion[LABELS.index(target)][LABELS.index(prediction)] += 1
                unchanged += before == prediction
                original_correct += before == before_gold
                joint_correct += before == before_gold and prediction == target
                gold_changed += target != before_gold
                followed_gold_change += target != before_gold and before == before_gold and prediction == target
                handle.write(json.dumps({"id":identifier,"variant":name,"target":target,"prediction":prediction})+"\n")
            count = len(samples[name])
            reports[name] = {"count":count,"not_applicable":len(samples["original"])-count,
                             "accuracy":sum(confusion[i][i] for i in range(3))/count if count else None,
                             "confusion":confusion,"original_correct_on_applicable":original_correct,
                             "jointly_correct":joint_correct,"unchanged_predictions":unchanged,
                             "gold_label_changed":gold_changed,"correctly_followed_gold_change":followed_gold_change,
                             "mean_input_tokens":sum(len(s[0]) for s in samples[name])/count if count else None}
            print(f"Evaluated {name}: {reports[name]}",flush=True)
    report = {"run":directory.name,"scope":"full IID robustness diagnostic" if limit is None else "IID robustness subset diagnostic",
              "interpretation":"High prediction invariance without joint correctness can reflect a constant classifier, not reasoning. Test diagnostics are not used for hyperparameter selection. This distractor intervention is not a length-matched depth control.",
              "representation":source,"source_learning_gate_passed":saved.get("source_learning_gate_passed"),
              "result_sha256":file_checksum(result_path),"checkpoint_sha256":file_checksum(checkpoint),
              "prediction_sha256":file_checksum(prediction_path),"probe_code_sha256":file_checksum(Path(__file__)),
              "dataset_manifest_sha256":file_checksum(manifest_path),"iid_sha256":file_checksum(corpus_path),
              "seed_rule":"first 16 hex digits of SHA256(instance ID + :robustness-v1)",
              "batch_size":batch_size,"threads":threads,"device":"cpu","limit":limit,
              "removed_fact_count_total":sum(removed_counts),"variants":reports}
    write_json(output,report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--limit",type=int)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--threads",type=int,default=1)
    args = parser.parse_args()
    if min(args.batch_size,args.threads) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, threads and optional limit must be positive")
    run(args.run_directory,args.data,args.limit,args.batch_size,args.threads)


if __name__ == "__main__":
    main()
