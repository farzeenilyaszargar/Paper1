"""Unlabeled TRAIN-only language-model warmup with matched token/step budgets.

The tied LM projection adds no parameters. Classification labels/proofs are never
rendered. Packed windows may cross/cut problem boundaries: this is an auxiliary
next-token task, not truncated answer-only inference. Both arms use the same
window size, batch size and step count, but may consume different record counts.
"""

import argparse
import json
import random
from itertools import islice
from pathlib import Path

import torch
from torch.nn import functional as F

from synthetic_dataset_generator.generator import file_checksum,write_json
from .representations import PAD,REPRESENTATIONS,encode,format_input
from .train import save_checkpoint,training_space_preflight
from .transformer import SIZES,TinyTransformer,expected_parameter_count


class TrainingStream:
    def __init__(self,path,source,treatment,seed,buffer_records=256):
        if path.name != "train.jsonl" or treatment not in ("source_only","all_formats"):
            raise ValueError("Warmup only accepts train.jsonl and a declared treatment")
        self.path,self.source,self.treatment = path,source,treatment
        self.seed,self.buffer_records = seed,buffer_records
        self.records_started = 0
        self.form_counts = dict.fromkeys(REPRESENTATIONS,0)

    def tokens(self):
        shuffle_rng,format_rng = random.Random(self.seed),random.Random(self.seed+303)
        forms = [self.source] if self.treatment == "source_only" else list(REPRESENTATIONS)
        cycle = []
        while True:
            count = 0
            with self.path.open() as handle:
                reader = map(json.loads,handle)
                while rows := list(islice(reader,self.buffer_records)):
                    shuffle_rng.shuffle(rows)
                    for row in rows:
                        if not cycle:
                            cycle = forms.copy()
                            format_rng.shuffle(cycle)
                        representation = cycle.pop()
                        self.form_counts[representation] += 1
                        self.records_started += 1
                        count += 1
                        # format_input reads only facts/rules/query, not targets.
                        yield from encode(format_input(row,representation))
            if count == 0:
                raise ValueError("Empty warmup training corpus")


def language_logits(model,tokens):
    if tokens.shape[1] > len(model.positions):
        raise ValueError("Warmup context exceeded")
    hidden = model.embedding(tokens)+model.positions[:tokens.shape[1]]
    for block in model.blocks:
        hidden = block(hidden)
    logits = F.linear(model.norm(hidden),model.embedding.weight)
    logits[...,PAD] = torch.finfo(logits.dtype).min
    return logits


def run(args):
    training_space_preflight(args.output,expected_parameter_count(args.size))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    manifest_path = args.data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads((args.data/"audit.json").read_text())
    if not audit.get("passed") or not audit.get("classical_solver_verified") or audit["manifest_sha256"] != file_checksum(manifest_path):
        raise ValueError("Requires a passing parent-corpus audit")
    train_path = args.data/"train.jsonl"
    if file_checksum(train_path) != manifest["files"]["train.jsonl"]["sha256"]:
        raise ValueError("Training corpus changed")
    model = TinyTransformer(args.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    config = {"size":args.size,"parameter_count":model.parameter_count,"seed":args.seed,
              "source":args.source,"treatment":args.treatment,"steps":args.steps,"batch_size":args.batch_size,
              "context":args.context,"lr":args.lr,"device":device,"threads":args.threads,
              "torch_version":str(torch.__version__),"dataset_manifest_sha256":file_checksum(manifest_path),
              "train_sha256":file_checksum(train_path),"token_budget":args.steps*args.batch_size*args.context,
              "warmup_code_sha256":file_checksum(Path(__file__)),
              "transformer_sha256":file_checksum(Path(__file__).with_name("transformer.py")),
              "renderer_sha256":file_checksum(Path(__file__).with_name("representations.py"))}
    report_path = args.output/"warmup_report.json"
    if report_path.exists():
        raise FileExistsError("A completed warmup already exists; do not overwrite it")
    args.output.mkdir(parents=True,exist_ok=args.resume)
    start_step,input_ids = 0,set()
    if args.resume:
        old = json.loads((args.output/"config.json").read_text())
        if old != config:
            raise ValueError("Warmup resume configuration/code mismatch")
        state = torch.load(args.output/"last.pt",map_location=device,weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        start_step,input_ids = state["step"],set(state["input_token_ids_observed"])
    else:
        write_json(args.output/"config.json",config)
    stream = TrainingStream(train_path,args.source,args.treatment,args.seed)
    tokens = stream.tokens()
    block_size = args.batch_size*args.context+1
    # Deterministic replay restores corpus position without reading validation/tests.
    for _ in range(start_step):
        for _ in range(block_size):
            next(tokens)
    model.train()
    with (args.output/"history.jsonl").open("a") as history:
        for step in range(start_step+1,args.steps+1):
            block = torch.tensor(list(islice(tokens,block_size)),dtype=torch.long,device=device)
            x,y = block[:-1].reshape(args.batch_size,args.context),block[1:].reshape(args.batch_size,args.context)
            if bool((y == PAD).any()):
                raise ValueError("Packed LM targets must never be PAD")
            input_ids.update(x.detach().cpu().unique().tolist())
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(language_logits(model,x).flatten(0,1),y.flatten())
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite warmup loss; no initialization may be published")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            if step%100 == 0 or step == args.steps:
                row = {"step":step,"last_batch_loss":float(loss.detach().cpu()),
                       "predicted_tokens":step*args.batch_size*args.context,
                       "records_started":stream.records_started,"forms_rendered":stream.form_counts}
                history.write(json.dumps(row)+"\n")
                history.flush()
                save_checkpoint({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
                                 "torch_rng":torch.get_rng_state(),"input_token_ids_observed":sorted(input_ids)},args.output/"last.pt")
                print(json.dumps(row),flush=True)
    initialization = args.output/"initialization.pt"
    save_checkpoint({"model":model.state_dict(),"config":config,"completed_steps":args.steps},initialization)
    forms = {args.source} if args.treatment == "source_only" else set(REPRESENTATIONS)
    result = {"status":"completed","config":config,"initialization_sha256":file_checksum(initialization),
              "input_token_ids_observed":sorted(input_ids),"forms_rendered":stream.form_counts,
              "all_requested_formats_observed":all(stream.form_counts[rep] > 0 for rep in forms),
              "records_started":stream.records_started,"stream_tokens_consumed":args.steps*block_size,
              "interpretation":"Unlabeled TRAIN-only packed-window LM exposure; no task accuracy measured. Both arms match prediction tokens/steps/window/batch, not necessarily problem counts or semantics. All-format exposure is richer than character coverage alone and is not zero-exposure transfer. Classifier optimizer is reset for subsequent source-only supervision."}
    write_json(report_path,result)
    return result


def initialization(path,size,seed,source,fingerprint):
    report = json.loads((path.parent/"warmup_report.json").read_text())
    state = torch.load(path,map_location="cpu",weights_only=True)
    config = state["config"]
    if report["status"] != "completed" or report["initialization_sha256"] != file_checksum(path) or report["config"] != config:
        raise ValueError("Incomplete or changed warmup provenance")
    if not report["all_requested_formats_observed"] or state["completed_steps"] != config["steps"]:
        raise ValueError("Warmup budget/exposure is incomplete")
    if (config["size"],config["seed"],config["source"],config["dataset_manifest_sha256"]) != (size,seed,source,fingerprint):
        raise ValueError("Warmup architecture/seed/source/corpus mismatch")
    return state["model"],report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--size",choices=SIZES,default="100k")
    parser.add_argument("--source",choices=REPRESENTATIONS,default="symbolic")
    parser.add_argument("--treatment",choices=("source_only","all_formats"),required=True)
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--steps",type=int,default=1000)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--context",type=int,default=256)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--device",choices=("auto","cpu","mps"),default="auto")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if min(args.steps,args.batch_size,args.context,args.threads) < 1 or args.context > 4096 or args.lr <= 0:
        parser.error("Positive budgets/rate required; context cannot exceed 4096")
    run(args)
