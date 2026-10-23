"""Restartable single-threaded CPU lanes for the recorded auxiliary experiments.

These queues never enter the primary baseline family or tune on test results.
--drain-pid adopts no optimizer state: it waits for one verified existing CPU
command to finish, then uses its saved report/checkpoint like any other run.
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from synthetic_dataset_generator.generator import SPLITS, file_checksum, write_json
from .architecture_provenance import verify_transformer_provenance
from .representations import REPRESENTATIONS

LANES = ("calibration_1m", "syntax_supervision")
PARAMETERS = {"100k":99075,"1m":1004547}


def planned_jobs(lane):
    common = dict(representation="symbolic", device="cpu", threads=1,
                  encoding="characters", initialization=None)
    if lane == "calibration_1m":
        return [dict(common, name=f"calibration_1m_symbolic_seed101_lr{tag}",
                     size="1m", seed=101, epochs=10, batch_size=4, accumulation=8,
                     lr=rate, train_limit=20000, eval_limit=2000,
                     purpose="pilot", validation_only=True)
                for tag,rate in (("001",.001),("0003",.0003),("003",.003))]
    if lane != "syntax_supervision":
        raise ValueError("Unknown auxiliary lane")
    return [dict(common, name=f"exposure_full_100k_symbolic_seed{seed}_{arm}_lr003",
                 size="100k", seed=seed, epochs=30, batch_size=16, accumulation=2,
                 lr=.003, train_limit=None, eval_limit=None, purpose="control",
                 validation_only=False,
                 initialization=None if arm == "no_warmup" else
                 f"models/warmups/100k_{arm}_seed{seed}/initialization.pt")
            for seed in (1,2,3) for arm in ("no_warmup","source_only","all_formats")]


def verify_protocol(lane):
    filename = "calibration_1m_protocol.json" if lane == "calibration_1m" else "syntax_exposure_protocol.json"
    path = Path("models")/filename
    protocol = json.loads(path.read_text())
    if lane == "calibration_1m":
        expected = dict(size="1m", parameters=1004547, seed=101, epochs=10, train_limit=20000,
                        validation_limit=2000, learning_rates=[.001,.0003,.003],
                        device="cpu", threads=1, microbatch=4, accumulation=8,
                        validation_only=True, representation="symbolic",encoding="characters")
        actual = protocol
    else:
        actual = protocol["supervised_budget"]
        expected = dict(size="100k", parameters=99075, representation="symbolic",
                        encoding="characters", seeds=[1,2,3], train_examples=100000,
                        validation_examples=10000, epochs=30, microbatch=16,
                        accumulation=2, effective_batch=32, learning_rate=.003,
                        device="cpu", threads=1, optimizer="AdamW", weight_decay=.01,
                        gradient_clip_norm=1.0, model_dropout=0.0, padding_bucket_tokens=128,
                        arms=["source_only","all_formats","no_warmup"])
        selection_path = Path(actual["calibration_selection_report"])
        selection = json.loads(selection_path.read_text())
        if (file_checksum(selection_path) != actual["calibration_selection_sha256"] or
                selection["status"] != "completed_subset_calibration_selection" or
                selection["selected_learning_rate"] != .003):
            raise ValueError("Changed or incomplete supervised learning-rate selection")
    if any(actual.get(key) != value for key,value in expected.items()):
        raise ValueError("Auxiliary queue differs from recorded protocol")
    return path


def matching_config(config, job, fingerprint):
    keys = set(job)-{"name"}
    return (all(config.get(key) == job[key] for key in keys)
            and config.get("dataset_manifest_sha256") == fingerprint
            and config.get("parameter_count") == PARAMETERS[job["size"]]
            and config.get("train_examples") == (job["train_limit"] or 100000)
            and config.get("validation_examples") == (job["eval_limit"] or 10000))


def verify_initialization(job, fingerprint):
    if job["initialization"] is None:
        return None
    path = Path(job["initialization"])
    report = json.loads((path.parent/"warmup_report.json").read_text())
    expected = dict(size=job["size"],seed=job["seed"],source=job["representation"],
                    steps=1000,batch_size=8,context=256,lr=.001,token_budget=2048000,
                    dataset_manifest_sha256=fingerprint,parameter_count=99075,
                    treatment="source_only" if "source_only" in path.parent.name else "all_formats")
    if (report["status"] != "completed" or not report["all_requested_formats_observed"] or
            any(report["config"].get(k) != v for k,v in expected.items()) or
            file_checksum(path) != report["initialization_sha256"]):
        raise ValueError("Warmup does not meet the recorded exposure budget/provenance")
    return report["initialization_sha256"]


def completed(directory, job, fingerprint):
    path = directory/"results.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    if not matching_config(report["config"],job,fingerprint):
        raise ValueError(f"Saved auxiliary configuration disagrees: {directory}")
    status = "validation_only_completed" if job["validation_only"] else "completed"
    if report["status"] != status:
        return False
    verify_transformer_provenance(report["config"])
    if report["config"].get("initialization_sha256") != verify_initialization(job,fingerprint):
        raise ValueError("Saved supervised initialization differs from verified warmup")
    history = [json.loads(line) for line in (directory/"history.jsonl").read_text().splitlines()]
    steps = math.ceil((job["train_limit"] or 100000)/(job["batch_size"]*job["accumulation"]))
    if ([row["epoch"] for row in history] != list(range(1,job["epochs"]+1)) or any(
            row["step"] != row["epoch"]*steps or
            row["training_examples_seen"] != row["epoch"]*(job["train_limit"] or 100000)
            for row in history) or not (directory/"best.pt").is_file()):
        raise ValueError("Completed auxiliary report lacks the retained training budget/checkpoint")
    if job["validation_only"]:
        if report["evaluations"] != {} or list(directory.glob("*.predictions.jsonl")):
            raise ValueError("Validation-only calibration contains test evaluations")
    else:
        splits = set(SPLITS)-{"train","validation"}
        if (set(report["evaluations"]) != set(REPRESENTATIONS) or any(
                set(report["evaluations"][rep]) != splits or any(
                    m["count"] != 10000 for m in report["evaluations"][rep].values())
                for rep in REPRESENTATIONS)):
            raise ValueError("Completed control lacks all 21 full test evaluations")
        if any(not (directory/f"{rep}_{split}.predictions.jsonl").is_file()
               for rep in REPRESENTATIONS for split in splits):
            raise ValueError("Completed control lacks retained prediction files")
    if (not math.isfinite(report["selected_validation"]["loss"]) or
            not math.isclose(report["selected_validation"]["loss"],
                             min(row["validation"]["loss"] for row in history),abs_tol=1e-8,rel_tol=0)):
        raise ValueError("Auxiliary checkpoint was not selected by minimum validation loss")
    return True


def live_training_commands():
    lines = subprocess.check_output(["ps","-axo","args="],text=True).splitlines()
    commands = []
    for line in lines:
        if " -m models.train " not in " "+line+" ":
            continue
        try:
            args = shlex.split(line)
        except ValueError:
            raise RuntimeError("Cannot safely identify a possible live trainer")
        if any(args[i:i+2] == ["-m","models.train"] for i in range(len(args)-1)):
            commands.append(args)
    return commands


def argument(args, key):
    values = [args[i+1] for i,item in enumerate(args[:-1]) if item == key]
    values += [item.split("=",1)[1] for item in args if item.startswith(key+"=")]
    return values[-1] if values else None


def drain(pid, planned, fingerprint):
    identity = None
    if pid <= 1 or pid == os.getpid():
        raise ValueError("Invalid CPU drain PID")
    while True:
        observed = subprocess.run(["ps","-p",str(pid),"-o","stat=","-o","args="],capture_output=True,text=True)
        if observed.returncode or not observed.stdout.strip():
            return
        state,command = observed.stdout.strip().split(None,1)
        if state.startswith("Z"):
            return
        args = shlex.split(command)
        output = argument(args,"--output")
        job = next((j for j in planned if output is not None and
                    Path(output).resolve() == (Path("models/runs")/j["name"]).resolve()),None)
        if (job is None or argument(args,"--device") != "cpu" or
                not any(args[i:i+2] == ["-m","models.train"] for i in range(len(args)-1)) or
                (identity is not None and command != identity)):
            raise RuntimeError("CPU drain PID is not the same planned trainer")
        config = json.loads((Path(output)/"config.json").read_text())
        if not matching_config(config,job,fingerprint):
            raise ValueError("Live CPU run differs from lane budget")
        if identity is None:
            print(f"Waiting for verified CPU trainer pid={pid}: {job['name']}",flush=True)
            identity = command
        time.sleep(5)


def command_for(job, directory, fingerprint):
    if any(argument(args,"--output") is not None and
           Path(argument(args,"--output")).resolve() == directory.resolve()
           for args in live_training_commands()):
        raise RuntimeError(f"Trainer already live for {directory}")
    verify_initialization(job,fingerprint)
    command = [sys.executable,"-u","-m","models.train","--data","datasets/default","--output",str(directory)]
    for key in ("size","representation","seed","epochs","batch_size","accumulation","lr",
                "device","threads","purpose","encoding","train_limit","eval_limit","initialization"):
        if job[key] is not None:
            command.extend(["--"+key.replace("_","-"),str(job[key])])
    if job["validation_only"]:
        command.append("--validation-only")
    if directory.exists():
        config = json.loads((directory/"config.json").read_text())
        if not matching_config(config,job,fingerprint) or not (directory/"last.pt").is_file():
            raise ValueError(f"Partial auxiliary run cannot safely resume: {directory}")
        command.append("--resume")
    return command


def run(lane, drain_pid=None):
    protocol = verify_protocol(lane)
    fingerprint = file_checksum(Path("datasets/default/manifest.json"))
    planned = planned_jobs(lane)
    state_path = Path("models")/f"{lane}_queue_state.json"
    with Path(f".auxiliary_{lane}.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor holds this CPU lane")
        state = dict(lane=lane, supervisor_pid=os.getpid(), status="draining" if drain_pid else "running",
                     drain_pid=drain_pid, conditions=planned, completed=[],
                     protocol_sha256=file_checksum(protocol), dataset_manifest_sha256=fingerprint,
                     interpretation="Auxiliary coverage only, not the primary matrix or final prediction audit")
        write_json(state_path,state)
        if drain_pid:
            drain(drain_pid,planned,fingerprint)
        for job in planned:
            verify_protocol(lane)
            directory = Path("models/runs")/job["name"]
            if not completed(directory,job,fingerprint):
                command = command_for(job,directory,fingerprint)
                Path("models/suite_logs").mkdir(parents=True,exist_ok=True)
                with (Path("models/suite_logs")/(job["name"]+"_aux.log")).open("a") as log:
                    child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    state.update(status="running", active_job=dict(name=job["name"],pid=child.pid,command=command))
                    write_json(state_path,state)
                    print(f"Started {job['name']}: pid={child.pid}",flush=True)
                    code = child.wait()
                if code:
                    state.update(status="failed",exit_code=code)
                    write_json(state_path,state)
                    raise RuntimeError(f"Auxiliary child failed with exit {code}; inspect its log")
                if not completed(directory,job,fingerprint):
                    raise RuntimeError("Auxiliary child lacks required completion coverage")
            state["completed"].append(job["name"])
            state.update(active_job=None)
            write_json(state_path,state)
            subprocess.run([sys.executable,"-m","models.plots"],check=True)
        state.update(status="completed_auxiliary_coverage")
        write_json(state_path,state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane",choices=LANES,required=True)
    parser.add_argument("--drain-pid",type=int)
    parser.add_argument("--allow-legacy-large-sweep", action="store_true")
    args = parser.parse_args()
    if not args.allow_legacy_large_sweep:
        parser.error("Auxiliary queues deferred by compact redesign; legacy replay requires --allow-legacy-large-sweep")
    run(args.lane,args.drain_pid)
