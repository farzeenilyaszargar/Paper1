"""Sequential, restartable baseline matrix on one local accelerator.

Resume only after verifying the previous supervisor/child are no longer live.
An OS-held lock prevents two supervisors from running concurrently.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from synthetic_dataset_generator.generator import file_checksum, write_json
from .representations import REPRESENTATIONS
from .transformer import PRIMARY_SIZES, expected_parameter_count
from synthetic_dataset_generator.generator import SPLITS


def jobs(epochs):
    # Same effective batch size 32; smaller microbatches constrain attention RAM.
    micro = {"100k": 16, "500k": 8, "1m": 8, "5m": 2, "10m": 2}
    conditions = [(s,"symbolic",1) for s in PRIMARY_SIZES]
    conditions += [(s,r,seed) for seed in (1,2,3) for r in REPRESENTATIONS for s in PRIMARY_SIZES
                   if (s,r,seed) not in conditions]
    return [{"name": f"baseline_full_{s}_{r}_seed{seed}", "size": s, "representation": r,
             "seed": seed, "epochs": epochs, "batch_size": micro[s], "accumulation": 32//micro[s],
             "learning_rate": 0.001, "parameter_count": expected_parameter_count(s)} for s,r,seed in conditions]


def completed(path, job, fingerprint):
    result_path = path / "results.json"
    if not result_path.exists():
        return False
    result = json.loads(result_path.read_text())
    config = result["config"]
    if result.get("status") != "completed" or result.get("scope") != "full corpus":
        return False
    for key in ("size","representation","seed","epochs","batch_size","accumulation","parameter_count"):
        if config.get(key) != job[key]:
            return False
    if config["dataset_manifest_sha256"] != fingerprint or config["train_examples"] != 100000 or config["validation_examples"] != 10000:
        return False
    if config.get("lr") != job["learning_rate"] or config.get("encoding","characters") != "characters":
        return False
    required = set(SPLITS)-{"train","validation"}
    return all(set(result["evaluations"].get(r,{})) == required and
               all(m["count"] == 10000 for m in result["evaluations"][r].values()) for r in REPRESENTATIONS)


def run(data, epochs, include_pilot):
    planned = jobs(epochs)
    manifest = json.loads((data/"manifest.json").read_text())
    if manifest["config"]["train_count"] != 100000 or manifest["config"]["evaluation_count"] != 10000:
        raise ValueError("Primary matrix requires the full 100K/10K corpus")
    fingerprint = file_checksum(data/"manifest.json")
    with Path(".suite.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another suite supervisor holds the accelerator lock")
        processes = subprocess.check_output(["ps","-axo","pid,args"],text=True)
        if any(" -m models.train " in line for line in processes.splitlines()):
            raise RuntimeError("An existing trainer is live; inspect it before starting another accelerator run")
        Path("models/suite_logs").mkdir(parents=True,exist_ok=True)
        state_path = Path("models/suite_state.json")
        state = {"status":"running", "supervisor_pid":os.getpid(), "dataset_manifest_sha256":fingerprint,
                 "phase":"fixed-budget baseline; validation calibration/controls remain separate",
                 "conditions":planned, "completed":[], "failed":[]}

        def dispatch(name, command):
            with (Path("models/suite_logs")/(name+".log")).open("a") as log:
                process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                state["active_job"] = {"name":name,"pid":process.pid,"command":command}
                write_json(state_path,state)
                print(f"Started {name}: pid={process.pid}",flush=True)
                status = process.wait()
            state["active_job"] = None
            if status:
                state["failed"].append({"name":name,"exit_code":status})
                state["status"] = "failed"
                write_json(state_path,state)
                raise RuntimeError(f"Training failed: {name}, exit {status}; inspect its log")

        if include_pilot:
            pilot = Path("models/runs/pilot_500k_symbolic_seed1")
            if (pilot/"last.pt").exists() and not (pilot/"results.json").exists():
                dispatch("resume_pilot_500k",[sys.executable,"-u","-m","models.train","--output",str(pilot),
                    "--size","500k","--epochs","30","--batch-size","32","--train-limit","20000",
                    "--eval-limit","1000","--device","mps","--resume"])
        for job in planned:
            output = Path("models/runs")/job["name"]
            if not completed(output,job,fingerprint):
                command = [sys.executable,"-u","-m","models.train","--data",str(data),"--output",str(output),
                    "--size",job["size"],"--representation",job["representation"],"--seed",str(job["seed"]),
                    "--epochs",str(job["epochs"]),"--batch-size",str(job["batch_size"]),
                    "--accumulation",str(job["accumulation"]),"--lr",str(job["learning_rate"]),"--device","mps"]
                if output.exists():
                    if not (output/"last.pt").exists():
                        raise RuntimeError(f"Partial run without a valid epoch checkpoint: {output}")
                    command.append("--resume")
                dispatch(job["name"],command)
                if not completed(output,job,fingerprint):
                    raise RuntimeError(f"Run did not satisfy full-matrix coverage: {output}")
            state["completed"].append(job["name"])
            write_json(state_path,state)
            subprocess.run([sys.executable,"-m","models.plots"],check=True)
        state["status"] = "baseline_matrix_completed"
        write_json(state_path,state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("datasets/default"))
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--include-pilot",action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.data,args.epochs,args.include_pilot)
