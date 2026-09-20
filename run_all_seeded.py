"""Finish the 24 runs, check them, build the six tables and publish.

If the full seed-42 process is already running, pass --wait-for-full42 to wait for it.
It stops at the first failed run and never publishes an incomplete table."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "seeded_runs"
OUTPUT = ROOT / "六表_新重跑_42_43_44"
ARCHIVE = Path(r"C:\Users\WKU-MATH-558\Desktop\gold1")
GIT = shutil.which("git") or str(Path(
    r"C:\Users\WKU-MATH-558\AppData\Local\CodexTools\gold1-github\git\cmd\git.exe"))
MODELS = ("random_walk", "random_forest", "xgboost", "lightgbm",
          "full", "no_fcnn", "without_fe", "cnn_lstm")


def status(folder):
    path = folder / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def announce(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def run_one(model, seed):
    folder = RUNS / f"seed{seed}" / model
    if status(folder) == "completed":
        announce(f"SKIP completed {model} seed {seed}")
        return
    if model == "cnn_lstm":
        if folder.exists():
            raise RuntimeError(f"CNN-LSTM has no fold resume; inspect existing folder: {folder}")
        cmd = [sys.executable, "CNN_LSTM.py", "--task", "both"]
    elif model in ("full", "no_fcnn", "without_fe"):
        cmd = [sys.executable, "run_seeded_ensembles.py", "--model", model]
    else:
        cmd = [sys.executable, "run_seeded_baselines.py", "--model", model]
    cmd += ["--seed", str(seed), "--output", str(folder)]
    log = RUNS / f"{model}_seed{seed}_console.log"
    announce(f"START {model} seed {seed}")
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)
    if result.returncode != 0 or status(folder) != "completed":
        raise RuntimeError(f"Failed {model} seed {seed}; see {log}")
    announce(f"DONE {model} seed {seed}")


def publish():
    if not OUTPUT.exists() or len(list(OUTPUT.glob("Table*.csv"))) != 4 or len(list(OUTPUT.glob("DM*.csv"))) != 2:
        raise RuntimeError("Six-table output is incomplete")
    subprocess.run([GIT, "add", str(OUTPUT.relative_to(ROOT))], cwd=ROOT, check=True)
    subprocess.run([GIT, "commit", "-m", "Publish six aligned seeded tables and DM audit"],
                   cwd=ROOT, check=True)
    subprocess.run([GIT, "push", "origin", "main"], cwd=ROOT, check=True)
    announce("PUBLISHED six tables to GitHub")
    if ARCHIVE.is_dir() and ARCHIVE.resolve() != ROOT.resolve():
        destination = ARCHIVE / OUTPUT.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite archival results: {destination}")
        shutil.copytree(OUTPUT, destination)
        announce(f"COPIED six tables to {destination}")


def main(wait_for_full42=False):
    RUNS.mkdir(exist_ok=True)
    if wait_for_full42:
        folder = RUNS / "seed42" / "full"
        announce("Waiting for the already running full seed-42 job")
        deadline = time.monotonic() + 24 * 3600
        while status(folder) == "running":
            if time.monotonic() > deadline:
                raise TimeoutError("Full seed-42 job did not finish within 24 hours")
            time.sleep(60)
        if status(folder) != "completed":
            raise RuntimeError(f"Existing full seed-42 job failed: {folder}")
    for seed in (42, 43, 44):
        for model in MODELS:
            run_one(model, seed)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    announce("Building six tables and paired DM audits")
    subprocess.run([sys.executable, "build_six_tables.py", "--runs", str(RUNS),
                    "--output", str(OUTPUT)], cwd=ROOT, check=True)
    publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-full42", action="store_true")
    args = parser.parse_args()
    main(wait_for_full42=args.wait_for_full42)
