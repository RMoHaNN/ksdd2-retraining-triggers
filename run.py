"""Load the recorded paper results and open their MLflow dashboard."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

from src.data import CONFIG, ROOT
from src.model import load_model
from src.experiments import load_records

RESULTS = ROOT / "results"
BUNDLE_URL = "https://github.com/RMoHaNN/ksdd2-retraining-triggers/releases/download/paper-run/paper_run.zip"
PAYLOADS = {"training.json", "conditions.json", "replays.json", "v1.pt"}


def valid_bundle(folder):
    try:
        checks = dict(line.split("  ", 1)[::-1] for line in
                      (folder / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines())
        return set(checks) == PAYLOADS and all(
            hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            for name, digest in checks.items()
        )
    except (OSError, ValueError):
        return False


def get_bundle(local_zip=None):
    RESULTS.mkdir(exist_ok=True)
    folder = RESULTS / "paper_run"
    if local_zip is None and valid_bundle(folder):
        return folder
    target = RESULTS / "paper_run.zip"
    if local_zip is not None:
        if Path(local_zip).resolve() != target.resolve():
            shutil.copyfile(local_zip, target)
    else:
        urllib.request.urlretrieve(BUNDLE_URL, target)
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as bundle:
        if sorted(bundle.namelist()) != sorted(PAYLOADS | {"SHA256SUMS.txt"}):
            raise ValueError("The bundle must contain the five expected paper files.")
        bundle.extractall(folder)
    if not valid_bundle(folder):
        raise ValueError("The paper bundle failed its SHA256SUMS check.")
    return folder


def print_tables(training, conditions, replays):
    print("\nTraining means (%)\nRegime      AP   Recall")
    for regime in ("plain", "managed"):
        rows = [row for row in training if row["regime"] == regime]
        ap, recall = (100 * sum(row[key] for row in rows) / len(rows)
                      for key in ("ap", "recall"))
        print(f"{regime:<10} {ap:4.1f}   {recall:4.1f}")
    print("\nTable 1\nCondition            Flags   AP (%)   Recall (%)")
    for row in conditions:
        print(f"{row['condition']:<20} {row['flagged_mean']:4.1f}     "
              f"{100 * row['ap']:4.1f}       {100 * row['recall']:4.1f}")
    print("\nTable 2\nSetting      Missed  False alarms  Retrains  Rejected  Promoted  GPU min")
    for name, replay in replays.items():
        row = replay["summary"]
        print(f"{name:<12} {row['missed_defects']:>4}   {row['false_alarms']:>5}"
              f"         {row['retrains']:>3}       {row['rejected']:>3}"
              f"       {row['promoted']:>3}       {row['gpu_minutes']:>4.1f}")


def dashboard():
    url = "http://127.0.0.1:5000"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 5000))
    command = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
               "sqlite:///" + (RESULTS / "mlflow.db").as_posix(),
               "--default-artifact-root", (RESULTS / "mlartifacts").as_uri(),
               "--host", "127.0.0.1", "--port", "5000", "--workers", "1"]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with (RESULTS / "dashboard.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("MLflow stopped; see results/dashboard.log.")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("MLflow did not start; see results/dashboard.log.")
            webbrowser.open(url)
            print(f"\nMLflow: {url}. Press Ctrl+C to stop.", flush=True)
            while server.poll() is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping MLflow.")
        finally:
            if server.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                else:
                    os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Use a local paper_run.zip")
    parser.add_argument("--data", type=Path, default=ROOT / CONFIG["data"]["path"],
                        help="KolektorSDD2 folder for the defect-crop figure")
    parser.add_argument("--no-ui", action="store_true", help="Load the results without opening MLflow")
    parser.add_argument("--force", action="store_true", help="Rebuild the local MLflow store")
    args = parser.parse_args()
    folder = get_bundle(args.bundle)
    if args.force:
        for path in RESULTS.glob("mlflow.db*"):
            path.unlink()
        shutil.rmtree(RESULTS / "mlartifacts", ignore_errors=True)
    training, conditions, replays = (json.loads((folder / name).read_text(encoding="utf-8"))
                                    for name in ("training.json", "conditions.json", "replays.json"))
    model, threshold = load_model(folder / "v1.pt")
    load_records(RESULTS, training, conditions, replays, model, threshold)
    print_tables(training, conditions, replays)
    if not args.no_ui:
        dashboard()


if __name__ == "__main__":
    main()
