import subprocess
import sys
import time

EXPERIMENTS = [
    ["--margin", "0.25"],
    ["--margin", "0.30"],
    ["--margin", "0.35"],
    ["--margin", "0.40"],

    ["--p", "16", "--k", "4"],
    ["--p", "10", "--k", "6"],
    ["--p", "8", "--k", "8"],

    ["--emb-dim", "128"],
    ["--emb-dim", "256"],

    ["--margin", "0.30", "--p", "16", "--k", "4"],
    ["--margin", "0.30", "--p", "16", "--k", "4", "--emb-dim", "256"],
]

for i, params in enumerate(EXPERIMENTS, start=1):
    print("=" * 80)
    print(f"Starting experiment {i}/{len(EXPERIMENTS)}: {' '.join(params)}")
    print("=" * 80)

    cmd = [sys.executable, "train.py"] + params
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Experiment failed. Stopping.")
        break

    print("Cooling CPU for 120 seconds...")
    time.sleep(120)
