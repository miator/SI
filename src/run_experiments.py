import subprocess
import sys
import time

EXPERIMENTS = [
    ["--margin", "0.22", "--p", "12", "--k", "5", "--emb-dim", "192", "--wd", "5e-5", "--lr-scheduler", "none"],
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
