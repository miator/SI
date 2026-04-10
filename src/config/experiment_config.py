from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXP_NAME = ""

RUNS_DIR = PROJECT_ROOT / "runs_conformer"
EXP_DIR = RUNS_DIR / EXP_NAME

TB_DIR = EXP_DIR / "tensorboard"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"

BEST_MODEL_PATH = CKPT_DIR / "best.pt"
LAST_MODEL_PATH = CKPT_DIR / "last.pt"
