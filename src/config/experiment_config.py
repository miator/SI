from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs_cnn1d"

EXP_NAME = ""
RUNS_DIR = DEFAULT_RUNS_DIR
EXP_DIR = RUNS_DIR
TB_DIR = EXP_DIR / "tensorboard"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"
BEST_MODEL_PATH = CKPT_DIR / "best.pt"
LAST_MODEL_PATH = CKPT_DIR / "last.pt"


def configure_experiment(*, exp_name: str, runs_dir: Optional[Path] = None) -> None:
    global EXP_NAME, RUNS_DIR, EXP_DIR, TB_DIR, CKPT_DIR, RESULTS_DIR, BEST_MODEL_PATH, LAST_MODEL_PATH

    EXP_NAME = exp_name
    if runs_dir is not None:
        RUNS_DIR = Path(runs_dir)

    EXP_DIR = RUNS_DIR / EXP_NAME if EXP_NAME else RUNS_DIR
    TB_DIR = EXP_DIR / "tensorboard"
    CKPT_DIR = EXP_DIR / "checkpoints"
    RESULTS_DIR = EXP_DIR / "results"
    BEST_MODEL_PATH = CKPT_DIR / "best.pt"
    LAST_MODEL_PATH = CKPT_DIR / "last.pt"
