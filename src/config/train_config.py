import os

EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-6
COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = MIN_LR

MARGIN = 0.22
P = 12
K = 5

LIGHTWEIGHT_VERIFY_EVERY_N_EPOCHS = 0
LIGHTWEIGHT_VERIFY_SPLIT = "dev_clean"
LIGHTWEIGHT_VERIFY_SAME_PAIRS = 2000
LIGHTWEIGHT_VERIFY_DIFF_PAIRS = 2000


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


TRAIN_NUM_WORKERS = _env_int("SI_TRAIN_NUM_WORKERS", min(8, max(1, (os.cpu_count() or 8) // 2)))
VAL_NUM_WORKERS = _env_int("SI_VAL_NUM_WORKERS", min(4, max(1, (os.cpu_count() or 8) // 4)))
VERIFY_NUM_WORKERS = _env_int("SI_VERIFY_NUM_WORKERS", min(4, max(1, (os.cpu_count() or 8) // 4)))
LIGHTWEIGHT_VERIFY_NUM_WORKERS = _env_int("SI_LIGHTWEIGHT_VERIFY_NUM_WORKERS", VERIFY_NUM_WORKERS)
PREFETCH_FACTOR = _env_int("SI_PREFETCH_FACTOR", 4)
VERIFY_FEATURE_FILES_ON_INIT = os.environ.get("SI_VERIFY_FEATURE_FILES_ON_INIT", "0") == "1"
