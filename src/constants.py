from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent

EXP_NAME = "cnn1d_baseline_1"

RUNS_DIR = PROJECT_ROOT / "runs"
EXP_DIR = RUNS_DIR / EXP_NAME

TB_DIR = EXP_DIR / "tensorboard"
CKPT_DIR = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"

BEST_MODEL_PATH = CKPT_DIR / "best_val_loss.pt"
LAST_MODEL_PATH = CKPT_DIR / "last_epoch.pt"

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
N_MFCC = 30
FMIN = 20.0
FMAX = 8000.0

MAX_FRAMES = 301
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

LR_SCHEDULER = "none"
LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LR = 1e-6
COSINE_T_MAX = EPOCHS
COSINE_ETA_MIN = MIN_LR

EMB_DIM = 192
DROPOUT = 0.3
EPS = 1e-6

TRIPLET_MARGIN = 0.22
P = 12
K = 5

TRAIN_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\train"
VAL_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\val"
TEST_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\test"

PRECOMPUTED_ROOT = Path(
    r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\logmel_cache"
)

MUSAN_NOISE_ROOT = Path(
    r"C:\Users\User\Desktop\Data\musan\musan\noise"
)

NOISE_SPLIT_SEED = 37
NOISE_TRAIN_FILES_FRACTION = 0.8

TRAIN_CLEAN_FEAT_ROOT = PRECOMPUTED_ROOT / "train"
TRAIN_NOISE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_noise"
VAL_FEAT_ROOT = PRECOMPUTED_ROOT / "val"
TEST_FEAT_ROOT = PRECOMPUTED_ROOT / "test"
VAL_NOISY_SNR15_FEAT_ROOT = PRECOMPUTED_ROOT / "val_noisy_snr15"
TEST_NOISY_SNR15_FEAT_ROOT = PRECOMPUTED_ROOT / "test_noisy_snr15"
TEST_NOISY_SNR10_FEAT_ROOT = PRECOMPUTED_ROOT / "test_noisy_snr10"

TRAIN_FEAT_ROOT = TRAIN_CLEAN_FEAT_ROOT

USE_PRECOMPUTED_FEATURES = True
TRAIN_FEATURE_MODE = "clean+noise"

USE_NOISE_AUG = False
NOISE_PROB = 0.5
SNR_MIN = 10.0
SNR_MAX = 20.0
MIN_NOISE_SECONDS = 3.0


def get_eval_split_definitions():
    return {
        "val": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_FEAT_ROOT,
            "snr": None,
            "is_noisy": False,
        },
        "val_noisy_snr15": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_NOISY_SNR15_FEAT_ROOT,
            "snr": 15.0,
            "is_noisy": True,
        },
        "test": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_FEAT_ROOT,
            "snr": None,
            "is_noisy": False,
        },
        "test_noisy_snr15": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_NOISY_SNR15_FEAT_ROOT,
            "snr": 15.0,
            "is_noisy": True,
        },
        "test_noisy_snr10": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_NOISY_SNR10_FEAT_ROOT,
            "snr": 10.0,
            "is_noisy": True,
        },
    }


def get_train_feat_roots(train_feature_mode: Optional[str] = None):
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    if mode == "clean":
        return [TRAIN_CLEAN_FEAT_ROOT]
    if mode == "noise":
        return [TRAIN_NOISE_FEAT_ROOT]
    if mode == "clean+noise":
        return [TRAIN_CLEAN_FEAT_ROOT, TRAIN_NOISE_FEAT_ROOT]
    raise ValueError(f"Unsupported TRAIN_FEATURE_MODE: {mode}")


def get_augmentation_metadata(
    train_feature_mode: Optional[str] = None,
    use_noise_aug: Optional[bool] = None,
):
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    use_aug = USE_NOISE_AUG if use_noise_aug is None else use_noise_aug
    has_noise = mode != "clean" or use_aug

    return {
        "augmentation": "musan_noise" if has_noise else "clean",
        "train_feature_mode": mode,
        "use_noise_aug": bool(use_aug),
        "noise_prob": NOISE_PROB if has_noise else 0.0,
        "snr_min": SNR_MIN if has_noise else None,
        "snr_max": SNR_MAX if has_noise else None,
        "min_noise_seconds": MIN_NOISE_SECONDS if has_noise else None,
        "musan_noise_root": str(MUSAN_NOISE_ROOT) if has_noise else None,
    }
