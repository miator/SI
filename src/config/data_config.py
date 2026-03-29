from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
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
TRAIN_WHITE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_white_snr10_20"
TRAIN_MUSAN_WHITE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_musan_white_snr10_20"

TRAIN_FEAT_ROOT = TRAIN_CLEAN_FEAT_ROOT
USE_PRECOMPUTED_FEATURES = True
TRAIN_FEATURE_MODE = "clean+noise"


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
    if mode == "clean+white":
        return [TRAIN_CLEAN_FEAT_ROOT, TRAIN_WHITE_FEAT_ROOT]
    if mode == "clean+musan+white":
        return [TRAIN_CLEAN_FEAT_ROOT, TRAIN_MUSAN_WHITE_FEAT_ROOT]
    if mode == "white":
        return [TRAIN_WHITE_FEAT_ROOT]
    if mode == "musan+white":
        return [TRAIN_MUSAN_WHITE_FEAT_ROOT]
    raise ValueError(f"Unsupported TRAIN_FEATURE_MODE: {mode}")
