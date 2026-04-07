from pathlib import Path
from typing import Optional, Union

TRAIN_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\train"
VAL_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\val"
TEST_ROOT = r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\wav\test"

PRECOMPUTED_ROOT = Path(
    r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech_standardized_chunks_3s\logmel_cache"
)

ESC50_NOISE_ROOT = Path(
    r"C:\Users\User\Desktop\Data\ESC-50-master-noise\audio_standardized_16k"
)
ESC50_TRAIN_NOISE_ROOT = ESC50_NOISE_ROOT / "train-noise"
ESC50_VAL_NOISE_ROOT = ESC50_NOISE_ROOT / "val-noise"
ESC50_TEST_NOISE_ROOT = ESC50_NOISE_ROOT / "test-noise"

TRAIN_CLEAN_FEAT_ROOT = PRECOMPUTED_ROOT / "train"
TRAIN_NOISE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_noise"
VAL_FEAT_ROOT = PRECOMPUTED_ROOT / "val"
VAL_NOISY_FEAT_ROOT = PRECOMPUTED_ROOT / "val_noise"
TEST_FEAT_ROOT = PRECOMPUTED_ROOT / "test"
TEST_NOISY_FEAT_ROOT = PRECOMPUTED_ROOT / "test_noise"

USE_PRECOMPUTED_FEATURES = True
TRAIN_FEATURE_MODE = "clean"


def get_eval_split_definitions() -> dict[str, dict[str, Union[Path, float, bool, None]]]:
    return {
        "val": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_FEAT_ROOT,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "val_noisy": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_NOISY_FEAT_ROOT,
            "noise_root": ESC50_VAL_NOISE_ROOT,
            "snr": 20.0,
            "is_noisy": True,
        },
        "test": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_FEAT_ROOT,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "test_noisy": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_NOISY_FEAT_ROOT,
            "noise_root": ESC50_TEST_NOISE_ROOT,
            "snr": 20.0,
            "is_noisy": True,
        },
    }


def get_train_feat_roots(train_feature_mode: Optional[str] = None):
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    if mode == "clean":
        return [TRAIN_CLEAN_FEAT_ROOT]
    if mode == "noise":
        return [TRAIN_NOISE_FEAT_ROOT]
    if mode == "both":
        return [TRAIN_CLEAN_FEAT_ROOT, TRAIN_NOISE_FEAT_ROOT]
    raise ValueError(f"Unsupported TRAIN_FEATURE_MODE: {mode}")
