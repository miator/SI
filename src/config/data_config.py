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
TRAIN_NOISE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_noise"  # esc50_snr20
TRAIN_WHITE_FEAT_ROOT = PRECOMPUTED_ROOT / "train_white"
VAL_FEAT_ROOT = PRECOMPUTED_ROOT / "val"
VAL_NOISY_FEAT_ROOT = PRECOMPUTED_ROOT / "val_noise"  # esc50_snr20
VAL_WHITE_FEAT_ROOT = PRECOMPUTED_ROOT / "val_white_snr20"
TEST_FEAT_ROOT = PRECOMPUTED_ROOT / "test"
TEST_NOISY_FEAT_ROOT = PRECOMPUTED_ROOT / "test_noise"  # esc50_snr20
TEST_WHITE_FEAT_ROOT = PRECOMPUTED_ROOT / "test_white_snr20"

USE_PRECOMPUTED_FEATURES = True
TRAIN_FEATURE_MODE = "clean"
TRAIN_FEATURE_PROBABILITIES: Optional[dict[str, float]] = None


def get_eval_split_definitions() -> dict[str, dict[str, Union[Path, float, bool, None, str]]]:
    return {
        "val": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "val_noise": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_NOISY_FEAT_ROOT,
            "augment_kind": "noise",
            "noise_root": ESC50_VAL_NOISE_ROOT,
            "snr": 20.0,
            "is_noisy": True,
        },
        "val_white": {
            "wav_root": Path(VAL_ROOT),
            "feat_root": VAL_WHITE_FEAT_ROOT,
            "augment_kind": "white",
            "noise_root": None,
            "snr": 20.0,
            "is_noisy": True,
        },
        "test": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "test_noise": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_NOISY_FEAT_ROOT,
            "augment_kind": "noise",
            "noise_root": ESC50_TEST_NOISE_ROOT,
            "snr": 20.0,
            "is_noisy": True,
        },
        "test_white": {
            "wav_root": Path(TEST_ROOT),
            "feat_root": TEST_WHITE_FEAT_ROOT,
            "augment_kind": "white",
            "noise_root": None,
            "snr": 20.0,
            "is_noisy": True,
        },
    }


def get_train_feature_root_keys(train_feature_mode: Optional[str] = None) -> tuple[str, ...]:
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    if mode == "clean":
        return "clean",
    if mode == "noise":
        return "noise",
    if mode == "white":
        return "white",
    if mode in {"clean+noise"}:
        return "clean", "noise"
    if mode == "clean+white":
        return "clean", "white"
    if mode == "noise+white":
        return "noise", "white"
    if mode in {"clean+noise+white", "all"}:
        return "clean", "noise", "white"
    raise ValueError(f"Unsupported TRAIN_FEATURE_MODE: {mode}")


def get_train_feat_root(key: str) -> Path:
    if key == "clean":
        return TRAIN_CLEAN_FEAT_ROOT
    if key == "noise":
        return TRAIN_NOISE_FEAT_ROOT
    if key == "white":
        return TRAIN_WHITE_FEAT_ROOT
    raise ValueError(f"Unsupported train feature root key: {key}")


def get_train_feat_roots(train_feature_mode: Optional[str] = None):
    return [
        get_train_feat_root(key)
        for key in get_train_feature_root_keys(train_feature_mode)]


def get_train_feature_probabilities(
    train_feature_mode: Optional[str] = None,
    train_feature_probabilities: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    mode_keys = get_train_feature_root_keys(train_feature_mode)
    configured = (
        TRAIN_FEATURE_PROBABILITIES
        if train_feature_probabilities is None
        else train_feature_probabilities
    )

    if configured is None:
        uniform_probability = 1.0 / len(mode_keys)
        return {key: uniform_probability for key in mode_keys}

    unknown = sorted(set(configured) - set(mode_keys))
    if unknown:
        raise ValueError(
            f"Unsupported train feature probability keys for mode {train_feature_mode!r}: {unknown}"
        )

    return {
        key: float(configured.get(key, 0.0))
        for key in mode_keys
    }
