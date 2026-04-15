from pathlib import Path
from typing import Optional, Union

TRAIN_DATA_ROOT = Path(
    r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s"
)
EVAL_DATA_ROOT = Path(
    r"C:\Users\User\Desktop\Data\Librispeech_eval_standardized_chunks_3s"
)

TRAIN_WAV_ROOT = TRAIN_DATA_ROOT / "wav" / "train_clean100"
EVAL_WAV_ROOT = EVAL_DATA_ROOT / "wav"
DEV_CLEAN_WAV_ROOT = EVAL_WAV_ROOT / "dev-clean"
DEV_OTHER_WAV_ROOT = EVAL_WAV_ROOT / "dev-other"
TEST_CLEAN_WAV_ROOT = EVAL_WAV_ROOT / "test-clean"
TEST_OTHER_WAV_ROOT = EVAL_WAV_ROOT / "test-other"

TRAIN_PRECOMPUTED_ROOT = TRAIN_DATA_ROOT / "logmel_cache"
TRAIN_CLEAN_FEAT_ROOT = TRAIN_PRECOMPUTED_ROOT / "train_clean100"
TRAIN_NOISE_FEAT_ROOTS = (
    TRAIN_PRECOMPUTED_ROOT / "train_noise_snr20",
)
TRAIN_WHITE_FEAT_ROOTS = (
    TRAIN_PRECOMPUTED_ROOT / "train_white_snr25",
)

EVAL_PRECOMPUTED_ROOT = EVAL_DATA_ROOT / "logmel_cache"
DEV_CLEAN_FEAT_ROOT = EVAL_PRECOMPUTED_ROOT / "dev-clean"
DEV_OTHER_FEAT_ROOT = EVAL_PRECOMPUTED_ROOT / "dev-other"
TEST_CLEAN_FEAT_ROOT = EVAL_PRECOMPUTED_ROOT / "test-clean"
TEST_OTHER_FEAT_ROOT = EVAL_PRECOMPUTED_ROOT / "test-other"

ESC50_NOISE_ROOT = Path(r"C:\Users\User\Desktop\Data\ESC-50-master-noise\audio_standardized_16k")
ESC50_TRAIN_NOISE_ROOT = ESC50_NOISE_ROOT / "train-noise"
ESC50_VAL_NOISE_ROOT = ESC50_NOISE_ROOT / "val-noise"
ESC50_TEST_NOISE_ROOT = ESC50_NOISE_ROOT / "test-noise"

TRAIN_SPLIT_NAME = "train_clean100"
TRAIN_ROOT = TRAIN_WAV_ROOT
VAL_ROOT = DEV_CLEAN_WAV_ROOT
TEST_ROOT = TEST_CLEAN_WAV_ROOT
PRECOMPUTED_ROOT = TRAIN_PRECOMPUTED_ROOT
TRAIN_NOISE_FEAT_ROOT = TRAIN_NOISE_FEAT_ROOTS[0]
TRAIN_WHITE_FEAT_ROOT = TRAIN_WHITE_FEAT_ROOTS[0]

VAL_FEAT_ROOT = DEV_CLEAN_FEAT_ROOT
TEST_FEAT_ROOT = TEST_CLEAN_FEAT_ROOT

USE_PRECOMPUTED_FEATURES = True
TRAIN_FEATURE_MODE = "clean"
TRAIN_FEATURE_PROBABILITIES: Optional[dict[str, float]] = None


def is_probabilistic_train_feature_mode(train_feature_mode: Optional[str] = None) -> bool:
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    return "|" in mode


def get_eval_split_definitions() -> dict[str, dict[str, Union[Path, float, bool, None, str]]]:
    return {
        "dev_clean": {
            "wav_root": DEV_CLEAN_WAV_ROOT,
            "feat_root": DEV_CLEAN_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "dev_other": {
            "wav_root": DEV_OTHER_WAV_ROOT,
            "feat_root": DEV_OTHER_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "test_clean": {
            "wav_root": TEST_CLEAN_WAV_ROOT,
            "feat_root": TEST_CLEAN_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
        "test_other": {
            "wav_root": TEST_OTHER_WAV_ROOT,
            "feat_root": TEST_OTHER_FEAT_ROOT,
            "augment_kind": None,
            "noise_root": None,
            "snr": None,
            "is_noisy": False,
        },
    }


def get_train_feature_root_keys(train_feature_mode: Optional[str] = None) -> tuple[str, ...]:
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode

    if mode == "clean|noise":
        return "clean", "noise"
    if mode == "clean|white":
        return "clean", "white"
    if mode == "noise|white":
        return "noise", "white"
    if mode in {"clean|noise|white", "random_all"}:
        return "clean", "noise", "white"

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


def get_train_feat_roots_for_key(key: str) -> tuple[Path, ...]:
    if key == "clean":
        return (TRAIN_CLEAN_FEAT_ROOT,)
    if key == "noise":
        return tuple(TRAIN_NOISE_FEAT_ROOTS)
    if key == "white":
        return tuple(TRAIN_WHITE_FEAT_ROOTS)
    raise ValueError(f"Unsupported train feature root key: {key}")


def get_train_feat_roots(train_feature_mode: Optional[str] = None):
    roots: list[Path] = []
    for key in get_train_feature_root_keys(train_feature_mode):
        roots.extend(get_train_feat_roots_for_key(key))
    return roots


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
