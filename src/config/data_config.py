import os
from pathlib import Path
from typing import Optional, Union


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


_DATA_ROOT = Path(os.environ.get("SI_DATA_ROOT", "/workspace/data")).expanduser()
PRECOMPUTED_DATA_ROOT = _env_path(
    "SI_PRECOMPUTED_DATA_ROOT",
    str(_DATA_ROOT / "logmel_cache"),
)
WAV_DATA_ROOT = _env_path(
    "SI_WAV_DATA_ROOT",
    str(_DATA_ROOT / "wav"),
)

TRAIN_CLEAN100_WAV_ROOT = _env_path(
    "SI_TRAIN_CLEAN100_WAV_ROOT",
    str(WAV_DATA_ROOT / "train_clean100"),
)
TRAIN_CLEAN360_WAV_ROOT = _env_path(
    "SI_TRAIN_CLEAN360_WAV_ROOT",
    str(WAV_DATA_ROOT / "train-clean-360"),
)
TRAIN_OTHER500_WAV_ROOT = _env_path(
    "SI_TRAIN_OTHER500_WAV_ROOT",
    str(WAV_DATA_ROOT / "train-other-500"),
)
TRAIN_WAV_ROOT = TRAIN_CLEAN100_WAV_ROOT
EVAL_WAV_ROOT = WAV_DATA_ROOT
DEV_CLEAN_WAV_ROOT = _env_path("SI_DEV_CLEAN_WAV_ROOT", str(WAV_DATA_ROOT / "dev-clean"))
DEV_OTHER_WAV_ROOT = _env_path("SI_DEV_OTHER_WAV_ROOT", str(WAV_DATA_ROOT / "dev-other"))
TEST_CLEAN_WAV_ROOT = _env_path("SI_TEST_CLEAN_WAV_ROOT", str(WAV_DATA_ROOT / "test-clean"))
TEST_OTHER_WAV_ROOT = _env_path("SI_TEST_OTHER_WAV_ROOT", str(WAV_DATA_ROOT / "test-other"))

TRAIN_CLEAN100_PRECOMPUTED_ROOT = PRECOMPUTED_DATA_ROOT
TRAIN_360_500_PRECOMPUTED_ROOT = PRECOMPUTED_DATA_ROOT
TRAIN_PRECOMPUTED_ROOT = PRECOMPUTED_DATA_ROOT
TRAIN_CLEAN100_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "train_clean100"
TRAIN_CLEAN360_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "train-clean-360"
TRAIN_OTHER500_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "train-other-500"
TRAIN_CLEAN_FEAT_ROOT = TRAIN_CLEAN100_FEAT_ROOT
TRAIN_NOISE_FEAT_ROOTS_BY_SET = {
    "clean100": PRECOMPUTED_DATA_ROOT / "train_noise_snr20",
    "clean360": PRECOMPUTED_DATA_ROOT / "train_noise_snr20",
    "other500": PRECOMPUTED_DATA_ROOT / "train_noise_snr20",
}
TRAIN_WHITE_FEAT_ROOTS_BY_SET = {
    "clean100": PRECOMPUTED_DATA_ROOT / "train_white_snr20-25",
    "clean360": PRECOMPUTED_DATA_ROOT / "train_white_snr20-25",
    "other500": PRECOMPUTED_DATA_ROOT / "train_white_snr20-25",
}
TRAIN_NOISE_FEAT_ROOTS = (TRAIN_NOISE_FEAT_ROOTS_BY_SET["clean100"],)
TRAIN_WHITE_FEAT_ROOTS = (TRAIN_WHITE_FEAT_ROOTS_BY_SET["clean100"],)

EVAL_PRECOMPUTED_ROOT = PRECOMPUTED_DATA_ROOT
DEV_CLEAN_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "dev-clean"
DEV_OTHER_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "dev-other"
TEST_CLEAN_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "test-clean"
TEST_OTHER_FEAT_ROOT = PRECOMPUTED_DATA_ROOT / "test-other"

ESC50_NOISE_ROOT = _env_path(
    "SI_ESC50_NOISE_ROOT",
    str(Path(_DATA_ROOT) / "ESC-50-master-noise" / "audio_standardized_16k")
    if _DATA_ROOT
    else r"C:\Users\User\Desktop\Data\ESC-50-master-noise\audio_standardized_16k",
)
ESC50_TRAIN_NOISE_ROOT = ESC50_NOISE_ROOT / "train-noise"
ESC50_VAL_NOISE_ROOT = ESC50_NOISE_ROOT / "val-noise"
ESC50_TEST_NOISE_ROOT = ESC50_NOISE_ROOT / "test-noise"

TRAIN_SPLIT_NAME = "train_clean100_clean360_other500"
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
TRAIN_SET_NAMES: tuple[str, ...] = ("clean100",)
TRAIN_DATA_MODE = "clean_only"
TRAIN_DATA_PROBABILITIES: Optional[dict[str, float]] = None
USE_OTHER_AS_AUGMENTATION = False


def is_probabilistic_train_feature_mode(train_feature_mode: Optional[str] = None) -> bool:
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    return "|" in mode


def get_all_train_split_definitions() -> dict[str, dict[str, Path | str]]:
    return {
        "clean100": {
            "set_name": "clean100",
            "name": Path(TRAIN_CLEAN100_FEAT_ROOT).name,
            "wav_root": TRAIN_CLEAN100_WAV_ROOT,
            "precomputed_root": TRAIN_CLEAN100_PRECOMPUTED_ROOT,
            "clean_feat_root": TRAIN_CLEAN100_FEAT_ROOT,
        },
        "clean360": {
            "set_name": "clean360",
            "name": Path(TRAIN_CLEAN360_FEAT_ROOT).name,
            "wav_root": TRAIN_CLEAN360_WAV_ROOT,
            "precomputed_root": TRAIN_360_500_PRECOMPUTED_ROOT,
            "clean_feat_root": TRAIN_CLEAN360_FEAT_ROOT,
        },
        "other500": {
            "set_name": "other500",
            "name": Path(TRAIN_OTHER500_FEAT_ROOT).name,
            "wav_root": TRAIN_OTHER500_WAV_ROOT,
            "precomputed_root": TRAIN_360_500_PRECOMPUTED_ROOT,
            "clean_feat_root": TRAIN_OTHER500_FEAT_ROOT,
        },
    }


def get_train_split_definitions(
    train_sets: Optional[tuple[str, ...]] = None,
) -> tuple[dict[str, Path | str], ...]:
    selected_sets = TRAIN_SET_NAMES if train_sets is None else train_sets
    definitions = get_all_train_split_definitions()
    unknown = sorted(set(selected_sets) - set(definitions))
    if unknown:
        raise ValueError(f"Unsupported train set names: {unknown}")
    return tuple(definitions[name] for name in selected_sets)


def resolve_train_clean_feature_path(wav_path: Path) -> Path:
    return resolve_train_feature_path(wav_path, "clean")


def get_train_split_feature_root(
    split_def: dict[str, Path | str],
    key: str,
) -> Path:
    set_name = str(split_def["set_name"])

    if key == "clean":
        return Path(split_def["clean_feat_root"])
    if key == "noise":
        return Path(TRAIN_NOISE_FEAT_ROOTS_BY_SET[set_name])
    if key == "white":
        return Path(TRAIN_WHITE_FEAT_ROOTS_BY_SET[set_name])
    raise ValueError(f"Unsupported train feature root key: {key}")


def resolve_train_feature_path(wav_path: Path, key: str) -> Path:
    wav_path = Path(wav_path)
    for split_def in get_train_split_definitions():
        wav_root = Path(split_def["wav_root"])
        try:
            rel = wav_path.relative_to(wav_root)
        except ValueError:
            continue
        return get_train_split_feature_root(split_def, key) / rel.with_suffix(".pt")
    raise ValueError(
        f"Wav path is not under any configured train split root for key {key!r}: {wav_path}"
    )


def resolve_train_feature_path_from_source_path(
    source_path: Path,
    source_key: str,
    target_key: str,
) -> Path:
    source_path = Path(source_path)
    for split_def in get_train_split_definitions():
        source_root = get_train_split_feature_root(split_def, source_key)
        try:
            rel = source_path.relative_to(source_root)
        except ValueError:
            continue
        return get_train_split_feature_root(split_def, target_key) / rel
    raise ValueError(
        "Source path is not under any configured train feature root "
        f"for key {source_key!r}: {source_path}"
    )


def resolve_train_feature_path_from_relative_path(
    *,
    split_root: Path,
    relative_path: Path,
    source_key: str,
    target_key: str,
) -> Path:
    split_root = Path(split_root)
    relative_path = Path(relative_path)

    for split_def in get_train_split_definitions():
        source_root = get_train_split_feature_root(split_def, source_key)
        if split_root == source_root:
            return get_train_split_feature_root(split_def, target_key) / relative_path

    raise ValueError(
        "Split root is not under any configured train feature root "
        f"for key {source_key!r}: {split_root}"
    )


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
    return tuple(
        get_train_split_feature_root(split_def, key)
        for split_def in get_train_split_definitions()
    )


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
