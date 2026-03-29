from pathlib import Path
from typing import Optional

USE_ON_THE_FLY_NOISE_AUG = False
NOISE_PROB = 0.5
SNR_MIN = 10.0
SNR_MAX = 20.0
MIN_NOISE_SECONDS = 3.0

MUSAN_NOISE_ROOT = Path(
    r"C:\Users\User\Desktop\Data\musan\musan\noise"
)


def get_augmentation_metadata(
    train_feature_mode: Optional[str] = None,
    use_noise_aug: Optional[bool] = None,
):
    mode = TRAIN_FEATURE_MODE if train_feature_mode is None else train_feature_mode
    use_aug = USE_ON_THE_FLY_NOISE_AUG if use_noise_aug is None else use_noise_aug
    has_precomputed_noise = mode != "clean"
    has_onthefly_noise = use_aug
    has_noise = has_precomputed_noise or has_onthefly_noise

    return {
        "augmentation": "musan_noise" if has_noise else "clean",
        "train_feature_mode": mode,
        "use_noise_aug": bool(use_aug),
        "noise_prob": NOISE_PROB if has_onthefly_noise else 0.0,
        "snr_min": SNR_MIN if has_noise else None,
        "snr_max": SNR_MAX if has_noise else None,
        "min_noise_seconds": MIN_NOISE_SECONDS if has_noise else None,
        "musan_noise_root": str(MUSAN_NOISE_ROOT) if has_noise else None,
    }
