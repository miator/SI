import random
import warnings
from pathlib import Path

from tqdm import tqdm

import torch
from src.config import augment_config as a
from src.config import data_config as d
from src.config import feature_config as f
from src.data.augment import build_waveform_augmenter
from src.data.dataset import (
    read_audio_fast,
    save_feature_tensor,
    scan_split,
    wav_path_to_feature_path,
)
from src.data.features import LogMelExtraction

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def _precompute_split(
        split_name: str,
        wav_root: Path,
        feat_root: Path,
        fe,
        augmenter=None,
        overwrite: bool = False
):
    utts = scan_split(wav_root)
    print(f"[{split_name}] files: {len(utts)}")

    for u in tqdm(utts, desc=f"precompute {split_name}"):
        out_path = wav_path_to_feature_path(u.path, wav_root, feat_root)

        if out_path.exists() and not overwrite:
            continue

        wav = read_audio_fast(u.path, f.SAMPLE_RATE)
        if augmenter is not None:
            wav = augmenter(wav)

        feat = fe(wav)  # (frames, n_mels)
        save_feature_tensor(out_path, feat)


def _precompute_augmented_eval_splits(fe, overwrite: bool = False):
    for split_name, split_def in d.get_eval_split_definitions().items():
        if not split_def["is_noisy"]:
            continue
        augment_kind = split_def["augment_kind"]
        if augment_kind is None:
            raise RuntimeError(f"No augmenter kind configured for noisy split: {split_name}")

        noise_root = split_def["noise_root"]
        if augment_kind == "noise" and noise_root is None:
            raise RuntimeError(f"No noise root configured for noisy split: {split_name}")

        augmenter = build_waveform_augmenter(
            kind=augment_kind,
            sample_rate=f.SAMPLE_RATE,
            noise_root=noise_root,
            prob=1.0,
            snr_min=split_def["snr"],
            snr_max=split_def["snr"],
        )
        _precompute_split(
            split_name=split_name,
            wav_root=split_def["wav_root"],
            feat_root=split_def["feat_root"],
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite
        )


def _maybe_precompute_train_split(
        *,
        split_name: str,
        feat_root: Path,
        fe,
        overwrite: bool = False,
        augmenter=None,
):
    train_root = Path(d.TRAIN_PRECOMPUTED_ROOT)
    if not train_root.exists():
        print(f"Skipping {split_name}: missing train root {train_root}")
        return

    _precompute_split(
        split_name=split_name,
        wav_root=train_root,
        feat_root=feat_root,
        fe=fe,
        augmenter=augmenter,
        overwrite=overwrite,
    )


def main():
    SEED = 37
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_mode = d.TRAIN_FEATURE_MODE
    overwrite = False
    include_augmented_eval = True
    train_feature_keys = d.get_train_feature_root_keys(train_mode)

    fe = LogMelExtraction(
        sample_rate=f.SAMPLE_RATE,
        n_fft=f.N_FFT,
        win_length=f.WIN_LENGTH,
        hop_length=f.HOP_LENGTH,
        n_mels=f.N_MELS,
        f_min=f.FMIN,
        f_max=f.FMAX,
        eps=f.EPS,
    )

    directories = [
        *d.get_train_feat_roots(train_mode),
        *[
            split_def["feat_root"]
            for split_def in d.get_eval_split_definitions().values()
        ],
    ]
    for path in dict.fromkeys(directories):
        path.mkdir(parents=True, exist_ok=True)

    if "clean" in train_feature_keys:
        _maybe_precompute_train_split(
            split_name="train_clean",
            feat_root=d.TRAIN_CLEAN_FEAT_ROOT,
            fe=fe,
            overwrite=overwrite,
        )

    if "noise" in train_feature_keys:
        augmenter = build_waveform_augmenter(
            kind="noise",
            sample_rate=f.SAMPLE_RATE,
            noise_root=d.ESC50_TRAIN_NOISE_ROOT,
            prob=a.NOISE_PROB,
            snr_min=a.SNR_MIN,
            snr_max=a.SNR_MAX,
        )
        _maybe_precompute_train_split(
            split_name="train_noise",
            feat_root=d.TRAIN_NOISE_FEAT_ROOT,
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite,
        )

    if "white" in train_feature_keys:
        augmenter = build_waveform_augmenter(
            kind="white",
            sample_rate=f.SAMPLE_RATE,
            prob=a.WHITE_NOISE_PROB,
            snr_min=a.WHITE_SNR_MIN,
            snr_max=a.WHITE_SNR_MAX,
        )
        _maybe_precompute_train_split(
            split_name="train_white",
            feat_root=d.TRAIN_WHITE_FEAT_ROOT,
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite,
        )

    for split_name, split_def in d.get_eval_split_definitions().items():
        if split_def["is_noisy"]:
            continue
        _precompute_split(
            split_name=split_name,
            wav_root=Path(split_def["wav_root"]),
            feat_root=Path(split_def["feat_root"]),
            fe=fe,
            overwrite=overwrite,
        )

    if include_augmented_eval:
        _precompute_augmented_eval_splits(fe, overwrite=overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
