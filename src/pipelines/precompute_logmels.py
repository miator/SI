import random
import warnings
from pathlib import Path

from tqdm import tqdm

import torch
from src.config import augment_config as a
from src.config import data_config as d
from src.config import feature_config as f
from src.data.augment import AdditiveNoise, scan_noise_files, split_noise_paths
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


def _build_train_eval_noise_file_lists():
    all_noise_paths = scan_noise_files(
        d.MUSAN_NOISE_ROOT,
        sample_rate=f.SAMPLE_RATE,
        min_noise_seconds=a.MIN_NOISE_SECONDS,
    )
    train_noise_paths, eval_noise_paths = split_noise_paths(all_noise_paths)

    if not train_noise_paths:
        raise RuntimeError("No noise files were assigned to the training subset.")
    if not eval_noise_paths:
        raise RuntimeError(
            "No noise files were assigned to the evaluation subset. "
            "Reduce NOISE_TRAIN_FILES_FRACTION or add more noise files."
        )

    return train_noise_paths, eval_noise_paths


def _precompute_noisy_eval_splits(fe, overwrite: bool = False):
    _, eval_noise_paths = _build_train_eval_noise_file_lists()

    for idx, (split_name, split_def) in enumerate(
        d.get_eval_split_definitions().items(), start=1
    ):
        if not split_def["is_noisy"]:
            continue

        augmenter = AdditiveNoise(
            sample_rate=f.SAMPLE_RATE,
            noise_paths=eval_noise_paths,
            prob=1.0,
            snr_min=split_def["snr"],
            snr_max=split_def["snr"],
            min_noise_seconds=a.MIN_NOISE_SECONDS,
        )
        _precompute_split(
            split_name=split_name,
            wav_root=split_def["wav_root"],
            feat_root=split_def["feat_root"],
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite
        )


def main():
    SEED = 37
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_mode = "clean"          # "clean", "noise", "both"
    overwrite = False             # False = skip existing, True = recompute
    include_noisy_eval = False    # True = also precompute noisy eval splits

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
        d.TRAIN_CLEAN_FEAT_ROOT,
        d.TRAIN_NOISE_FEAT_ROOT,
        d.VAL_FEAT_ROOT,
        d.TEST_FEAT_ROOT,
        d.VAL_NOISY_SNR15_FEAT_ROOT,
        d.TEST_NOISY_SNR15_FEAT_ROOT,
        d.TEST_NOISY_SNR10_FEAT_ROOT,
    ]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)

    if train_mode in {"clean", "both"}:
        _precompute_split(
            split_name="train_clean",
            wav_root=Path(d.TRAIN_ROOT),
            feat_root=d.TRAIN_CLEAN_FEAT_ROOT,
            fe=fe,
            overwrite=overwrite,
        )

    if train_mode in {"noise", "both"}:
        augmenter = AdditiveNoise(
            sample_rate=f.SAMPLE_RATE,
            noise_root=d.MUSAN_NOISE_ROOT,
            prob=a.NOISE_PROB,
            snr_min=a.SNR_MIN,
            snr_max=a.SNR_MAX,
            min_noise_seconds=a.MIN_NOISE_SECONDS,
        )
        _precompute_split(
            split_name="train_noise",
            wav_root=Path(d.TRAIN_ROOT),
            feat_root=d.TRAIN_NOISE_FEAT_ROOT,
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite,
        )

    _precompute_split(
        split_name="val",
        wav_root=Path(d.VAL_ROOT),
        feat_root=d.VAL_FEAT_ROOT,
        fe=fe,
        overwrite=overwrite,
    )
    _precompute_split(
        split_name="test",
        wav_root=Path(d.TEST_ROOT),
        feat_root=d.TEST_FEAT_ROOT,
        fe=fe,
        overwrite=overwrite,
    )

    if include_noisy_eval:
        _precompute_noisy_eval_splits(fe, overwrite=overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
