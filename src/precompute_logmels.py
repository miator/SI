import argparse
import random
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
import warnings

import constants as c
from augment import AdditiveNoise, scan_noise_files, split_noise_paths
from dataset import (
    scan_split,
    read_audio_fast,
    wav_path_to_feature_path,
    save_feature_tensor,
)
from features import LogMelExtraction

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute clean and/or MUSAN-noisy log-mel features."
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="clean",
        choices=["clean", "noise", "both"],
        help="Which train feature set(s) to precompute.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files instead of skipping them.",
    )
    parser.add_argument(
        "--include-noisy-eval",
        action="store_true",
        help="Also precompute the configured noisy evaluation feature splits.",
    )
    return parser.parse_args()


def precompute_split(
    split_name: str,
    wav_root: Path,
    feat_root: Path,
    fe,
    augmenter=None,
    overwrite: bool = False,
    seed: Optional[int] = None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    utts = scan_split(wav_root)
    print(f"[{split_name}] files: {len(utts)}")

    for u in tqdm(utts, desc=f"precompute {split_name}"):
        out_path = wav_path_to_feature_path(u.path, wav_root, feat_root)

        if out_path.exists() and not overwrite:
            continue

        wav = read_audio_fast(u.path, c.SAMPLE_RATE)
        if augmenter is not None:
            wav = augmenter(wav)
        feat = fe(wav)  # (frames, n_mels)
        save_feature_tensor(out_path, feat)


def build_train_eval_noise_file_lists():
    all_noise_paths = scan_noise_files(
        c.MUSAN_NOISE_ROOT,
        sample_rate=c.SAMPLE_RATE,
        min_noise_seconds=c.MIN_NOISE_SECONDS,
    )
    train_noise_paths, eval_noise_paths = split_noise_paths(all_noise_paths)
    if not train_noise_paths:
        raise RuntimeError("No MUSAN noise files were assigned to the training subset.")
    if not eval_noise_paths:
        raise RuntimeError(
            "No MUSAN noise files were assigned to the evaluation subset. "
            "Reduce NOISE_TRAIN_FILES_FRACTION or add more MUSAN noise files."
        )
    return train_noise_paths, eval_noise_paths


def precompute_noisy_eval_splits(fe, overwrite: bool = False):
    _train_noise_paths, eval_noise_paths = build_train_eval_noise_file_lists()

    for idx, (split_name, split_def) in enumerate(c.get_eval_split_definitions().items(), start=1):
        if not split_def["is_noisy"]:
            continue

        augmenter = AdditiveNoise(
            sample_rate=c.SAMPLE_RATE,
            noise_paths=eval_noise_paths,
            prob=1.0,
            snr_min=split_def["snr"],
            snr_max=split_def["snr"],
            min_noise_seconds=c.MIN_NOISE_SECONDS,
        )
        split_def["feat_root"].mkdir(parents=True, exist_ok=True)
        precompute_split(
            split_name=split_name,
            wav_root=split_def["wav_root"],
            feat_root=split_def["feat_root"],
            fe=fe,
            augmenter=augmenter,
            overwrite=overwrite,
            seed=c.NOISE_SPLIT_SEED + idx,
        )


def main():
    args = parse_args()

    fe = LogMelExtraction(
        sample_rate=c.SAMPLE_RATE,
        n_fft=c.N_FFT,
        win_length=c.WIN_LENGTH,
        hop_length=c.HOP_LENGTH,
        n_mels=c.N_MELS,
        f_min=c.FMIN,
        f_max=c.FMAX,
        eps=c.EPS,
    )

    c.TRAIN_CLEAN_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.TRAIN_NOISE_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.VAL_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.TEST_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.VAL_NOISY_SNR15_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.TEST_NOISY_SNR15_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.TEST_NOISY_SNR10_FEAT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.train_mode in {"clean", "both"}:
        precompute_split(
            "train_clean",
            Path(c.TRAIN_ROOT),
            c.TRAIN_CLEAN_FEAT_ROOT,
            fe,
            overwrite=args.overwrite,
        )

    if args.train_mode in {"noise", "both"}:
        augmenter = AdditiveNoise(
            sample_rate=c.SAMPLE_RATE,
            noise_root=c.MUSAN_NOISE_ROOT,
            prob=c.NOISE_PROB,
            snr_min=c.SNR_MIN,
            snr_max=c.SNR_MAX,
            min_noise_seconds=c.MIN_NOISE_SECONDS,
        )
        precompute_split(
            "train_noise",
            Path(c.TRAIN_ROOT),
            c.TRAIN_NOISE_FEAT_ROOT,
            fe,
            augmenter=augmenter,
            overwrite=args.overwrite,
        )

    precompute_split("val", Path(c.VAL_ROOT), c.VAL_FEAT_ROOT, fe, overwrite=args.overwrite)
    precompute_split("test", Path(c.TEST_ROOT), c.TEST_FEAT_ROOT, fe, overwrite=args.overwrite)

    if args.include_noisy_eval:
        precompute_noisy_eval_splits(fe, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
