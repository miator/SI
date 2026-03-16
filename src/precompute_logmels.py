import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

import constants as c
from augment import AdditiveNoise
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
    return parser.parse_args()


def precompute_split(split_name: str, wav_root: Path, feat_root: Path, fe, augmenter=None, overwrite: bool = False):
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
            noise_root=c.MUSAN_NOISE_ROOT,
            sample_rate=c.SAMPLE_RATE,
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

    print("Done.")


if __name__ == "__main__":
    main()
