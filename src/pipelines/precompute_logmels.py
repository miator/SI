import os
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

TRAIN_360_500_PCM16_ROOT = Path(
    os.environ.get(
        "SI_TRAIN_360_500_WAV_ROOT",
        r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\wav",
    )
).expanduser()
TRAIN_360_500_LOGMEL_ROOT = Path(
    os.environ.get(
        "SI_TRAIN_360_500_LOGMEL_ROOT",
        r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\logmel_cache",
    )
).expanduser()
TRAIN_360_500_SPLITS = {
    "train-clean-360": {
        "wav_root": TRAIN_360_500_PCM16_ROOT / "train-clean-360",
        "feat_root": TRAIN_360_500_LOGMEL_ROOT / "train-clean-360",
    },
    # "train-other-500": {
    #     "wav_root": TRAIN_360_500_PCM16_ROOT / "train-other-500",
    #     "feat_root": TRAIN_360_500_LOGMEL_ROOT / "train-other-500",
    # },
}


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

        try:
            wav = read_audio_fast(u.path, f.SAMPLE_RATE)

            if wav.numel() == 0 or wav.shape[-1] == 0:
                print(f"Skipping empty wav: {u.path}")
                continue

            if augmenter is not None:
                wav = augmenter(wav)

            feat = fe(wav)  # (frames, n_mels)
            save_feature_tensor(out_path, feat)

        except Exception as e:
            print(f"Skipping bad file: {u.path} | {e}")
            continue


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

    overwrite = False

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

    directories = [split_def["feat_root"] for split_def in TRAIN_360_500_SPLITS.values()]
    for path in dict.fromkeys(directories):
        path.mkdir(parents=True, exist_ok=True)

    for split_name, split_def in TRAIN_360_500_SPLITS.items():
        _precompute_split(
            split_name=split_name,
            wav_root=Path(split_def["wav_root"]),
            feat_root=Path(split_def["feat_root"]),
            fe=fe,
            overwrite=overwrite,
        )

    print("Done.")


if __name__ == "__main__":
    main()
