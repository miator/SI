from pathlib import Path
from tqdm import tqdm
import warnings

import constants as c
from dataset import (
    scan_split,
    read_audio_fast,
    wav_path_to_feature_path,
    save_feature_tensor,
)
from features import LogMelExtraction

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def precompute_split(split_name: str, wav_root: Path, feat_root: Path, fe):
    utts = scan_split(wav_root)
    print(f"[{split_name}] files: {len(utts)}")

    for u in tqdm(utts, desc=f"precompute {split_name}"):
        out_path = wav_path_to_feature_path(u.path, wav_root, feat_root)

        if out_path.exists():
            continue

        wav = read_audio_fast(u.path, c.SAMPLE_RATE)
        feat = fe(wav)  # (frames, n_mels)
        save_feature_tensor(out_path, feat)


def main():
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

    c.TRAIN_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.VAL_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    c.TEST_FEAT_ROOT.mkdir(parents=True, exist_ok=True)

    precompute_split("train", Path(c.TRAIN_ROOT), c.TRAIN_FEAT_ROOT, fe)
    precompute_split("val", Path(c.VAL_ROOT), c.VAL_FEAT_ROOT, fe)
    precompute_split("test", Path(c.TEST_ROOT), c.TEST_FEAT_ROOT, fe)

    print("Done.")


if __name__ == "__main__":
    main()
