from __future__ import annotations
from pathlib import Path
import csv
import time
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
import torchaudio
import torch

SRC_SPLITS_ROOT = Path(r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech")
OUT_WAV_ROOT = Path(r"C:\Users\User\Desktop\Data\librispeech_train-clean-100\LibriSpeech_standardized")
TARGET_SR = 16000
TARGET_MONO = True

SPLITS = {
    "train": "train-200spk",
    "val": "val-25spk",
    "test": "test-26spk",
}


@dataclass(frozen=True)
class FileRow:
    split: str
    speaker_id: str
    book_id: str
    utt_id: str
    src_path: str
    out_wav_path: str
    sample_rate: int
    channels: int
    num_samples: int
    duration_sec: float


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_wav_pcm16(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(str(path), wav, sr, encoding="PCM_S", bits_per_sample=16)


def main():
    t0 = time.perf_counter()

    OUT_WAV_ROOT.mkdir(parents=True, exist_ok=True)

    all_tasks: List[Tuple[str, Path]] = []
    for split_name, split_folder in SPLITS.items():
        split_dir = SRC_SPLITS_ROOT / split_folder
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        flacs = sorted([p for p in split_dir.rglob("*.flac") if p.is_file()])
        all_tasks.extend([(split_name, p) for p in flacs])

    print("SRC_SPLITS_ROOT:", SRC_SPLITS_ROOT)
    print("OUT_WAV_ROOT:", OUT_WAV_ROOT)
    print("Total FLAC files found:", len(all_tasks))

    bad: List[Tuple[str, str]] = []
    rows: List[FileRow] = []

    for split_name, flac_path in tqdm(all_tasks, desc="Standardizing FLAC->WAV"):
        parts = flac_path.parts

        split_folder = SPLITS[split_name]
        try:
            i = parts.index(split_folder)
        except ValueError:
            bad.append((str(flac_path), f"Could not locate split folder '{split_folder}' in path parts"))
            continue

        if i + 2 >= len(parts):
            bad.append((str(flac_path), "Path too short to extract speaker_id/book_id"))
            continue

        speaker_id = parts[i + 1]
        book_id = parts[i + 2]
        utt_id = flac_path.stem

        out_wav_path = OUT_WAV_ROOT / split_name / speaker_id / book_id / f"{utt_id}.wav"

        try:
            wav, sr = torchaudio.load(str(flac_path))  # (C, T) float32
            if TARGET_MONO:
                wav = to_mono(wav)
            wav, sr = resample_if_needed(wav, sr, TARGET_SR)
            save_wav_pcm16(out_wav_path, wav, sr)

            ch = int(wav.size(0))
            ns = int(wav.size(1))
            dur = float(ns) / float(sr) if sr > 0 else 0.0

            rows.append(
                FileRow(
                    split=split_name,
                    speaker_id=speaker_id,
                    book_id=book_id,
                    utt_id=utt_id,
                    src_path=str(flac_path),
                    out_wav_path=str(out_wav_path),
                    sample_rate=int(sr),
                    channels=ch,
                    num_samples=ns,
                    duration_sec=dur,
                )
            )

        except Exception as e:
            bad.append((str(flac_path), repr(e)))

    meta_csv = OUT_WAV_ROOT / "standardized_utterances.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "speaker_id",
                "book_id",
                "utt_id",
                "src_path",
                "out_wav_path",
                "sample_rate",
                "channels",
                "num_samples",
                "duration_sec",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.split,
                    r.speaker_id,
                    r.book_id,
                    r.utt_id,
                    r.src_path,
                    r.out_wav_path,
                    r.sample_rate,
                    r.channels,
                    r.num_samples,
                    f"{r.duration_sec:.6f}",
                ]
            )

    dt = time.perf_counter() - t0
    print("\nDONE standardizing.")
    print("Wrote utterance metadata:", meta_csv)
    print("Failed files:", len(bad))
    if bad[:10]:
        print("Failure examples:")
        for p, err in bad[:10]:
            print(" ", p)
            print("   ", err)
    print("Time:", f"{dt:.2f}s")


if __name__ == "__main__":
    main()
