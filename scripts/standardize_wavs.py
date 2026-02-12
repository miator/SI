from __future__ import annotations
from pathlib import Path
import time
from tqdm import tqdm
import torchaudio
import torch
import io
import contextlib

SRC = Path(r"C:\Users\User\Desktop\50spk_1h\50_speakers_audio_data")
OUT = Path(r"C:\Users\User\Desktop\50spk_1h\pcm16_16k")
TARGET_SR = 16000


def load_any(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))  # (C, T), float32
    return wav, sr


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> tuple[torch.Tensor, int]:
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def save_pcm16(path: Path, wav: torch.Tensor, sr: int) -> None:
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(str(path), wav, sr, encoding="PCM_S", bits_per_sample=16)


def load_with_warning_capture(path: Path):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        wav, sr = torchaudio.load(str(path))
    warn = buf.getvalue()
    return wav, sr, warn


def main():
    t0 = time.perf_counter()

    wav_files = sorted([p for p in SRC.rglob("*.wav") if p.is_file()])
    print("Found files:", len(wav_files))

    OUT.mkdir(parents=True, exist_ok=True)

    bad = []
    warned = []
    for p in tqdm(wav_files, desc="Standardizing"):
        rel = p.relative_to(SRC)
        out_path = OUT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                wav, sr = torchaudio.load(str(p))

            warn_msg = buf.getvalue()

            if "layer3.c" in warn_msg or "mpg123" in warn_msg:
                warned.append(str(p))
                continue  # skip writing this file

            wav = to_mono(wav)
            wav, sr = resample_if_needed(wav, sr, TARGET_SR)
            save_pcm16(out_path, wav, sr)

        except Exception as e:
            bad.append((str(p), repr(e)))

    warn_file = OUT / "warned_files.txt"
    with open(warn_file, "w", encoding="utf-8") as f:
        for path in warned:
            f.write(path + "\n")

    print("Files with mpg123 warnings:", len(warned))
    print("List saved to:", warn_file)

    dt = time.perf_counter() - t0
    print("Done in:", f"{dt:.2f}s")
    print("Failed files:", len(bad))
    if bad[:10]:
        print("Examples:")
        for path, err in bad[:10]:
            print(path)
            print(" ", err)


if __name__ == "__main__":
    main()
