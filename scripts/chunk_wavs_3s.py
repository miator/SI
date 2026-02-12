from __future__ import annotations
import csv
import time
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torchaudio


@dataclass(frozen=True)
class ChunkCfg:
    sample_rate: int = 16000
    chunk_seconds: float = 3.0
    hop_seconds: float = 3.0
    # min_tail_seconds: float = 2.0
    mono: bool = True


def load_mono_resample(path: Path, target_sr: int, mono: bool) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if mono and wav.dim() == 2:
        wav = wav.mean(dim=0, keepdim=True)  # (1, T)
    return wav.to(torch.float32)  # (1, T) float32


def iter_chunks(wav_1ch: torch.Tensor, cfg: ChunkCfg) -> Iterable[Tuple[int, int, torch.Tensor]]:
    """
    Yields (start_sample, end_sample, chunk_tensor) where chunk_tensor is (1, chunk_len).
    Pads are NOT used; only full chunks plus optional last tail chunk.
    """
    assert wav_1ch.dim() == 2 and wav_1ch.size(0) == 1
    T = int(wav_1ch.size(1))

    chunk_len = int(round(cfg.chunk_seconds * cfg.sample_rate))
    hop = int(round(cfg.hop_seconds * cfg.sample_rate))
    # min_tail = int(round(cfg.min_tail_seconds * cfg.sample_rate))

    if T < chunk_len:
        return  # skip too-short files

    starts = list(range(0, T - chunk_len + 1, hop))

    # optional last chunk to cover tail
    # tail = T - (starts[-1] + chunk_len) if starts else T - chunk_len
    # if tail >= min_tail:
    #     last_start = T - chunk_len
    #     if not starts or last_start != starts[-1]:
    #         starts.append(last_start)

    for s in starts:
        e = s + chunk_len
        yield s, e, wav_1ch[:, s:e]


def chunk_dataset(
    src_root: Path,
    out_root: Path,
    cfg: ChunkCfg,
    manifest_csv: Path,
) -> None:

    """
    src_root/
      Speaker_0000/*.wav
      Speaker_0001/*.wav
    out_root/
      Speaker_0000/<origstem>__s000000__e048000.wav
      ...
    manifest_csv columns: utt_id, filepath, speaker
    """

    t0_global = time.perf_counter()

    src_root = Path(src_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    speaker_dirs = sorted([p for p in src_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    total_chunks = 0

    for spk_dir in tqdm(speaker_dirs, desc="Speakers"):
        t0_spk = time.perf_counter()

        spk = spk_dir.name
        (out_root / spk).mkdir(parents=True, exist_ok=True)

        wav_files = sorted([p for p in spk_dir.rglob("*.wav") if p.is_file()])

        spk_chunks = 0

        for wav_path in wav_files:
            wav = load_mono_resample(wav_path, cfg.sample_rate, cfg.mono)

            for s, e, chunk in iter_chunks(wav, cfg):
                out_name = f"{wav_path.stem}__s{s:07d}__e{e:07d}.wav"
                out_path = out_root / spk / out_name

                torchaudio.save(str(out_path), chunk, cfg.sample_rate)

                utt_id = out_path.stem
                rel_path = out_path.as_posix()
                rows.append((utt_id, rel_path, spk))

                spk_chunks += 1
                total_chunks += 1

        t_spk = time.perf_counter() - t0_spk
        print(f"[{spk}] chunks: {spk_chunks} | time: {t_spk:.2f}s")

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "filepath", "speaker"])
        w.writerows(rows)

    t_total = time.perf_counter() - t0_global
    print(f"\nTotal chunks: {total_chunks}")
    print(f"Total time: {t_total:.2f}s")
    print(f"Avg time per chunk: {t_total / max(total_chunks,1):.4f}s")


if __name__ == "__main__":
    SRC = Path(r"C:\Users\User\Desktop\50spk_1h\pcm16_16k")
    OUT = Path(r"C:\Users\User\Desktop\50spk_1h\pcm16_16k_chunks_3s")
    MANIFEST = OUT / "all_chunks.csv"

    cfg = ChunkCfg(
        sample_rate=16000,
        chunk_seconds=3.0,
        hop_seconds=3.0,
        # min_tail_seconds=2.0,
        mono=True,
    )

    chunk_dataset(SRC, OUT, cfg, MANIFEST)
    print("Done.")
    print("Chunks saved to:", OUT)
    print("Manifest saved to:", MANIFEST)
