from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_STD_WAV_ROOT = Path(r"C:\Users\User\Desktop\Data\librispeech-train-clean-100/LibriSpeech_standardized")
OUT_CHUNKS_ROOT = Path(r"C:\Users\User\Desktop\Data\librispeech-train-clean-100/LibriSpeech_standardized_chunks_3s")

WRITE_DUPLICATE_PCM16_TREE = True

SPLITS = ["train", "val", "test"]

TARGET_SR = 16000
CHUNK_SECONDS = 3.0
HOP_SECONDS = 3.0  # no overlap
CHUNK_LEN = int(round(TARGET_SR * CHUNK_SECONDS))
HOP_LEN = int(round(TARGET_SR * HOP_SECONDS))

# Multiprocessing workers (CPU)
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)


@dataclass(frozen=True)
class ChunkRow:
    split: str
    speaker_id: str
    book_id: str
    utt_id: str
    chunk_idx: int
    start_sample: int
    end_sample: int
    sample_rate: int
    channels: int
    num_samples: int
    duration_sec: float
    src_wav_path: str
    out_wav_path: str
    out_pcm16_path: str


def load_wav_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.to(torch.float32)


def save_pcm16_wav(path: Path, wav_1ch: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_1ch = torch.clamp(wav_1ch, -1.0, 1.0)
    torchaudio.save(str(path), wav_1ch, sr, encoding="PCM_S", bits_per_sample=16)


def iter_full_chunks(wav_1ch: torch.Tensor) -> List[Tuple[int, int, torch.Tensor]]:
    """
    Return list of (start, end, chunk_tensor) for full non-overlapping chunks only.
    Discards tails; no padding.
    """
    assert wav_1ch.dim() == 2 and wav_1ch.size(0) == 1
    T = int(wav_1ch.size(1))
    if T < CHUNK_LEN:
        return []

    chunks = []
    # Non-overlapping: starts = 0, CHUNK_LEN, 2*CHUNK_LEN, ...
    for s in range(0, T - CHUNK_LEN + 1, HOP_LEN):
        e = s + CHUNK_LEN
        chunks.append((s, e, wav_1ch[:, s:e]))
    return chunks


# WORKER: PROCESS ONE UTTERANCE FILE
def process_one_file(task: Dict[str, Any]) -> List[ChunkRow]:
    """
    Worker executed in a separate process.
    Reads one utterance WAV, produces and writes chunk WAVs, returns metadata rows.
    """
    split = task["split"]
    speaker_id = task["speaker_id"]
    book_id = task["book_id"]
    utt_id = task["utt_id"]
    src_path = Path(task["src_path"])

    out_wav_dir = OUT_CHUNKS_ROOT / "wav" / split / speaker_id
    out_pcm16_dir = OUT_CHUNKS_ROOT / "pcm16" / split / speaker_id

    wav = load_wav_mono_16k(src_path)
    chunks = iter_full_chunks(wav)
    if not chunks:
        return []

    rows: List[ChunkRow] = []

    for idx, (s, e, chunk) in enumerate(chunks):
        out_name = f"{utt_id}_c{idx:04d}.wav"

        out_wav_path = out_wav_dir / out_name
        save_pcm16_wav(out_wav_path, chunk, TARGET_SR)

        # Optional duplicate tree
        out_pcm16_path_str = ""
        if WRITE_DUPLICATE_PCM16_TREE:
            out_pcm16_path = out_pcm16_dir / out_name
            save_pcm16_wav(out_pcm16_path, chunk, TARGET_SR)
            out_pcm16_path_str = str(out_pcm16_path)

        rows.append(
            ChunkRow(
                split=split,
                speaker_id=speaker_id,
                book_id=book_id,
                utt_id=utt_id,
                chunk_idx=idx,
                start_sample=s,
                end_sample=e,
                sample_rate=TARGET_SR,
                channels=1,
                num_samples=CHUNK_LEN,
                duration_sec=CHUNK_SECONDS,
                src_wav_path=str(src_path),
                out_wav_path=str(out_wav_path),
                out_pcm16_path=out_pcm16_path_str,
            )
        )

    return rows


# 4) BUILD TASK LIST FROM STANDARDIZED TREE
def build_tasks() -> List[Dict[str, Any]]:
    """
    Scan SRC_STD_WAV_ROOT for utterances:
      SRC_STD_WAV_ROOT/<split>/<speaker>/<book>/<utt>.wav
    Return a list of dict tasks with extracted ids.
    """
    tasks: List[Dict[str, Any]] = []

    for split in SPLITS:
        split_dir = SRC_STD_WAV_ROOT / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        # speaker folders directly under split
        speaker_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name)

        for spk_dir in speaker_dirs:
            speaker_id = spk_dir.name
            # book folders under speaker
            book_dirs = sorted([p for p in spk_dir.iterdir() if p.is_dir()], key=lambda p: p.name)

            for book_dir in book_dirs:
                book_id = book_dir.name
                wav_files = sorted([p for p in book_dir.glob("*.wav") if p.is_file()])

                for wav_path in wav_files:
                    utt_id = wav_path.stem
                    tasks.append(
                        {
                            "split": split,
                            "speaker_id": speaker_id,
                            "book_id": book_id,
                            "utt_id": utt_id,
                            "src_path": str(wav_path),
                        }
                    )

    return tasks


# 5) MAIN CHUNKING PIPELINE
def main() -> None:
    t0 = time.perf_counter()

    OUT_CHUNKS_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_CHUNKS_ROOT / "wav").mkdir(parents=True, exist_ok=True)
    if WRITE_DUPLICATE_PCM16_TREE:
        (OUT_CHUNKS_ROOT / "pcm16").mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    print("SRC_STD_WAV_ROOT:", SRC_STD_WAV_ROOT)
    print("OUT_CHUNKS_ROOT:", OUT_CHUNKS_ROOT)
    print("Utterances found:", len(tasks))
    print("MAX_WORKERS:", MAX_WORKERS)

    all_rows: List[ChunkRow] = []
    failures: List[Tuple[str, str]] = []
    produced_chunks = 0

    # Multiprocessing over utterances
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunking"):
            try:
                rows = fut.result()
                if rows:
                    all_rows.extend(rows)
                    produced_chunks += len(rows)
            except Exception as e:
                failures.append(("<worker>", repr(e)))

    # Write metadata CSV (one file for all splits)
    meta_csv = OUT_CHUNKS_ROOT / "chunks_metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "speaker_id",
                "book_id",
                "utt_id",
                "chunk_idx",
                "start_sample",
                "end_sample",
                "sample_rate",
                "channels",
                "num_samples",
                "duration_sec",
                "src_wav_path",
                "out_wav_path",
                "out_pcm16_path",
            ]
        )
        for r in all_rows:
            w.writerow(
                [
                    r.split,
                    r.speaker_id,
                    r.book_id,
                    r.utt_id,
                    r.chunk_idx,
                    r.start_sample,
                    r.end_sample,
                    r.sample_rate,
                    r.channels,
                    r.num_samples,
                    f"{r.duration_sec:.3f}",
                    r.src_wav_path,
                    r.out_wav_path,
                    r.out_pcm16_path,
                ]
            )

    dt = time.perf_counter() - t0
    print("\nDONE chunking.")
    print("Chunks produced:", produced_chunks)
    print("Metadata:", meta_csv)
    print("Failures:", len(failures))
    if failures[:10]:
        print("Failure examples:")
        for p, err in failures[:10]:
            print(" ", p)
            print("   ", err)
    print("Time:", f"{dt:.2f}s")
    if produced_chunks > 0:
        print("Avg time per chunk:", f"{dt / produced_chunks:.6f}s")


if __name__ == "__main__":
    main()