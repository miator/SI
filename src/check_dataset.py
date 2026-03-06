from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter

import torchaudio


def sec_to_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:05.2f}s"
    return f"{m:d}m {s:05.2f}s"


def percentile(sorted_vals: list[float], p: float) -> float:
    # p in [0, 100]
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if c == f:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ext", type=str, default=".flac", help="Audio extension to scan (default: .flac)")
    ap.add_argument(
        "--bin",
        type=float,
        default=1.0,
        help="Histogram bin size in seconds (default: 1.0)",
    )
    ap.add_argument(
        "--max_bins",
        type=int,
        default=40,
        help="Max histogram bins printed (default: 40). Last bin is overflow.",
    )
    ap.add_argument(
        "--write_csv",
        action="store_true",
        help="Write per-speaker CSV summary in the current directory.",
    )
    args = ap.parse_args()

    ROOT = Path(r"C:\Users\User\Desktop\Data\librispeech-train-clean-100\LibriSpeech\train-clean-100")
    EXT = args.ext.lower()

    if not ROOT.exists():
        raise FileNotFoundError(f"Root does not exist: {ROOT}")

    speaker_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
    print("Root:", ROOT)
    print("Speakers:", len(speaker_dirs))
    print("Extension:", EXT)

    ext_cnt = Counter()
    sr_cnt = Counter()
    ch_cnt = Counter()

    # Per speaker
    spk_utt_counts: dict[str, int] = {}
    spk_book_counts: dict[str, int] = {}
    spk_seconds: dict[str, float] = {}

    # Per book (optional useful stat)
    books_per_speaker = []

    # Per utterance durations
    durs = []
    bad_info = []   # cannot read metadata

    total_files = 0
    total_seconds = 0.0

    # Histogram
    bin_w = float(args.bin)
    nb = int(args.max_bins)
    hist = [0] * nb  # last bin is overflow (>= (nb-1)*bin_w)

    def hist_add(x: float) -> None:
        if x < 0:
            return
        idx = int(x // bin_w)
        if idx >= nb - 1:
            hist[-1] += 1
        else:
            hist[idx] += 1

    for spk_dir in speaker_dirs:
        book_dirs = sorted([b for b in spk_dir.iterdir() if b.is_dir()], key=lambda p: p.name)
        spk_book_counts[spk_dir.name] = len(book_dirs)
        books_per_speaker.append(len(book_dirs))

        files = sorted([p for p in spk_dir.rglob(f"*{EXT}") if p.is_file()])
        spk_ok = 0
        spk_sec = 0.0

        for p in files:
            total_files += 1
            ext_cnt[p.suffix.lower()] += 1

            # Try metadata first (cheap)
            try:
                info = torchaudio.info(str(p))
            except Exception as e:
                bad_info.append((str(p), repr(e)))
                continue

            sr_cnt[int(info.sample_rate)] += 1
            ch_cnt[int(info.num_channels)] += 1

            # Duration from metadata
            dur = float(info.num_frames) / float(info.sample_rate) if info.sample_rate > 0 else 0.0
            spk_ok += 1
            spk_sec += dur

            durs.append(dur)
            hist_add(dur)

        spk_utt_counts[spk_dir.name] = spk_ok
        spk_seconds[spk_dir.name] = spk_sec
        total_seconds += spk_sec

    print("\n--- Extensions (all files found) ---")
    for k, v in ext_cnt.most_common():
        print(f"{k or '<noext>'}: {v}")

    print("\n--- Sample rates (info ok files) ---")
    for k, v in sr_cnt.most_common():
        print(f"{k}: {v}")

    print("\n--- Channels (info ok files) ---")
    for k, v in ch_cnt.most_common():
        print(f"{k}: {v}")

    # Global duration stats
    durs_sorted = sorted(durs)
    if durs_sorted:
        mn = durs_sorted[0]
        mx = durs_sorted[-1]
        avg = sum(durs_sorted) / len(durs_sorted)

        print("\n--- Utterance duration stats (seconds) ---")
        print("Count:", len(durs_sorted))
        print("Total duration:", sec_to_hms(total_seconds))
        print(f"Min/Avg/Max: {mn:.3f} / {avg:.3f} / {mx:.3f}")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"P{p:02d}: {percentile(durs_sorted, p):.3f}")

        print("\n--- Duration histogram ---")
        # bins: [0,bin), [bin,2bin), ...
        for i in range(nb - 1):
            a = i * bin_w
            b = (i + 1) * bin_w
            c = hist[i]
            if c == 0:
                continue
            print(f"[{a:5.1f}, {b:5.1f}): {c}")
        if hist[-1] > 0:
            print(f"[{(nb-1)*bin_w:5.1f}, +inf): {hist[-1]}")

    # Speaker distribution stats
    spk_secs_sorted = sorted(spk_seconds.items(), key=lambda x: x[1])

    print("\n--- Per-speaker distribution ---")
    print("Speakers with any decoded files:", len(spk_secs_sorted))
    if spk_secs_sorted:
        secs_only = sorted([v for _, v in spk_secs_sorted])
        print("Speaker total duration: min/avg/max =",
              sec_to_hms(secs_only[0]),
              sec_to_hms(sum(secs_only)/len(secs_only)),
              sec_to_hms(secs_only[-1]))
        print("Speaker total utterances: min/avg/max =",
              min(spk_utt_counts.values()),
              f"{sum(spk_utt_counts.values())/len(spk_utt_counts):.2f}",
              max(spk_utt_counts.values()))
        if books_per_speaker:
            print("Books per speaker: min/avg/max =",
                  min(books_per_speaker),
                  f"{sum(books_per_speaker)/len(books_per_speaker):.2f}",
                  max(books_per_speaker))

    print("\nBottom 10 speakers by total seconds:")
    for spk, sec in spk_secs_sorted[:10]:
        print(f"{spk}: {sec_to_hms(sec)} | utt={spk_utt_counts.get(spk, 0)} | books={spk_book_counts.get(spk, 0)}")

    print("\nTop 10 speakers by total seconds:")
    for spk, sec in spk_secs_sorted[-10:][::-1]:
        print(f"{spk}: {sec_to_hms(sec)} | utt={spk_utt_counts.get(spk, 0)} | books={spk_book_counts.get(spk, 0)}")

    print("\n--- Bad files ---")
    print("info() failures:", len(bad_info))
    if bad_info[:10]:
        print("Examples:")
        for path, err in bad_info[:10]:
            print(" ", path)
            print("   ", err)


if __name__ == "__main__":
    main()
