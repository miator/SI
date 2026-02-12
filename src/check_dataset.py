from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
# import statistics as stats

import torchaudio


ROOT = Path(r"C:\Users\User\Desktop\50spk_1h\pcm16_16k")


def sec_to_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:d}h {m:02d}m {s:05.2f}s"
    return f"{m:d}m {s:05.2f}s"


def main():
    speaker_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)
    print("Root:", ROOT)
    print("Speakers:", len(speaker_dirs))

    ext_cnt = Counter()
    sr_cnt = Counter()
    ch_cnt = Counter()

    # per speaker stats
    spk_file_counts = {}
    spk_durations = {}  # seconds
    spk_bad_files = defaultdict(list)

    bad_info = []   # cannot read metadata
    bad_load = []   # cannot load waveform

    total_files = 0
    total_seconds = 0.0

    for spk_dir in speaker_dirs:
        wavs = sorted([p for p in spk_dir.rglob("*") if p.is_file()])
        spk_total = 0.0
        spk_ok_files = 0

        for p in wavs:
            total_files += 1
            ext_cnt[p.suffix.lower()] += 1

            # Try metadata first (cheap)
            try:
                info = torchaudio.info(str(p))
            except Exception as e:
                bad_info.append((str(p), repr(e)))
                spk_bad_files[spk_dir.name].append((p.name, "info_fail"))
                continue

            sr_cnt[int(info.sample_rate)] += 1
            ch_cnt[int(info.num_channels)] += 1

            # Duration from metadata
            dur = float(info.num_frames) / float(info.sample_rate) if info.sample_rate > 0 else 0.0

            # Now try a real decode load (catches mpg123 errors)
            try:
                _wav, _sr = torchaudio.load(str(p))
            except Exception as e:
                bad_load.append((str(p), repr(e)))
                spk_bad_files[spk_dir.name].append((p.name, "load_fail"))
                continue

            spk_ok_files += 1
            spk_total += dur

        spk_file_counts[spk_dir.name] = spk_ok_files
        spk_durations[spk_dir.name] = spk_total
        total_seconds += spk_total

    print("\n--- Extensions (all files found) ---")
    for k, v in ext_cnt.most_common():
        print(f"{k or '<noext>'}: {v}")

    print("\n--- Sample rates (info ok files) ---")
    for k, v in sr_cnt.most_common():
        print(f"{k}: {v}")

    print("\n--- Channels (info ok files) ---")
    for k, v in ch_cnt.most_common():
        print(f"{k}: {v}")

    ok_speakers = list(spk_file_counts.keys())
    file_counts = [spk_file_counts[s] for s in ok_speakers]
    durations = [spk_durations[s] for s in ok_speakers]

    print("\n--- Per-speaker summary (decoded ok files only) ---")
    print("Total decoded files:", sum(file_counts))
    print("Total decoded duration:", sec_to_hms(total_seconds))
    if file_counts:
        print("Files per speaker: min/avg/max =",
              min(file_counts), f"{sum(file_counts)/len(file_counts):.2f}", max(file_counts))
    if durations:
        print("Duration per speaker: min/avg/max =",
              sec_to_hms(min(durations)), sec_to_hms(sum(durations)/len(durations)), sec_to_hms(max(durations)))

    # List a few worst speakers by duration
    worst = sorted(spk_durations.items(), key=lambda x: x[1])[:10]
    print("\n--- Lowest-duration speakers (top 10) ---")
    for spk, sec in worst:
        print(f"{spk}: {sec_to_hms(sec)} (files {spk_file_counts[spk]})")

    # Bad files report
    print("\n--- Bad files ---")
    print("info() failures:", len(bad_info))
    if bad_info[:10]:
        print("Examples (info_fail):")
        for path, err in bad_info[:10]:
            print(" ", path)
            print("   ", err)

    print("load() failures:", len(bad_load))
    if bad_load[:10]:
        print("Examples (load_fail):")
        for path, err in bad_load[:10]:
            print(" ", path)
            print("   ", err)

    # Speakers with any bad files
    spk_with_bad = [(spk, len(items)) for spk, items in spk_bad_files.items()]
    spk_with_bad.sort(key=lambda x: x[1], reverse=True)
    print("\nSpeakers with bad files:", len(spk_with_bad))
    for spk, nbad in spk_with_bad[:10]:
        print(f"{spk}: bad_files={nbad}")


if __name__ == "__main__":
    main()
