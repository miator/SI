from collections import Counter
import constants as c
from pathlib import Path
from dataset import load_csv, read_audio
from features import LogMelExtraction

root_train = Path(c.TRAIN_WAV_ROOT)
rows_train = load_csv(c.TRAIN_CSV)
counts = {spk.name: len(list(spk.rglob("*.wav")))
          for spk in root_train.iterdir() if spk.is_dir()}

total = sum(counts.values())
n_spk = len(counts)

print("Speakers:", n_spk)
print("Min files:", min(counts.values()))
print("Max files:", max(counts.values()))
print("Avg files:", total / n_spk)
print("Total train files:", len(rows_train))

# Optional: show most imbalanced
for k, v in sorted(counts.items(), key=lambda x: x[1]):
    print(k, v)


rows_test = load_csv(c.TEST_CSV)
speakers = [row["speaker"] for row in rows_test]
paths = [row["file_path"] for row in rows_test]

cnt = Counter(speakers)

print("Total test files:", len(rows_test))
print("Speakers:", len(cnt))
print("Min files:", min(cnt.values()))
print("Max files:", max(cnt.values()))
print("Avg files:", sum(cnt.values()) / len(cnt))

for spk, n in sorted(cnt.items(), key=lambda x: x[1]):
    print(spk, n)

# sanity check: paths exist
missing = [p for p in paths if not (Path(c.TEST_WAV_ROOT) / p).is_file()]
print("Missing files:", len(missing))
if missing:
    print(missing[:5])


# checks LogMelExtraction functionality
wav = read_audio(Path("C://Users/User/desktop/data/train/clb/arctic_a0145.wav"), sample_rate=c.SAMPLE_RATE)
extractor = LogMelExtraction(
    sample_rate=c.SAMPLE_RATE,
    n_fft=c.N_FFT,
    win_length=c.WIN_LENGTH,
    hop_length=c.HOP_LENGTH,
    n_mels=c.N_MELS,
    f_min=c.FMIN,
    f_max=c.FMAX,
    eps=1e-5
)

x = extractor(wav)
print("shape:", x.shape)
print("dtype:", x.dtype)
print("min:", x.min().item(), "max:", x.max().item())
