import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random

torchaudio.set_audio_backend("soundfile")


def load_csv(path):
    # for testing the find_classes_from_csv function
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_classes_from_folders(train_root: str):
    train_root = Path(train_root)
    classes = sorted([p.name for p in train_root.iterdir() if p.is_dir])
    class_to_index = {name: i for i, name in enumerate(classes)}
    return classes, class_to_index


def build_utterances_from_train_folder(train_root: str, class_to_index: dict):
    train_root = Path(train_root)
    utterances = []
    for speaker_dir in train_root.iterdir():
        if not speaker_dir.is_dir():
            continue
        spk = speaker_dir.name
        if spk not in class_to_index:
            continue
        label = class_to_index[spk]
        for wav_path in speaker_dir.rglob("*.wav"):
            utterances.append((wav_path, label))
    return utterances


def find_classes(rows, speaker_col: str):
    # the name says itself☺
    classes = sorted({row[speaker_col] for row in rows})
    class_to_index = {spk: i for i, spk in enumerate(classes)}
    return classes, class_to_index


def build_utterances(csv_rows, wav_root: str, class_to_index: dict,
                     path_col: str, speaker_col: str):
    wav_root = Path(wav_root)
    utterances = []
    for row in csv_rows:
        rel = Path(row[path_col])
        audio_path = rel if rel.is_absolute() else wav_root / rel

        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing file: {audio_path}")

        label = class_to_index[row[speaker_col]]
        utterances.append((audio_path, label))
    return utterances


# add multithreading for I/O task
def read_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """ Load wav as mono float32 torch tensor. Shape: (T,) """
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0)  # (T, )
    return wav.to(dtype=torch.float32)


class AudioDataset(Dataset):
    def __init__(self, utterances, sample_rate: int, feature_extractor):
        self.utterances = utterances
        self.sample_rate = sample_rate
        self.fe = feature_extractor
        self.cache = {}

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):  # how to load one sample
        path, label = self.utterances[idx]
        key = str(path)
        if key in self.cache:
            feat = self.cache[key]
        else:
            wav = read_audio(path, self.sample_rate)  # (T, )
            feat = self.fe(wav)  # (frames, n_mfcc)
            self.cache[key] = feat
        return feat, torch.tensor(label, dtype=torch.long)


def pad_trunc_collate(max_frames: int):
    def _fn(batch):
        feats, labels = zip(*batch)  # feats: List[(Ti, F)]
        f = feats[0].shape[1]
        out = torch.zeros((len(feats), max_frames, f), dtype=feats[0].dtype)
        lengths = torch.zeros((len(feats),), dtype=torch.long)

        for i, x in enumerate(feats):
            t = x.shape[0]
            t2 = min(t, max_frames)
            out[i, :t2] = x[:t2]
            lengths[i] = t2

        return out, torch.stack(labels), lengths
    return _fn


def create_data_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def pad_trunc_collate_fn(batch, max_frames: int):
    feats, labels = zip(*batch)
    f = feats[0].shape[1]
    out = torch.zeros((len(feats), max_frames, f), dtype=feats[0].dtype)
    lengths = torch.zeros((len(feats),), dtype=torch.long)

    for i, x in enumerate(feats):
        t = x.shape[0]
        t2 = min(t, max_frames)
        out[i, :t2] = x[:t2]
        lengths[i] = t2

    return out, torch.stack(labels), lengths


def pad_collate(batch):
    wavs, labels = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.stack([torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in wavs])
    labels = torch.stack(labels)
    return padded, labels, lengths


def split_train_folder_by_speaker(train_root: str, ratios=(0.8, 0.1, 0.1), seed=37):
    train_root = Path(train_root)
    r_tr, r_va, r_te = ratios
    assert abs((r_tr + r_va + r_te) - 1.0) < 1e-6

    rng = random.Random(seed)

    train_utts, val_utts, test_utts = [], [], []

    speaker_dirs = [p for p in train_root.iterdir() if p.is_dir()]
    speaker_dirs.sort(key=lambda p: p.name)

    classes = [p.name for p in speaker_dirs]
    class_to_index = {name: i for i, name in enumerate(classes)}

    for spk_dir in speaker_dirs:
        spk = spk_dir.name
        label = class_to_index[spk]

        files = sorted(spk_dir.rglob("*.wav"))
        if len(files) < 3:
            # too few files: keep all in train to avoid empty splits
            for f in files:
                train_utts.append((f, label))
            continue

        rng.shuffle(files)

        n = len(files)
        n_tr = max(1, int(n * r_tr))
        n_va = max(1, int(n * r_va))
        # remainder goes to test
        n_va = min(n_va, n - n_tr - 1)

        tr = files[:n_tr]
        va = files[n_tr:n_tr + n_va]
        te = files[n_tr + n_va:]

        train_utts += [(f, label) for f in tr]
        val_utts += [(f, label) for f in va]
        test_utts += [(f, label) for f in te]

    return classes, class_to_index, train_utts, val_utts, test_utts
