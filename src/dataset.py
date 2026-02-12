import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

PathLike = Union[str, Path]
Utterance = Tuple[Path, int]


def find_classes_from_folders(root: PathLike) -> Tuple[List[str], Dict[str, int]]:
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_index = {name: i for i, name in enumerate(classes)}
    return classes, class_to_index


def build_utterances_from_train_folder(root: PathLike,
                                       class_to_index: Dict[str, int],
                                       exts: Tuple[str, ...] = (".wav",)) -> List[Utterance]:
    """Scans each speaker folder and returns list of (wav_path,  label)"""
    root = Path(root)
    utterances: List[Utterance] = []
    for spk_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        spk = spk_dir.name
        if spk not in class_to_index:
            continue
        label = class_to_index[spk]
        for wav_path in sorted(spk_dir.rglob("*.")):
            if wav_path.suffix.lower() in exts:
                utterances.append((wav_path, label))
    return utterances


def split_by_speaker(root: PathLike,
                     ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 37
                     ) -> Tuple[List[str], Dict[str, int], List[Utterance], List[Utterance], List[Utterance]]:
    r_tr, r_va, r_te = ratios
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    classes, class_to_index = find_classes_from_folders(root)

    rng = random.Random(seed)
    speakers = classes[:]  # already sorted
    rng.shuffle(speakers)

    n = len(speakers)
    n_tr = max(1, int(n * r_tr))
    n_va = max(1, int(n * r_va))
    n_tr = min(n_tr, n - 2) if n >= 3 else 1
    n_va = min(n_va, n - n_tr - 1) if n >= 3 else 0  # enough files to split

    tr_spk = set(speakers[:n_tr])
    va_spk = set(speakers[n_tr:n_tr + n_va])
    # te_spk = set(speakers[n_tr + n_va:])

    root = Path(root)
    train_utts: List[Utterance] = []
    val_utts: List[Utterance] = []
    test_utts: List[Utterance] = []

    for spk in classes:
        spk_dir = root / spk
        label = class_to_index[spk]
        files = sorted([p for p in spk_dir.rglob("*.wav") if p.is_file()])
        if spk in tr_spk:
            train_utts += [(p, label) for p in files]
        elif spk in va_spk:
            val_utts += [(p, label) for p in files]
        else:
            test_utts += [(p, label) for p in files]

    return classes, class_to_index, train_utts, val_utts, test_utts


def split_within_speaker(root: PathLike, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 37
                         ) -> Tuple[List[str], Dict[str, int], List[Utterance], List[Utterance], List[Utterance]]:
    """
    File-level split inside each speaker folder.
    Speakers are present in all splits -> valid for CrossEntropy classification.
    """
    r_tr, r_va, r_te = ratios
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    root = Path(root)
    classes, class_to_index = find_classes_from_folders(root)
    rng = random.Random(seed)

    train_utts: List[Utterance] = []
    val_utts: List[Utterance] = []
    test_utts: List[Utterance] = []

    for spk in classes:
        spk_dir = root / spk
        label = class_to_index[spk]
        files = sorted([p for p in spk_dir.rglob("*.wav") if p.is_file()])
        if len(files) < 3:
            train_utts += [(p, label) for p in files]
            continue

        rng.shuffle(files)
        n = len(files)
        n_tr = max(1, int(n * r_tr))
        n_va = max(1, int(n * r_va))
        n_va = min(n_va, n - n_tr - 1)  # leave at least 1 for test

        tr = files[:n_tr]
        va = files[n_tr:n_tr + n_va]
        te = files[n_tr + n_va:]

        train_utts += [(p, label) for p in tr]
        val_utts += [(p, label) for p in va]
        test_utts += [(p, label) for p in te]

    return classes, class_to_index, train_utts, val_utts, test_utts


def read_audio(path: PathLike, sample_rate: int) -> torch.Tensor:
    """ Load wav as mono float32 torch tensor. Shape: (T,) """
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(dim=0)  # (T, )
    return wav.to(dtype=torch.float32)


def read_audio_fast(path: Path, expected_sr: int) -> torch.Tensor:
    """
    Assumes: PCM WAV, mono, expected_sr=16k already.
    Returns: (T,) float32
    """
    wav, sr = torchaudio.load(str(path))  # (1, T) expected
    if sr != expected_sr:
        raise ValueError(f"Unexpected sample rate {sr} for {path}, expected {expected_sr}")
    if wav.dim() != 2 or wav.size(0) != 1:
        raise ValueError(f"Expected mono (1, T) for {path}, got {tuple(wav.shape)}")
    return wav.squeeze(0).to(torch.float32)


@dataclass(frozen=True)
class SegmentConfig:
    segment_seconds: Optional[float] = None
    random_crop: bool = True


class AudioDataset(Dataset):
    """
    Returns features (frames, n_feats) and label.
    Feature extractor must accept wav (T,) float tensor.
    """
    def __init__(self, utterances: Sequence[Utterance],
                 sample_rate: int, feature_extractor,
                 segment: SegmentConfig = SegmentConfig()):
        self.utterances = list(utterances)
        self.sample_rate = sample_rate
        self.fe = feature_extractor
        self.segment = segment

    def __len__(self) -> int:
        return len(self.utterances)

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        if self.segment.segment_seconds is None:
            return wav
        seg_len = int(self.segment.segment_seconds * self.sample_rate)
        if seg_len <= 0 or wav.numel() <= seg_len:
            return wav

        if self.segment.random_crop:
            start = int(torch.randint(0, wav.numel() - seg_len + 1, (1,)).item())
        else:
            start = int((wav.numel() - seg_len) // 2)
        return wav[start:start + seg_len]

    def __getitem__(self, idx: int):  # how to load one sample
        path, label = self.utterances[idx]
        wav = read_audio_fast(path, self.sample_rate)  # (T, )
        wav = self._crop(wav)                     # (T՛) maybe cropped
        feat = self.fe(wav)                       # (frames, n_feats)
        return feat, torch.tensor(label, dtype=torch.long)


def pad_trunc_collate(max_frames: int):
    """Pads/truncates features to (B, max_frames, F) and returns lengths"""
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


def create_data_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                       collate_fn=None, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn)


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
