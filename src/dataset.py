from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, Optional, List, Dict, Tuple

import torch
from torch.utils.data import Dataset
import torchaudio

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Utterance:
    path: Path
    speaker_id: str
    label: Optional[int] = None


def scan_split(split_root: PathLike) -> List[Utterance]:
    """
    Scan one split folder with layout:
        split_root/speaker_id/*.wav
    Returns samples with original speaker IDs preserved.
    """
    split_root = Path(split_root)
    utterances: List[Utterance] = []

    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    speaker_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        wav_paths = sorted(speaker_dir.rglob("*.wav"))
        for wav_path in wav_paths:
            utterances.append(Utterance(path=wav_path, speaker_id=speaker_id, label=None))

    return utterances


def build_label_map(utterances: Sequence[Utterance]) -> Dict[str, int]:
    """
    Build speaker_id -> numeric label map from the provided samples.
    Use this on train split for train-speaker supervision.
    """
    speaker_ids = sorted({u.speaker_id for u in utterances})
    return {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}


def attach_labels(
    utterances: Sequence[Utterance],
    label_map: Dict[str, int]
) -> List[Utterance]:
    """
    Attach numeric labels using a provided speaker_id -> label map.
    """
    labeled: List[Utterance] = []
    for u in utterances:
        if u.speaker_id not in label_map:
            raise KeyError(f"Speaker ID {u.speaker_id} not found in label map")
        labeled.append(Utterance(path=u.path, speaker_id=u.speaker_id, label=label_map[u.speaker_id]))
    return labeled


def read_audio_fast(path: Path, expected_sr: int) -> torch.Tensor:
    """
    Assumes: PCM WAV, mono, expected_sr=16k already.
    Returns: (T,) float32
    """
    wav, sr = torchaudio.load(str(path))
    if sr != expected_sr:
        raise ValueError(f"Unexpected sample rate {sr} for {path}, expected {expected_sr}")
    if wav.dim() != 2 or wav.size(0) != 1:
        raise ValueError(f"Expected mono (1, T) for {path}, got {tuple(wav.shape)}")
    return wav.squeeze(0).to(torch.float32)


def wav_path_to_feature_path(wav_path: Path, split_root: PathLike, feat_root: PathLike) -> Path:
    split_root = Path(split_root)
    feat_root = Path(feat_root)

    rel = wav_path.relative_to(split_root)
    return feat_root / rel.with_suffix(".pt")


def save_feature_tensor(path: Path, feat: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat.cpu(), path)


def load_feature_tensor(path: Path) -> torch.Tensor:
    feat = torch.load(path, map_location="cpu")
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(feat)}")
    return feat.float()


class AudioDataset(Dataset):
    """
    Returns features (frames, n_feats) and label.
    Feature extractor must accept wav (T,) float tensor.
    This dataset expects numeric labels to already be attached.
    """
    def __init__(
        self,
        utterances: Sequence[Utterance],
        sample_rate: int,
        feature_extractor,
        waveform_transform=None,
    ):
        self.utterances = list(utterances)
        self.sample_rate = sample_rate
        self.fe = feature_extractor
        self.waveform_transform = waveform_transform

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int):
        sample = self.utterances[idx]

        if sample.label is None:
            raise ValueError(
                f"Sample at index {idx} has no numeric label. "
                f"Use attach_labels(...) before creating AudioDataset."
            )

        wav = read_audio_fast(sample.path, self.sample_rate)
        if self.waveform_transform is not None:
            wav = self.waveform_transform(wav)
        feat = self.fe(wav)
        return feat, torch.tensor(sample.label, dtype=torch.long)


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


class PrecomputedFeatureDataset(Dataset):
    """
    Returns precomputed features (frames, n_feats) and label.
    Expects .pt tensors already saved on disk.
    Optimized: all file paths and labels are prepared once in __init__.
    """
    def __init__(self, utterances: Sequence[Utterance], split_root: PathLike, feat_root: PathLike):
        self.split_root = Path(split_root)

        if isinstance(feat_root, (str, Path)):
            feat_roots = [Path(feat_root)]
        else:
            feat_roots = [Path(root) for root in feat_root]

        if not feat_roots:
            raise ValueError("feat_root must contain at least one feature root")

        self.samples: List[Tuple[Path, torch.Tensor]] = []

        for root in feat_roots:
            for u in utterances:
                if u.label is None:
                    raise ValueError(
                        "All utterances must already have numeric labels. "
                        "Use attach_labels(...) before creating PrecomputedFeatureDataset."
                    )

                feat_path = wav_path_to_feature_path(u.path, self.split_root, root)
                if not feat_path.exists():
                    raise FileNotFoundError(f"Missing precomputed feature file: {feat_path}")

                label_tensor = torch.tensor(u.label, dtype=torch.long)
                self.samples.append((feat_path, label_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_path, label = self.samples[idx]
        feat = load_feature_tensor(feat_path)
        return feat, label
