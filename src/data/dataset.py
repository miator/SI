from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, Optional

import torch
from torch.utils.data import Dataset
import torchaudio

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Utterance:
    """
    Storing one sample's metadata
    """
    path: Path
    speaker_id: str
    label: Optional[int] = None


def scan_split(split_root: PathLike) -> list[Utterance]:
    """
    Scan one split folder with layout:
        split_root/speaker_id/*.wav
    Returns samples with original speaker IDs preserved
    """
    split_root = Path(split_root)
    utterances: list[Utterance] = []

    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    speaker_dirs = sorted([p for p in split_root.iterdir() if p.is_dir()])

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        wav_paths = sorted(speaker_dir.glob("*.wav"))  # rglob if a speaker folder may contain nested subfolders
        for wav_path in wav_paths:
            utterances.append(Utterance(path=wav_path, speaker_id=speaker_id, label=None))

    return utterances


def build_label_map(utterances: Sequence[Utterance]) -> dict[str, int]:
    """
    Build speaker_id -> numeric label map from the provided samples.
    Use this on train split for train-speaker supervision.
    """
    speaker_ids = sorted({u.speaker_id for u in utterances})
    return {speaker_id: idx for idx, speaker_id in enumerate(speaker_ids)}


def attach_labels(
    utterances: Sequence[Utterance],
    label_map: dict[str, int]
) -> list[Utterance]:
    """
    Attach numeric labels using a provided speaker_id -> label map
    """
    labeled: list[Utterance] = []
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
    wav, sr = torchaudio.load(path)
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
    feat = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(feat)}")
    return feat.float()


def _require_numeric_label(utterance: Utterance, context: str) -> int:
    if utterance.label is None:
        raise ValueError(
            f"{context} requires numeric labels. "
            f"Use attach_labels(...) before creating the dataset."
        )
    return int(utterance.label)


def _resolve_feature_roots(feat_root) -> list[Path]:
    if isinstance(feat_root, (str, Path)):
        feat_roots = [Path(feat_root)]
    else:
        feat_roots = [Path(root) for root in feat_root]

    if not feat_roots:
        raise ValueError("feat_root must contain at least one feature root")

    return feat_roots


class AudioDataset(Dataset):
    """
    Returns:
        feat: (frames, n_feats)
        label: scalar torch.long (64 bit int)

    The feature extractor must accept a waveform tensor of shape (T,).
    Numeric labels must already be attached to all utterances.
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
        self.feature_extractor = feature_extractor
        self.waveform_transform = waveform_transform

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        utterance = self.utterances[idx]
        label = _require_numeric_label(utterance, "AudioDataset")

        waveform = read_audio_fast(utterance.path, self.sample_rate)
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)

        feat = self.feature_extractor(waveform)
        label = torch.tensor(label, dtype=torch.long)
        return feat, label


def pad_trunc_collate_fn(batch, max_frames: int):
    """
    Collate a batch of variable-length feature tensors into:
        out: (B, max_frames, n_feats)
        labels: (B,)
        lengths: (B,)

    Each feature tensor is truncated to at most max_frames and zero-padded if shorter.
    """
    feats, labels = zip(*batch)

    n_feats = feats[0].shape[1]
    batch_size = len(feats)

    out = torch.zeros((batch_size, max_frames, n_feats), dtype=feats[0].dtype)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    for i, feat in enumerate(feats):
        num_frames = feat.shape[0]
        used_frames = min(num_frames, max_frames)
        out[i, :used_frames] = feat[:used_frames]
        lengths[i] = used_frames

    return out, torch.stack(labels), lengths


class PrecomputedFeatureDataset(Dataset):
    """
    Returns:
        feat: (frames, n_feats)
        label: scalar torch.long 64bit int

    Expects .pt tensors already saved on disk.
    Stores relative paths plus labels so worker startup stays light.
    """
    def __init__(
        self,
        utterances: Sequence[Utterance],
        split_root: PathLike,
        feat_root: PathLike,
    ):
        self.split_root = Path(split_root)
        self.feat_roots = _resolve_feature_roots(feat_root)

        self.rel_feature_paths: list[Path] = []
        self.labels: list[int] = []

        for utterance in utterances:
            label = _require_numeric_label(utterance, "PrecomputedFeatureDataset")
            rel_feat_path = utterance.path.relative_to(self.split_root).with_suffix(".pt")

            for root in self.feat_roots:
                feat_path = root / rel_feat_path
                if not feat_path.exists():
                    raise FileNotFoundError(f"Missing precomputed feature file: {feat_path}")

            self.rel_feature_paths.append(rel_feat_path)
            self.labels.append(label)

        self.num_base_samples = len(self.rel_feature_paths)
        self.num_feature_roots = len(self.feat_roots)

    def __len__(self) -> int:
        return self.num_base_samples * self.num_feature_roots

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        root_idx, sample_idx = divmod(idx, self.num_base_samples)

        feat_path = self.feat_roots[root_idx] / self.rel_feature_paths[sample_idx]
        feat = load_feature_tensor(feat_path)
        label = torch.tensor(self.labels[sample_idx], dtype=torch.long)

        return feat, label
