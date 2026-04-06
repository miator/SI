import csv
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Callable, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.dataset import (
    scan_split,
    read_audio_fast,
    wav_path_to_feature_path,
    load_feature_tensor)
from src.config import data_config as d
from src.config import experiment_config as e
from src.config import feature_config as f
from src.config import model_config as m
from src.config import train_config as t

from src.data.features import LogMelExtraction
from src.models.model import CNN1DNET
from src.metrics import compute_roc_auc_eer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


SUMMARY_FIELDNAMES = [
    "experiment",
    "checkpoint_type",
    "split",
    "auc",
    "eer",
    "same_mean",
    "same_std",
    "diff_mean",
    "diff_std"]


def resolve_checkpoint_path(run_root: Path, checkpoint_type: str) -> Path:
    if checkpoint_type == "best":
        path = run_root / "checkpoints" / "best.pt"
    elif checkpoint_type == "last":
        path = run_root / "checkpoints" / "last.pt"
    else:
        raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def load_metrics_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            raw_row: dict[str, str]
            row = dict(raw_row)
            if "checkpoint_type" not in row or not row.get("checkpoint_type"):
                row["checkpoint_type"] = "best"
            rows.append(row)
        return rows


def upsert_metrics_rows(csv_path: Path, rows: Iterable[dict]):
    existing_rows = load_metrics_rows(csv_path)
    rows_by_key = {
        (row["experiment"], row["checkpoint_type"], row["split"]): row
        for row in existing_rows
    }
    for row in rows:
        rows_by_key[(row["experiment"], row["checkpoint_type"], row["split"])] = row

    merged_rows = list(rows_by_key.values())
    merged_rows.sort(key=lambda row: (row["experiment"], row["checkpoint_type"], row["split"]))

    sanitized_rows = [
        {field: row.get(field, "") for field in SUMMARY_FIELDNAMES}
        for row in merged_rows]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(sanitized_rows)


def build_metrics_row(
    experiment_name: str,
    checkpoint_type: str,
    split_name: str,
    metrics: dict,
    same_scores: torch.Tensor,
    diff_scores: torch.Tensor,
) -> dict:
    return {
        "experiment": experiment_name,
        "checkpoint_type": checkpoint_type,
        "split": split_name,
        "auc": float(metrics["auc"]),
        "eer": float(metrics["eer"]),
        "same_mean": float(same_scores.mean().item()),
        "same_std": float(same_scores.std().item()),
        "diff_mean": float(diff_scores.mean().item()),
        "diff_std": float(diff_scores.std().item())}


class VerificationDataset(Dataset):
    """
    Dataset for embedding extraction on verification splits.
    Returns feature tensor, original speaker_id, and path.
    Can read either raw WAVs or precomputed .pt log-mel tensors.
    """
    def __init__(
            self,
            utterances,
            sample_rate: Optional[int] = None,
            feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            split_root: Optional[Union[Path, str]] = None,
            feat_root: Optional[Union[Path, str]] = None):

        self.utterances = list(utterances)
        self.sample_rate = sample_rate
        self.fe = feature_extractor
        self.split_root = Path(split_root) if split_root is not None else None
        self.feat_root = Path(feat_root) if feat_root is not None else None

        using_precomputed = self.split_root is not None or self.feat_root is not None
        if using_precomputed and (self.split_root is None or self.feat_root is None):
            raise ValueError("Both split_root and feat_root must be provided for precomputed features.")
        if not using_precomputed and (self.sample_rate is None or self.fe is None):
            raise ValueError("sample_rate and feature_extractor must be provided for on-the-fly features.")

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        u = self.utterances[idx]

        if self.feat_root is not None:
            feat_path = wav_path_to_feature_path(u.path, self.split_root, self.feat_root)
            if not feat_path.exists():
                raise FileNotFoundError(f"Missing precomputed feature file: {feat_path}")
            feat = load_feature_tensor(feat_path)
        else:
            wav = read_audio_fast(u.path, self.sample_rate)
            feat = self.fe(wav)

        return feat, u.speaker_id, str(u.path)


def pad_trunc_collate_verify(
    batch: list[tuple[torch.Tensor, str, str]],
    max_frames: int,
) -> tuple[torch.Tensor, list[str], list[str], torch.Tensor]:
    feats, speaker_ids, paths = zip(*batch)
    n_feats = feats[0].shape[1]
    out = torch.zeros((len(feats), max_frames, n_feats), dtype=feats[0].dtype)
    lengths = torch.zeros((len(feats),), dtype=torch.long)

    for i, feat in enumerate(feats):
        num_frames = feat.shape[0]
        used_frames = min(num_frames, max_frames)
        out[i, :used_frames] = feat[:used_frames]
        lengths[i] = used_frames

    return out, list(speaker_ids), list(paths), lengths


@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    model.eval()

    all_embeddings: list[torch.Tensor] = []
    all_speaker_ids: list[str] = []
    all_paths: list[str] = []

    for features, speaker_ids, paths, _lengths in tqdm(loader, desc="extract embeddings"):
        features = features.to(device, non_blocking=True)
        emb = model(features)

        all_embeddings.append(emb.cpu())
        all_speaker_ids.extend(speaker_ids)
        all_paths.extend(paths)

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_speaker_ids, all_paths


def sample_pairs(
    indices_by_speaker: dict[str, list[int]],
    max_pairs: int,
    pair_kind: str,
    seed: int = 37,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)

    if pair_kind == "same":
        speakers = [spk for spk, idxs in indices_by_speaker.items() if len(idxs) >= 2]
        if not speakers:
            return []

        pairs = set()
        attempts = 0
        max_attempts = max_pairs * 20

        while len(pairs) < max_pairs and attempts < max_attempts:
            spk = rng.choice(speakers)
            i, j = rng.sample(indices_by_speaker[spk], 2)
            if i > j:
                i, j = j, i
            pairs.add((i, j))
            attempts += 1
        return list(pairs)

    if pair_kind == "diff":
        speakers = list(indices_by_speaker.keys())
        if len(speakers) < 2:
            return []

        pairs = set()
        attempts = 0
        max_attempts = max_pairs * 20

        while len(pairs) < max_pairs and attempts < max_attempts:
            spk1, spk2 = rng.sample(speakers, 2)
            i = rng.choice(indices_by_speaker[spk1])
            j = rng.choice(indices_by_speaker[spk2])
            if i > j:
                i, j = j, i
            pairs.add((i, j))
            attempts += 1
        return list(pairs)
    raise ValueError(f"Unsupported pair_kind: {pair_kind}")


def cosine_scores_from_pairs(
    embeddings: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    if not pairs:
        return torch.empty(0, dtype=torch.float32)

    idx1 = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    idx2 = torch.tensor([j for _, j in pairs], dtype=torch.long)

    e1 = embeddings[idx1]
    e2 = embeddings[idx2]

    scores = F.cosine_similarity(e1, e2, dim=1)
    return scores.cpu()


def evaluate_split(
    model: torch.nn.Module,
    split_name: str,
    checkpoint_type: str,
    split_root: Path,
    feat_root: Path,
    device: str,
    output_dir: Path,
    experiment_name: str,
    same_pairs: int = 10000,
    diff_pairs: int = 10000,
    seed: int = 37,
    save_artifacts: bool = False
):

    utterances = scan_split(split_root)
    if len(utterances) == 0:
        raise RuntimeError(f"No utterances found in split: {split_name}")

    if d.USE_PRECOMPUTED_FEATURES:
        ds = VerificationDataset(
            utterances,
            split_root=split_root,
            feat_root=feat_root)
        print(f"Using precomputed log-mel features for {split_name}.")
    else:
        fe = LogMelExtraction(
            sample_rate=f.SAMPLE_RATE,
            n_fft=f.N_FFT,
            win_length=f.WIN_LENGTH,
            hop_length=f.HOP_LENGTH,
            n_mels=f.N_MELS,
            f_min=f.FMIN,
            f_max=f.FMAX,
            eps=f.EPS)
        ds = VerificationDataset(
            utterances,
            sample_rate=f.SAMPLE_RATE,
            feature_extractor=fe)
        print(f"Using on-the-fly log-mel extraction for {split_name}.")

    collate = partial(pad_trunc_collate_verify, max_frames=f.MAX_FRAMES)

    loader = DataLoader(
        ds,
        batch_size=t.P * t.K,
        shuffle=False,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available())

    embeddings, speaker_ids, paths = extract_embeddings(model, loader, device)

    indices_by_speaker: dict[str, list[int]] = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        indices_by_speaker[spk].append(idx)

    same = sample_pairs(indices_by_speaker, max_pairs=same_pairs, pair_kind="same", seed=seed)
    diff = sample_pairs(indices_by_speaker, max_pairs=diff_pairs, pair_kind="diff", seed=seed + 1)

    same_scores = cosine_scores_from_pairs(embeddings, same)
    diff_scores = cosine_scores_from_pairs(embeddings, diff)

    if same_scores.numel() == 0 or diff_scores.numel() == 0:
        raise RuntimeError(
            f"Could not build enough verification pairs for split={split_name}. "
            f"same={same_scores.numel()} diff={diff_scores.numel()}")

    metrics = compute_roc_auc_eer(same_scores.numpy(), diff_scores.numpy())
    auc = metrics["auc"]
    eer = metrics["eer"]

    print(f"\n[{split_name.upper()}]")
    print(f"speakers: {len(indices_by_speaker)}")
    print(f"files: {len(utterances)}")
    print(f"same pairs: {len(same)}")
    print(f"diff pairs: {len(diff)}")
    print(f"AUC: {auc:.6f}")
    print(f"EER: {eer:.6f}")
    print(f"same cosine: {same_scores.mean().item():.3f} | {same_scores.std().item():.3f}")
    print(f"diff cosine: {diff_scores.mean().item():.3f} | {diff_scores.std().item():.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    row = build_metrics_row(
        experiment_name=experiment_name,
        checkpoint_type=checkpoint_type,
        split_name=split_name,
        metrics=metrics,
        same_scores=same_scores,
        diff_scores=diff_scores)

    if save_artifacts:
        torch.save(
            {
                "experiment": experiment_name,
                "checkpoint_type": checkpoint_type,
                "split": split_name,
                "embeddings": embeddings,
                "speaker_ids": speaker_ids,
                "paths": paths,
                "same_scores": same_scores,
                "diff_scores": diff_scores,
                "metrics": metrics,
                "summary_row": row},
            output_dir / f"{split_name}_{checkpoint_type}_embeddings.pt")
    return row


def main():
    emb_dim = m.EMB_DIM
    save_artifacts = False

    checkpoint_type = "best"  # best / last
    run_name = e.EXP_NAME
    run_root = Path(e.RUNS_DIR) / run_name

    experiment_name = run_root.name
    verify_dir = run_root / "results" / "verify"
    per_run_csv = verify_dir / "metrics_summary.csv"
    global_summary_name = "verify_summary_new.csv"
    global_csv = Path(e.RUNS_DIR) / global_summary_name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_checkpoint_path(run_root, checkpoint_type)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CNN1DNET(
        n_feats=f.N_MELS,
        emb_dim=emb_dim,
        dropout=0.3
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    rows = []
    eval_definitions = d.get_eval_split_definitions()
    selected_splits = {"val", "test"}

    for split_name, split_def in eval_definitions.items():
        if split_name not in selected_splits:
            continue
        row = evaluate_split(
            model=model,
            split_name=split_name,
            checkpoint_type=checkpoint_type,
            split_root=Path(split_def["wav_root"]),
            feat_root=Path(split_def["feat_root"]),
            device=device,
            output_dir=verify_dir,
            experiment_name=experiment_name,
            save_artifacts=save_artifacts)

        rows.append(row)

    upsert_metrics_rows(per_run_csv, rows)
    upsert_metrics_rows(global_csv, rows)

    print(f"\nPer-run summary saved to: {per_run_csv}")
    print(f"Global summary updated at: {global_csv}")


if __name__ == "__main__":
    main()
