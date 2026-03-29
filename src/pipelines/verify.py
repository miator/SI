import argparse
import csv
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import constants as c
from dataset import (
    scan_split,
    build_label_map,
    read_audio_fast,
    wav_path_to_feature_path,
    load_feature_tensor,
)
from features import LogMelExtraction
from model import CNN1DNET
from metrics import compute_roc_auc_eer

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
    "diff_std",
    "margin",
    "P",
    "K",
    "embedding_dim",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify a trained experiment and log split metrics."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder name under runs/, or a full path to the experiment folder.",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "val_noisy_snr15", "test", "test_noisy_snr15", "test_noisy_snr10"],
        help="Evaluation splits to score.",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Which checkpoint to evaluate.",
    )
    return parser.parse_args()


def parse_margin_token(token: str):
    if not token:
        return None
    if "." in token:
        return float(token)
    if token.startswith("0") and len(token) > 1:
        return float(f"0.{token[1:]}")
    return float(token)


def parse_experiment_hparams(experiment_name: str) -> dict:
    match = re.search(
        r"emb(?P<embedding_dim>\d+).*?_m(?P<margin>[0-9.]+)_P(?P<P>\d+)K(?P<K>\d+)",
        experiment_name,
    )
    if match is None:
        return {
            "margin": None,
            "P": None,
            "K": None,
            "embedding_dim": None,
        }

    return {
        "margin": parse_margin_token(match.group("margin")),
        "P": int(match.group("P")),
        "K": int(match.group("K")),
        "embedding_dim": int(match.group("embedding_dim")),
    }


def resolve_run_root(experiment_arg: str) -> Path:
    run_root = Path(experiment_arg)
    if not run_root.is_absolute():
        run_root = Path(c.RUNS_DIR) / experiment_arg
    run_root = run_root.resolve()

    if not run_root.exists():
        raise FileNotFoundError(f"Experiment folder not found: {run_root}")
    return run_root


def resolve_checkpoint_path(run_root: Path, checkpoint_type: str) -> Path:
    if checkpoint_type == "best":
        candidates = [
            run_root / "checkpoints" / "best.pt",
            run_root / "checkpoints" / "best_val_loss.pt",
        ]
    elif checkpoint_type == "last":
        candidates = [
            run_root / "checkpoints" / "last.pt",
            run_root / "checkpoints" / "last_epoch.pt",
        ]
    else:
        raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type}")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find a {checkpoint_type} checkpoint. Looked for:\n"
        + "\n".join(str(path) for path in candidates)
    )


def infer_emb_dim_from_checkpoint(state_dict: dict, fallback: int) -> int:
    emb_weight = state_dict.get("emb.1.weight")
    if emb_weight is not None:
        return int(emb_weight.shape[0])

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None:
        return int(classifier_weight.shape[1])

    return fallback


def append_metrics_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_metrics_rows(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_rows(csv_path: Path):
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
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
    write_metrics_rows(csv_path, merged_rows)


def build_metrics_row(
    experiment_name: str,
    checkpoint_type: str,
    split_name: str,
    metrics: dict,
    same_scores: torch.Tensor,
    diff_scores: torch.Tensor,
    hparams: dict,
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
        "diff_std": float(diff_scores.std().item()),
        "margin": hparams["margin"],
        "P": hparams["P"],
        "K": hparams["K"],
        "embedding_dim": hparams["embedding_dim"],
    }


class VerificationDataset(Dataset):
    """
    Dataset for embedding extraction on verification splits.
    Returns feature tensor, original speaker_id, and path.
    Can read either raw WAVs or precomputed .pt log-mel tensors.
    """
    def __init__(
        self,
        utterances,
        sample_rate=None,
        feature_extractor=None,
        split_root=None,
        feat_root=None,
    ):
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

    def __getitem__(self, idx: int):
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


def pad_trunc_collate_verify(batch, max_frames: int):
    feats, speaker_ids, paths = zip(*batch)
    f = feats[0].shape[1]
    out = torch.zeros((len(feats), max_frames, f), dtype=feats[0].dtype)
    lengths = torch.zeros((len(feats),), dtype=torch.long)

    for i, x in enumerate(feats):
        t = x.shape[0]
        t2 = min(t, max_frames)
        out[i, :t2] = x[:t2]
        lengths[i] = t2

    return out, list(speaker_ids), list(paths), lengths


@torch.inference_mode()
def extract_embeddings(model, loader, device: str):
    model.eval()

    all_embeddings = []
    all_speaker_ids = []
    all_paths = []

    for x, speaker_ids, paths, _lengths in tqdm(loader, desc="extract embeddings"):
        x = x.to(device)
        emb = model(x, return_embedding=True)
        emb = F.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu())
        all_speaker_ids.extend(speaker_ids)
        all_paths.extend(paths)

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, all_speaker_ids, all_paths


def sample_same_pairs(indices_by_speaker, max_pairs: int, seed: int = 37):
    rng = random.Random(seed)
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


def sample_diff_pairs(indices_by_speaker, max_pairs: int, seed: int = 37):
    rng = random.Random(seed)
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


def cosine_scores_from_pairs(embeddings: torch.Tensor, pairs):
    if not pairs:
        return torch.empty(0, dtype=torch.float32)

    idx1 = torch.tensor([i for i, _ in pairs], dtype=torch.long)
    idx2 = torch.tensor([j for _, j in pairs], dtype=torch.long)

    e1 = embeddings[idx1]
    e2 = embeddings[idx2]

    scores = (e1 * e2).sum(dim=1)
    return scores.cpu()


def evaluate_split(
    model,
    split_name: str,
    checkpoint_type: str,
    split_root: Path,
    feat_root: Path,
    device: str,
    output_dir: Path,
    experiment_name: str,
    hparams: dict,
    same_pairs: int = 10000,
    diff_pairs: int = 10000,
    seed: int = 37,
):
    utterances = scan_split(split_root)
    if len(utterances) == 0:
        raise RuntimeError(f"No utterances found in split: {split_name}")

    if c.USE_PRECOMPUTED_FEATURES:
        ds = VerificationDataset(
            utterances,
            split_root=split_root,
            feat_root=feat_root,
        )
        print(f"Using precomputed log-mel features for {split_name}.")
    else:
        fe = LogMelExtraction(
            sample_rate=c.SAMPLE_RATE,
            n_fft=c.N_FFT,
            win_length=c.WIN_LENGTH,
            hop_length=c.HOP_LENGTH,
            n_mels=c.N_MELS,
            f_min=c.FMIN,
            f_max=c.FMAX,
            eps=c.EPS
        )
        ds = VerificationDataset(
            utterances,
            sample_rate=c.SAMPLE_RATE,
            feature_extractor=fe,
        )
        print(f"Using on-the-fly log-mel extraction for {split_name}.")

    collate = partial(pad_trunc_collate_verify, max_frames=c.MAX_FRAMES)

    loader = DataLoader(
        ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True
    )

    embeddings, speaker_ids, paths = extract_embeddings(model, loader, device)

    indices_by_speaker = defaultdict(list)
    for idx, spk in enumerate(speaker_ids):
        indices_by_speaker[spk].append(idx)

    same = sample_same_pairs(indices_by_speaker, max_pairs=same_pairs, seed=seed)
    diff = sample_diff_pairs(indices_by_speaker, max_pairs=diff_pairs, seed=seed + 1)

    same_scores = cosine_scores_from_pairs(embeddings, same)
    diff_scores = cosine_scores_from_pairs(embeddings, diff)

    if same_scores.numel() == 0 or diff_scores.numel() == 0:
        raise RuntimeError(
            f"Could not build enough verification pairs for split={split_name}. "
            f"same={same_scores.numel()} diff={diff_scores.numel()}"
        )

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
        diff_scores=diff_scores,
        hparams=hparams,
    )

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
            "summary_row": row,
        },
        output_dir / f"{split_name}_{checkpoint_type}_embeddings.pt"
    )

    return row


def main():
    args = parse_args()

    run_root = resolve_run_root(args.experiment)
    experiment_name = run_root.name
    verify_dir = run_root / "results" / "verify"
    per_run_csv = verify_dir / "metrics_summary.csv"
    global_csv = Path(c.RUNS_DIR) / "verify_summary.csv"

    hparams = parse_experiment_hparams(experiment_name)

    train_utts = scan_split(c.TRAIN_ROOT)
    train_label_map = build_label_map(train_utts)
    num_classes = len(train_label_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_checkpoint_path(run_root, args.checkpoint_type)
    ckpt = torch.load(ckpt_path, map_location=device)

    emb_dim = infer_emb_dim_from_checkpoint(
        ckpt["model_state_dict"],
        hparams["embedding_dim"] or c.EMB_DIM,
    )
    if hparams["embedding_dim"] is None:
        hparams["embedding_dim"] = emb_dim

    model = CNN1DNET(
        n_feats=c.N_MELS,
        num_classes=num_classes,
        emb_dim=emb_dim,
        dropout=0.3
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    rows = []
    eval_definitions = c.get_eval_split_definitions()

    for split_name in args.eval_splits:
        if split_name not in eval_definitions:
            raise ValueError(
                f"Unsupported split '{split_name}'. "
                f"Supported: {', '.join(eval_definitions.keys())}"
            )

        split_def = eval_definitions[split_name]
        row = evaluate_split(
            model,
            split_name,
            args.checkpoint_type,
            Path(split_def["wav_root"]),
            Path(split_def["feat_root"]),
            device=device,
            output_dir=verify_dir,
            experiment_name=experiment_name,
            hparams=hparams,
        )
        rows.append(row)

    upsert_metrics_rows(per_run_csv, rows)
    upsert_metrics_rows(global_csv, rows)

    print(f"\nPer-run summary saved to: {per_run_csv}")
    print(f"Global summary updated at: {global_csv}")


if __name__ == "__main__":
    main()
