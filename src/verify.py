import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import constants as c
from dataset import scan_split, build_label_map, read_audio_fast
from features import LogMelExtraction
from model import CNN1DNET
from metrics import compute_roc_auc_eer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class VerificationDataset(Dataset):
    """
    Dataset for embedding extraction on verification splits.
    Returns feature tensor, original speaker_id, and path.
    """
    def __init__(self, utterances, sample_rate: int, feature_extractor):
        self.utterances = list(utterances)
        self.sample_rate = sample_rate
        self.fe = feature_extractor

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int):
        u = self.utterances[idx]
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


def evaluate_split(model, split_name: str, split_root: Path, device: str,
                   same_pairs: int = 10000, diff_pairs: int = 10000, seed: int = 37):
    utterances = scan_split(split_root)
    if len(utterances) == 0:
        raise RuntimeError(f"No utterances found in split: {split_name}")

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

    ds = VerificationDataset(utterances, c.SAMPLE_RATE, fe)
    collate = partial(pad_trunc_collate_verify, max_frames=c.MAX_FRAMES)

    loader = DataLoader(
        ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        prefetch_factor=4,
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
    print(f"same cosine: {same_scores.mean().item():.3f} ± {same_scores.std().item():.3f}")
    print(f"diff cosine: {diff_scores.mean().item():.3f} ± {diff_scores.std().item():.3f}")

    out_dir = Path(c.RESULTS_DIR) / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "split": split_name,
            "embeddings": embeddings,
            "speaker_ids": speaker_ids,
            "paths": paths,
            "same_scores": same_scores,
            "diff_scores": diff_scores,
            "metrics": metrics,
        },
        out_dir / f"{split_name}_embeddings.pt"
    )


def main():
    train_utts = scan_split(c.TRAIN_ROOT)
    train_label_map = build_label_map(train_utts)
    num_classes = len(train_label_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN1DNET(
        n_feats=c.N_MELS,
        num_classes=num_classes,
        emb_dim=c.EMB_DIM,
        dropout=0.3
    ).to(device)

    ckpt_path = Path(c.BEST_MODEL_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    evaluate_split(model, "val", Path(c.VAL_ROOT), device=device)
    evaluate_split(model, "test", Path(c.TEST_ROOT), device=device)


if __name__ == "__main__":
    main()
