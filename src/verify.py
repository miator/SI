import os
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

import constants as c
from dataset import (
    AudioDataset,
    pad_trunc_collate_fn,
    split_within_speaker)
from features import LogMelExtraction
from model import CNN1DNET
import metrics


BEST_PATH = c.BEST_MODEL_PATH
OUT_DIR = "emb_outputs_spk50_1h"


def _to_int64_labels(labels_any) -> np.ndarray:
    out = np.empty(len(labels_any), dtype=np.int64)
    for i, v in enumerate(labels_any):
        out[i] = int(v[0]) if isinstance(v, (tuple, list)) else int(v)
    return out


def load_model(num_classes: int, device: str):
    # emb_dim can be constant or hardcoded; keep dropout=0 for eval stability
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.0).to(device)
    state = torch.load(BEST_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_loader(utterances, fe):
    ds = AudioDataset(utterances, c.SAMPLE_RATE, fe)
    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)
    return DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)


@torch.inference_mode()
def extract_embeddings(model, loader, device: str):
    embs = []
    labels = []

    for x, y, _lengths in loader:
        x = x.to(device)
        e = model(x, return_embedding=True)
        embs.append(e.cpu())
        labels.extend(int(yi) for yi in y)

    embs = torch.cat(embs, dim=0).numpy().astype(np.float32)
    labels = _to_int64_labels(labels)
    return embs, labels


def save_embeddings(out_dir: str, split: str, embs: np.ndarray, labels: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_embeddings.npy"), embs)

    csv_path = os.path.join(out_dir, f"{split}_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "spk_id"])
        for i, spk in enumerate(labels):
            w.writerow([i, spk])


def sample_pairs(labels: np.ndarray, n_same=20000, n_diff=20000, seed=37):
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64)

    idx_by_label = defaultdict(list)
    for i, spk in enumerate(labels):
        idx_by_label.setdefault(spk, []).append(i)

    valid_spk = [k for k, v in idx_by_label.items() if len(v) >= 2]
    if not valid_spk:
        raise ValueError("No speakers with >=2 samples to form same-speaker pairs.")

    same = []
    for _ in range(n_same):
        spk = int(rng.choice(valid_spk))
        a, b = rng.choice(idx_by_label[spk], size=2, replace=False)
        same.append((a, b))

    diff = []
    all_idx = np.arange(len(labels))
    for _ in range(n_diff):
        a = int(rng.choice(all_idx))
        b = int(rng.choice(all_idx))
        while labels[b] == labels[a]:
            b = int(rng.choice(all_idx))
        diff.append((a, b))

    return np.array(same, dtype=np.int64), np.array(diff, dtype=np.int64)


def cosine_scores(embs: np.ndarray, pairs: np.ndarray):
    a = embs[pairs[:, 0]]
    b = embs[pairs[:, 1]]
    return np.sum(a * b, axis=1)  # cosine because embeddings are L2-normalized


def summarize(same_scores, diff_scores):
    print("same mean/std:", float(same_scores.mean()), float(same_scores.std()))
    print("diff mean/std:", float(diff_scores.mean()), float(diff_scores.std()))


def leakage_checks(train_utts, test_utts):
    # 1) path overlap
    tr_paths = {os.path.normcase(os.path.abspath(str(p))) for p, _ in train_utts}
    te_paths = {os.path.normcase(os.path.abspath(str(p))) for p, _ in test_utts}
    inter = sorted(tr_paths.intersection(te_paths))
    print(f"[LEAK PATH] train={len(tr_paths)} test={len(te_paths)} overlap={len(inter)}")
    if inter[:5]:
        print("  examples:", inter[:5])

    # 2) chunk-origin overlap (same original recording in both splits)
    # filename format: <origstem>__sXXXXXXX__eXXXXXXX.wav
    def origin_id(path: Path) -> str:
        stem = path.stem
        return stem.split("__s", 1)[0] if "__s" in stem else stem

    tr_origin = defaultdict(int)
    te_origin = defaultdict(int)
    for p, _ in train_utts:
        tr_origin[origin_id(Path(p))] += 1
    for p, _ in test_utts:
        te_origin[origin_id(Path(p))] += 1

    overlap_origin = sorted(set(tr_origin.keys()).intersection(te_origin.keys()))
    print(f"[LEAK ORIGIN] shared_original_recordings={len(overlap_origin)}")
    if overlap_origin[:5]:
        print("  examples:", overlap_origin[:5])

    if inter or overlap_origin:
        print("Some data sources appear in both train and test. Fix split before chunking or split by original file.")
    else:
        print("[LEAK OK] No path/origin overlap detected.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rebuild the SAME mapping as training
    classes, class_to_index, train_utts, _val_utts, test_utts = split_within_speaker(
        c.DATA_ROOT, ratios=(0.8, 0.1, 0.1), seed=37
    )
    num_classes = len(classes)

    model = load_model(num_classes, device)

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

    test_loader = make_loader(test_utts, fe)

    leakage_checks(train_utts, test_utts)

    embs, labels = extract_embeddings(model, test_loader, device)
    save_embeddings(OUT_DIR, "test", embs, labels)

    same, diff = sample_pairs(labels, n_same=20000, n_diff=20000, seed=37)
    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)
    summarize(same_scores, diff_scores)

    res = metrics.compute_roc_auc_eer(-same_scores, -diff_scores)
    print(f"ROC AUC: {res['auc']:.6f}")
    print(f"EER: {res['eer']:.6f} @ thr {res['eer_threshold']:.6f} "
          f"(FPR {res['fpr_at_eer']:.6f}, FNR {res['fnr_at_eer']:.6f})")

    if res["auc"] > 0.95 and res["eer"] < 0.10:
        print("[VERDICT] Metrics look good for a baseline.")
    else:
        print("[VERDICT] Metrics are weak; check data, split, or model.")


if __name__ == "__main__":
    main()
