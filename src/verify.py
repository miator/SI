import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

import constants as c
from dataset import (
    AudioDataset,
    load_csv,
    build_utterances,
    pad_trunc_collate_fn,
    split_train_folder_by_speaker,
)
from features import LogMelExtraction
from model import CNN1DNET


BEST_PATH = "cnn_speakerid_best.pth"   # same as train.py
OUT_DIR = "emb_outputs"


def _to_int64_labels(labels_any) -> np.ndarray:
    out = np.empty(len(labels_any), dtype=np.int64)

    for i, v in enumerate(labels_any):
        # tuple or list → take first element
        if isinstance(v, (tuple, list)):
            out[i] = int(v[0])
        else:
            out[i] = int(v)

    return out


def load_model(num_classes: int, device: str):
    # emb_dim can be constant or hardcoded; keep dropout=0 for eval stability
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=192, dropout=0.0).to(device)
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


def sample_pairs(labels: np.ndarray, n_same=2000, n_diff=2000, seed=37):
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.int64)

    idx_by_label = {}
    for i, spk in enumerate(labels):
        idx_by_label.setdefault(spk, []).append(i)

    valid_spk = [k for k, v in idx_by_label.items() if len(v) >= 2]

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
    return np.sum(a * b, axis=1)  # dot == cosine because embeddings are normalized


def summarize(same_scores, diff_scores):
    print("same mean/std:", float(same_scores.mean()), float(same_scores.std()))
    print("diff mean/std:", float(diff_scores.mean()), float(diff_scores.std()))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rebuild the SAME mapping as training
    classes, class_to_index, _train_utts, _val_utts, _ = split_train_folder_by_speaker(
        c.TRAIN_WAV_ROOT, ratios=(0.9, 0.1, 0.0), seed=37
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
    )

    test_rows = load_csv(c.TEST_CSV)
    test_utts = build_utterances(
        test_rows, c.TEST_WAV_ROOT, class_to_index, path_col="file_path", speaker_col="speaker"
    )
    test_loader = make_loader(test_utts, fe)

    embs, labels = extract_embeddings(model, test_loader, device)
    save_embeddings(OUT_DIR, "test", embs, labels)

    same, diff = sample_pairs(labels, n_same=2000, n_diff=2000, seed=37)
    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)
    summarize(same_scores, diff_scores)


if __name__ == "__main__":
    main()
