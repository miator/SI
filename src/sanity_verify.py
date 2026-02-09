import os
import hashlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import constants as c
import metrics

from dataset import (
    load_csv,
    build_utterances,
    split_train_folder_by_speaker,
    read_audio,
)
from features import LogMelExtraction
from model import CNN1DNET
from verify import (
    make_loader,
    extract_embeddings,
    sample_pairs,
    cosine_scores,
)


BEST_PATH = c.BEST_MODEL_PATH if hasattr(c, "BEST_MODEL_PATH") else "cnn_speakerid_best.pth"


def load_model(num_classes: int, device: str):
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.0).to(device)
    state = torch.load(BEST_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def abs_norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


def path_overlap_check(train_utts, test_utts):
    tr = {abs_norm_path(p) for (p, _y) in train_utts}
    te = {abs_norm_path(p) for (p, _y) in test_utts}
    inter = sorted(tr.intersection(te))
    print(f"[PATH OVERLAP] train={len(tr)} test={len(te)} overlap={len(inter)}")
    if inter[:5]:
        print("  examples:", inter[:5])
    return inter


def pcm16_hash(wav: torch.Tensor) -> str:
    """
    Stable hashing: convert float waveform to int16 PCM then hash bytes.
    """
    x = wav.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x * 32767.0).round().to(torch.int16).numpy()
    h = hashlib.sha256(x.tobytes()).hexdigest()
    return h


def audio_hash_overlap_check(train_utts, test_utts):
    """
    Hash each waveform after read_audio() (resample+mono) and check duplicates across splits.
    """
    tr_hash = {}
    for p, _y in train_utts:
        wav = read_audio(p, c.SAMPLE_RATE)
        tr_hash[pcm16_hash(wav)] = abs_norm_path(p)

    overlap = []
    for p, _y in test_utts:
        wav = read_audio(p, c.SAMPLE_RATE)
        hh = pcm16_hash(wav)
        if hh in tr_hash:
            overlap.append((tr_hash[hh], abs_norm_path(p)))

    print(f"[AUDIO HASH OVERLAP] duplicates={len(overlap)}")
    if overlap[:5]:
        print("  examples:")
        for a, b in overlap[:5]:
            print("   ", a, "<->", b)
    return overlap


def pairs_sanity(labels, same_pairs, diff_pairs):
    ok_same = np.all(labels[same_pairs[:, 0]] == labels[same_pairs[:, 1]])
    ok_diff = np.all(labels[diff_pairs[:, 0]] != labels[diff_pairs[:, 1]])
    print(f"[PAIR SANITY] same_ok={bool(ok_same)} diff_ok={bool(ok_diff)} "
          f"(same={len(same_pairs)}, diff={len(diff_pairs)})")
    return ok_same and ok_diff


def compute_metrics_with_current_convention(same_scores, diff_scores):
    """
    Your current pipeline used: metrics.compute_roc_auc_eer(-same, -diff)
    Keep it consistent here.
    """
    res = metrics.compute_roc_auc_eer(-same_scores, -diff_scores)
    return res


def sklearn_auc_crosscheck(same_scores, diff_scores):
    """
    Independent ROC AUC check using sklearn.
    Assumes higher score => same-speaker (positive class).
    """
    same = np.asarray(same_scores, dtype=np.float64).reshape(-1)
    diff = np.asarray(diff_scores, dtype=np.float64).reshape(-1)

    scores = np.concatenate([same, diff], axis=0)
    labels = np.concatenate([
        np.ones(len(same), dtype=np.int64),
        np.zeros(len(diff), dtype=np.int64)
    ], axis=0)

    auc = float(roc_auc_score(labels, scores))
    print(f"[SKLEARN AUC] {auc:.6f}")
    return auc


def label_shuffle_control(embs, labels, n_same=20000, n_diff=20000, seed=123):
    rng = np.random.default_rng(seed)
    labels_shuf = labels.copy()
    rng.shuffle(labels_shuf)

    same, diff = sample_pairs(labels_shuf, n_same=n_same, n_diff=n_diff, seed=seed)
    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)

    res = compute_metrics_with_current_convention(same_scores, diff_scores)
    print(f"[LABEL SHUFFLE] AUC={res['auc']:.6f} EER={res['eer']:.6f} (expected ~0.5, ~0.5)")
    return res


def random_embedding_control(num_items, emb_dim, labels, n_same=20000, n_diff=20000, seed=456):
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((num_items, emb_dim)).astype(np.float32)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    same, diff = sample_pairs(labels, n_same=n_same, n_diff=n_diff, seed=seed)
    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)

    res = compute_metrics_with_current_convention(same_scores, diff_scores)
    print(f"[RANDOM EMB] AUC={res['auc']:.6f} EER={res['eer']:.6f} (expected ~0.5, ~0.5)")
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rebuild mapping exactly like training
    classes, class_to_index, train_utts, _val_utts, _ = split_train_folder_by_speaker(
        c.TRAIN_WAV_ROOT, ratios=(0.9, 0.1, 0.0), seed=37
    )
    num_classes = len(classes)

    # Build test utterances from CSV (your current setup)
    test_rows = load_csv(c.TEST_CSV)
    test_utts = build_utterances(test_rows, c.TEST_WAV_ROOT, class_to_index,
                                 path_col="file_path", speaker_col="speaker")

    # Leakage checks (paths + audio content)
    path_overlap_check(train_utts, test_utts)
    audio_hash_overlap_check(train_utts, test_utts)

    # Feature extractor + loader
    fe = LogMelExtraction(
        sample_rate=c.SAMPLE_RATE,
        n_fft=c.N_FFT,
        win_length=c.WIN_LENGTH,
        hop_length=c.HOP_LENGTH,
        n_mels=c.N_MELS,
        f_min=c.FMIN,
        f_max=c.FMAX,
        eps=c.EPS,
    )

    model = load_model(num_classes, device)
    test_loader = make_loader(test_utts, fe)

    embs, labels = extract_embeddings(model, test_loader, device)

    # Main evaluation at larger scale (reduces “lucky sampling”)
    same, diff = sample_pairs(labels, n_same=50000, n_diff=50000, seed=37)
    pairs_sanity(labels, same, diff)

    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)

    print("[SCORE STATS] same mean/std:", float(same_scores.mean()), float(same_scores.std()))
    print("[SCORE STATS] diff mean/std:", float(diff_scores.mean()), float(diff_scores.std()))

    res = compute_metrics_with_current_convention(same_scores, diff_scores)
    print(f"[MAIN] ROC AUC: {res['auc']:.6f}")
    print(f"[MAIN] EER: {res['eer']:.6f} @ thr {res['eer_threshold']:.6f} "
          f"(FPR {res['fpr_at_eer']:.6f}, FNR {res['fnr_at_eer']:.6f})")

    # Independent AUC cross-check (optional)
    sklearn_auc_crosscheck(same_scores, diff_scores)

    # Controls: must collapse to random performance
    label_shuffle_control(embs, labels, n_same=20000, n_diff=20000, seed=123)
    random_embedding_control(len(labels), embs.shape[1], labels, n_same=20000, n_diff=20000, seed=456)


if __name__ == "__main__":
    main()
