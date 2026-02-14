import time
import json
from pathlib import Path
from functools import partial
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants as c
from dataset import AudioDataset, split_within_speaker, pad_trunc_collate_fn
from features import LogMelExtraction
from model import CNN1DNET
from samplers import PKSampler
from triplet import BatchHardTripletLoss
import metrics


warnings.filterwarnings("ignore", module="torchaudio")


def run_epoch_batchhard(model, loader, loss_fn, optimizer, device, train: bool, desc: str):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _len in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            y = y.to(device).view(-1)

            emb = model(x, return_embedding=True)
            loss = loss_fn(emb, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(1, total_batches)


@torch.inference_mode()
def extract_embs(model, loader, device):
    embs, labels = [], []
    for x, y, _len in tqdm(loader, desc="extract", leave=False):
        x = x.to(device)
        e = model(x, return_embedding=True).cpu()
        embs.append(e)
        labels.extend(int(v) for v in y)
    embs = torch.cat(embs, dim=0).numpy()
    labels = torch.tensor(labels, dtype=torch.long).numpy()
    return embs, labels


def sample_pairs(labels, n_same=20000, n_diff=20000, seed=37):
    import numpy as np
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    labels = labels.astype("int64")

    idx_by = defaultdict(list)
    for i, spk in enumerate(labels):
        idx_by[int(spk)].append(i)

    valid = [k for k, v in idx_by.items() if len(v) >= 2]

    same = []
    for _ in range(n_same):
        spk = int(rng.choice(valid))
        a, b = rng.choice(idx_by[spk], size=2, replace=False)
        same.append((a, b))

    diff = []
    all_idx = np.arange(len(labels))
    for _ in range(n_diff):
        a = int(rng.choice(all_idx))
        b = int(rng.choice(all_idx))
        while labels[b] == labels[a]:
            b = int(rng.choice(all_idx))
        diff.append((a, b))

    return np.array(same, dtype="int64"), np.array(diff, dtype="int64")


def cosine_scores(embs, pairs):
    a = embs[pairs[:, 0]]
    b = embs[pairs[:, 1]]
    return (a * b).sum(axis=1)


def verify(model, test_loader, device, out_dir: Path):
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    embs, labels = extract_embs(model, test_loader, device)
    np.save(out_dir / "test_embeddings.npy", embs)
    np.save(out_dir / "test_labels.npy", labels)

    same, diff = sample_pairs(labels, n_same=20000, n_diff=20000, seed=37)
    same_scores = cosine_scores(embs, same)
    diff_scores = cosine_scores(embs, diff)

    res = metrics.compute_roc_auc_eer(-same_scores, -diff_scores)
    return {
        "same_mean": float(np.mean(same_scores)),
        "same_std": float(np.std(same_scores)),
        "diff_mean": float(np.mean(diff_scores)),
        "diff_std": float(np.std(diff_scores)),
        "auc": float(res["auc"]),
        "eer": float(res["eer"]),
        "eer_thr": float(res["eer_threshold"]),
    }


def train_one(run_dir: Path, tag: str, P: int, K: int, margin: float, lr_mult: float, epochs: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes, _, train_utts, val_utts, test_utts = split_within_speaker(
        c.DATA_ROOT, ratios=(0.8, 0.1, 0.1), seed=37
    )
    num_classes = len(classes)

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

    train_ds = AudioDataset(train_utts, c.SAMPLE_RATE, fe)
    val_ds = AudioDataset(val_utts, c.SAMPLE_RATE, fe)
    test_ds = AudioDataset(test_utts, c.SAMPLE_RATE, fe)
    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_labels = [lab for _, lab in train_utts]
    sampler = PKSampler(train_labels, P=P, K=K, seed=37)

    train_loader = DataLoader(
        train_ds,
        batch_size=P * K,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True
    )

    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.3).to(device)
    loss_fn = BatchHardTripletLoss(margin=margin, normalize=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.LEARNING_RATE * lr_mult, weight_decay=1e-4)

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / f"{tag}_spk50_best.pth"

    print("\n" + "=" * 90)
    print(f"RUN: {tag}")
    print(f"  device={device} | P={P} K={K} batch={P*K} | margin={margin} | lr={c.LEARNING_RATE}*{lr_mult}={c.LEARNING_RATE*lr_mult}")
    print(f"  save_ckpt: {best_path}")
    print("=" * 90)

    best_val = float("inf")
    best_epoch = -1

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch_batchhard(model, train_loader, loss_fn, optimizer, device, True, "train")
        va = run_epoch_batchhard(model, val_loader, loss_fn, optimizer, device, False, "val")
        dt = time.perf_counter() - t0

        improved = va < best_val
        if improved:
            best_val = va
            best_epoch = ep
            torch.save(model.state_dict(), best_path)

        mark = " *BEST*" if improved else ""
        print(f"epoch {ep:02d}/{epochs} | train {tr:.4f} | val {va:.4f} | time {dt:.2f}s{mark}")

    model.load_state_dict(torch.load(best_path, map_location=device))

    emb_out_dir = run_dir / f"emb_outputs_bh_P{P}K{K}_m{margin}_e{epochs}_lr{lr_mult}"
    v = verify(model, test_loader, device, emb_out_dir)

    result = {
        "tag": tag,
        "P": P,
        "K": K,
        "margin": margin,
        "lr_mult": lr_mult,
        "epochs": epochs,
        "best_val": best_val,
        "best_epoch": best_epoch,
        **v,
    }

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("-" * 90)
    print(
        f"RESULT: AUC {v['auc']:.6f} | EER {v['eer']:.6f} | "
        f"same {v['same_mean']:.3f}±{v['same_std']:.3f} | "
        f"diff {v['diff_mean']:.3f}±{v['diff_std']:.3f} | "
        f"best_val {best_val:.4f} @ epoch {best_epoch}"
    )
    print(f"verify_outputs: {emb_out_dir}")
    print("=" * 90 + "\n")

    return v


def main():
    runs_root = Path("runs/spk50_1h")
    epochs = 10

    pk_list = [(16, 4), (12, 5), (8, 8)]
    margins = [0.30, 0.35]
    lr_mults = [1.0, 0.5]

    for P, K in pk_list:
        for m in margins:
            for lr_mult in lr_mults:
                tag = f"triplet_bh_P{P}K{K}_m{m}_e{epochs}_lr{lr_mult}"
                run_dir = runs_root / tag
                train_one(run_dir, tag, P, K, m, lr_mult, epochs)


if __name__ == "__main__":
    main()
