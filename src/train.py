import json
import time
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants as c
from dataset import (AudioDataset, split_within_speaker, pad_trunc_collate_fn)
from features import LogMelExtraction
from model import CNN1DNET
from triplet import BatchHardTripletLoss
from samplers import PKSampler

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


def make_random_triplets(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build explicit (a_idx, p_idx, n_idx) triplets from labels within a batch.
    One triplet per anchor when possible.
    """
    device = labels.device
    B = labels.size(0)

    a_list, p_list, n_list = [], [], []
    for i in range(B):
        yi = labels[i]
        pos = torch.where(labels == yi)[0]
        pos = pos[pos != i]
        neg = torch.where(labels != yi)[0]

        if pos.numel() == 0 or neg.numel() == 0:
            continue

        p = pos[torch.randint(0, pos.numel(), (1,), device=device).item()]
        n = neg[torch.randint(0, neg.numel(), (1,), device=device).item()]

        a_list.append(i)
        p_list.append(int(p))
        n_list.append(int(n))

    if len(a_list) == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    return (
        torch.tensor(a_list, dtype=torch.long, device=device),
        torch.tensor(p_list, dtype=torch.long, device=device),
        torch.tensor(n_list, dtype=torch.long, device=device),
    )


def run_epoch_batchhard(model, loader, triplet_loss, optimizer, device, train: bool, desc: str):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _lengths in tqdm(loader, desc=desc):
            x = x.to(device)
            y = y.to(device).view(-1)   # ensure shape (B,)

            emb = model(x, return_embedding=True)  # (B, D)
            loss = triplet_loss(emb, y)            # scalar (can be 0.0)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No batches were produced by the loader (check sampler/batch_size).")
    return total_loss / total_batches


def main():
    classes, class_to_index, train_utts, val_utts, test_utts = split_within_speaker(
        c.DATA_ROOT, ratios=(0.8, 0.1, 0.1), seed=37
    )

    num_classes = len(classes)
    print("Speakers:", num_classes)
    print("Train files:", len(train_utts))
    print("Val files:", len(val_utts))
    print("Test files:", len(test_utts))

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

    train_ds = AudioDataset(train_utts, c.SAMPLE_RATE, fe)
    val_ds = AudioDataset(val_utts, c.SAMPLE_RATE, fe)
    test_ds = AudioDataset(test_utts, c.SAMPLE_RATE, fe)

    P, K = 12, 5
    train_labels = [lab for _, lab in train_utts]

    sampler = PKSampler(train_labels, P=P, K=K, seed=37)

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_loader = DataLoader(
        train_ds, batch_size=P*K, sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate,
        num_workers=4, prefetch_factor=4, persistent_workers=True
    )

    x, y, _ = next(iter(train_loader))
    print("unique speakers in batch:", len(torch.unique(y)), "batch size:", len(y))

    val_loader = DataLoader(
        val_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
        num_workers=4, prefetch_factor=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
        num_workers=4, prefetch_factor=4, persistent_workers=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.3).to(device)

    triplet_loss = BatchHardTripletLoss(margin=0.3, normalize=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.LEARNING_RATE, weight_decay=1e-4)
    history: list[dict] = []

    patience = 5
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0

    margin = float(getattr(triplet_loss, "margin", 0.0))
    wd = float(optimizer.param_groups[0].get("weight_decay", 0.0))
    run_dir = Path(c.RUNS_DIR) / f"bh_P{P}K{K}_m{margin}_e{c.EPOCHS}_lr{c.LEARNING_RATE}_wd{wd}_emb{c.EMB_DIM}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "P": P, "K": K, "margin": margin, "epochs": c.EPOCHS,
            "lr": c.LEARNING_RATE, "weight_decay": wd, "emb_dim": c.EMB_DIM,
            "dropout": 0.3, "seed": 37,
        }, f, indent=2)

    best_path = run_dir / "best.pt"
    train_result_path = run_dir / "train_result.json"

    best_epoch = -1

    # ---- TRAIN LOOP ----
    for epoch in range(c.EPOCHS):
        start = time.perf_counter()

        tr_loss = run_epoch_batchhard(model, train_loader, triplet_loss, optimizer, device, train=True, desc="train")
        va_loss = run_epoch_batchhard(model, val_loader, triplet_loss, optimizer, device, train=False, desc="val")

        elapsed = time.perf_counter() - start
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "time_sec": float(elapsed),
            "lr": float(optimizer.param_groups[0].get("lr", c.LEARNING_RATE)),
            "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        })

        improved = va_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = va_loss
            best_epoch = epoch + 1
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch + 1}")
                break

        mark = " *BEST*" if improved else ""
        print(
            f"epoch {epoch + 1:02d}/{c.EPOCHS} | "
            f"train {tr_loss:.4f} | val {va_loss:.4f} | "
            f"time {elapsed:.2f}s{mark}"
        )

    # ---- TEST ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    t0 = time.perf_counter()
    te_loss = run_epoch_batchhard(model, test_loader, triplet_loss, optimizer, device, train=False, desc="test")
    t_test = time.perf_counter() - t0
    print(f"test triplet loss {te_loss:.4f} time {t_test:.2f}s")

    result = {
        "task": "train_triplet",
        "device": device,
        "data_root": str(c.DATA_ROOT),
        "split_seed": 37,
        "model": "CNN1DNET",
        "emb_dim": int(c.EMB_DIM),
        "features": "LogMel",
        "P": int(P),
        "K": int(K),
        "margin": float(getattr(triplet_loss, "margin", 0.0)),
        "epochs_planned": int(c.EPOCHS),
        "epochs_ran": int(len(history)),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_ckpt_path": str(best_path),
        "test_loss": float(te_loss),
        "test_time_sec": float(t_test),
        "history": history,
    }

    with open(train_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {train_result_path}")


if __name__ == "__main__":
    main()
