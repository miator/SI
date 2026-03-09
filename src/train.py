import time
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants as c
from dataset import AudioDataset, pad_trunc_collate_fn, scan_split, build_label_map, attach_labels
from features import LogMelExtraction
from model import CNN1DNET
from triplet import BatchHardTripletLoss
from samplers import PKSampler

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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
    train_utts = scan_split(c.TRAIN_ROOT)
    val_utts = scan_split(c.VAL_ROOT)
    test_utts = scan_split(c.TEST_ROOT)

    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)

    # local labels only for temporary val triplet-loss monitoring
    val_label_map = build_label_map(val_utts)
    val_utts = attach_labels(val_utts, val_label_map)

    num_classes = len(train_label_map)
    print("Train speakers:", len(train_label_map))
    print("Train files:", len(train_utts))
    print("Val speakers:", len(val_label_map))
    print("Val files:", len(val_utts))
    print("Test speakers:", len({u.speaker_id for u in test_utts}))
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

    P, K = 12, 5
    train_labels = [u.label for u in train_utts]  # is a list of Utterance

    sampler = PKSampler(train_labels, P=P, K=K, seed=37)

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_loader = DataLoader(
        train_ds, batch_size=P*K, sampler=sampler, shuffle=False, drop_last=True, collate_fn=collate,
        num_workers=8, prefetch_factor=4, persistent_workers=True
    )

    x, y, _ = next(iter(train_loader))
    print("unique speakers in batch:", len(torch.unique(y)), "batch size:", len(y))

    val_loader = DataLoader(
        val_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
        num_workers=8, prefetch_factor=4, persistent_workers=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.3).to(device)
    triplet_loss = BatchHardTripletLoss(margin=0.35, normalize=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.LEARNING_RATE, weight_decay=1e-4)

    patience = 5
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = Path(c.BEST_MODEL_PATH)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- TRAIN LOOP ----
    for epoch in range(c.EPOCHS):
        start = time.perf_counter()

        tr_loss = run_epoch_batchhard(model, train_loader, triplet_loss, optimizer, device, train=True, desc="train")
        va_loss = run_epoch_batchhard(model, val_loader, triplet_loss, optimizer, device, train=False, desc="val")

        elapsed = time.perf_counter() - start
        if va_loss < best_val_loss - min_delta:
            best_val_loss = va_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch + 1}")
                break

        print(
            f"epoch {epoch + 1}/{c.EPOCHS} | "
            f"train triplet loss {tr_loss:.4f} | "
            f"val triplet loss {va_loss:.4f} | "
            f"time {elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
