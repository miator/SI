import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Tuple

import constants as c
from dataset import (AudioDataset, split_within_speaker, pad_trunc_collate_fn)
from features import LogMelExtraction
from model import CNN1DNET

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

Utterance = Tuple[Path, int]


def load_chunks_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8"):
        return list(csv.DictReader)


def build_class_mapping(rows: List[dict], speaker_col: str = "speaker") -> Dict[str, int]:
    speakers = sorted({r[speaker_col] for r in rows})
    return {spk: i for i, spk in enumerate(speakers)}


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool, desc: str):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _lengths in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs

    return total_loss / n, total_acc / n


def main():
    classes, class_to_index, train_utts, val_utts, test_utts = split_within_speaker(
        c.DATA_ROOT, ratios=(0.8, 0.1, 0.1), seed=37)

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

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_loader = DataLoader(train_ds, batch_size=c.BATCH_SIZE, shuffle=True, collate_fn=collate,
                              num_workers=4, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
                            num_workers=4, prefetch_factor=4, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
                             num_workers=4, prefetch_factor=4, persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(n_feats=c.N_MELS, num_classes=num_classes, emb_dim=c.EMB_DIM, dropout=0.3).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.LEARNING_RATE,
        weight_decay=1e-4)

    patience = 5
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = Path(c.BEST_MODEL_PATH)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- TRAIN LOOP ----
    for epoch in range(c.EPOCHS):
        start = time.perf_counter()

        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, desc="train")
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False, desc="val")

        elapsed = time.perf_counter() - start

        if va_loss < best_val_loss - min_delta:
            best_val_loss = va_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch +1}")
                break

        print(
            f"epoch {epoch + 1}/{c.EPOCHS} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"time {elapsed:.2f}s"
        )

    # ---- TEST ----
    model.load_state_dict((torch.load(best_path, map_location=device)))
    t0 = time.perf_counter()
    te_loss, te_acc = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False, desc="test")
    t_test = time.perf_counter() - t0
    print(
        f"test loss {te_loss:.4f} "
        f"acc {te_acc:.4f} "
        f"time{t_test:.2f}s")


if __name__ == "__main__":
    main()
