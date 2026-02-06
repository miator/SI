# import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from functools import partial

import constants as c
from dataset import (AudioDataset, split_train_folder_by_speaker,
                     build_utterances_from_train_folder, pad_trunc_collate_fn)
from features import MFCCExtraction
from model import FeedForwardNet

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


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

    classes, class_to_index, train_utts, val_utts, _ = split_train_folder_by_speaker(
        c.TRAIN_WAV_ROOT, ratios=(0.9, 0.1, 0.0), seed=37)
    missing = []
    for p in Path(c.TEST_WAV_ROOT).iterdir():
        if p.is_dir() and p.name not in class_to_index:
            missing.append(p.name)
    if missing:
        raise ValueError(f"Test contains speaker not in train set: {missing}")

    num_classes = len(classes)
    test_utts = build_utterances_from_train_folder(c.TEST_WAV_ROOT, class_to_index)

    fe = MFCCExtraction(
        sample_rate=c.SAMPLE_RATE,
        n_mfcc=c.N_MFCC,
        n_mels=c.N_MELS,
        n_fft=c.N_FFT,
        win_length=c.WIN_LENGTH,
        hop_length=c.HOP_LENGTH,
        f_min=c.FMIN,
        f_max=c.FMAX,
    )

    train_ds = AudioDataset(train_utts, c.SAMPLE_RATE, fe)
    val_ds = AudioDataset(val_utts, c.SAMPLE_RATE, fe)
    test_ds = AudioDataset(test_utts, c.SAMPLE_RATE, fe)

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)
    train_loader = DataLoader(train_ds, batch_size=c.BATCH_SIZE, shuffle=True, collate_fn=collate,
                              num_workers=4, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=c.BATCH_SIZE, shuffle=False, collate_fn=collate,
                             num_workers=4, prefetch_factor=4, persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeedForwardNet(n_mfcc=c.N_MFCC, max_frames=c.MAX_FRAMES, num_classes=num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.LEARNING_RATE,
        weight_decay=1e-4)

    patience = 5
    min_delta = 1e-4
    best_val_loss = float("inf")
    # best_val = -1.0
    bad_epochs = 0
    best_path = "ff_speakerid_best.pth"

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
            f"epoch {epoch + 1}/{c.EPOCHS} "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} "
            f"time {elapsed:.2f}s")

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
