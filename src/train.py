import time
from pathlib import Path
from functools import partial
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import constants as c
from dataset import (
    AudioDataset,
    PrecomputedFeatureDataset,
    pad_trunc_collate_fn,
    scan_split,
    build_label_map,
    attach_labels)
from features import LogMelExtraction
from model import CNN1DNET
from triplet import BatchHardTripletLoss
from samplers import PKSampler

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

torch.set_num_threads(10)
torch.set_num_interop_threads(2)


def run_epoch_batchhard(model, loader, triplet_loss, optimizer, device, train: bool, desc: str):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _lengths in tqdm(loader, desc=desc):
            x = x.to(device)
            y = y.to(device).view(-1)

            emb = model(x, return_embedding=True)
            loss = triplet_loss(emb, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("No batches were produced by the loader (check sampler/batch_size).")

    return total_loss / total_batches


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main():
    train_utts = scan_split(c.TRAIN_ROOT)
    val_utts = scan_split(c.VAL_ROOT)
    test_utts = scan_split(c.TEST_ROOT)

    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)

    val_label_map = build_label_map(val_utts)
    val_utts = attach_labels(val_utts, val_label_map)

    num_classes = len(train_label_map)
    print("Train speakers:", len(train_label_map))
    print("Train files:", len(train_utts))
    print("Val speakers:", len(val_label_map))
    print("Val files:", len(val_utts))
    print("Test speakers:", len({u.speaker_id for u in test_utts}))
    print("Test files:", len(test_utts))

    if c.USE_PRECOMPUTED_FEATURES:
        train_ds = PrecomputedFeatureDataset(
            train_utts,
            split_root=c.TRAIN_ROOT,
            feat_root=c.TRAIN_FEAT_ROOT
        )
        val_ds = PrecomputedFeatureDataset(
            val_utts,
            split_root=c.VAL_ROOT,
            feat_root=c.VAL_FEAT_ROOT
        )
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

        train_ds = AudioDataset(train_utts, c.SAMPLE_RATE, fe)
        val_ds = AudioDataset(val_utts, c.SAMPLE_RATE, fe)
        print("Using on-the-fly log-mel extraction.")

    train_labels = [u.label for u in train_utts]
    sampler = PKSampler(train_labels, P=c.P, K=c.K, seed=37)

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=c.P * c.K,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        num_workers=7,
        prefetch_factor=1,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=9,
        prefetch_factor=1,
        persistent_workers=False,
    )

    x0, y0, _ = next(iter(train_loader))
    print("batch:", x0.shape, x0.dtype, y0.shape, y0.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(
        n_feats=c.N_MELS,
        num_classes=num_classes,
        emb_dim=c.EMB_DIM,
        dropout=0.3
    ).to(device)
    triplet_loss = BatchHardTripletLoss(margin=c.TRIPLET_MARGIN, normalize=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c.LEARNING_RATE,
        weight_decay=c.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=c.LR_FACTOR,
        patience=c.LR_PATIENCE,
        min_lr=c.MIN_LR,
    )

    Path(c.TB_DIR).mkdir(parents=True, exist_ok=True)
    Path(c.CKPT_DIR).mkdir(parents=True, exist_ok=True)

    run_name = (
        f"cnn1d_emb{c.EMB_DIM}_m{c.TRIPLET_MARGIN}"
        f"_P{c.P}K{c.K}_lr{c.LEARNING_RATE}"
    )
    writer = SummaryWriter(log_dir=str(Path(c.TB_DIR) / run_name))

    writer.add_text("config", "\n".join([
        f"sample_rate={c.SAMPLE_RATE}",
        f"n_mels={c.N_MELS}",
        f"max_frames={c.MAX_FRAMES}",
        f"emb_dim={c.EMB_DIM}",
        f"margin={c.TRIPLET_MARGIN}",
        f"P={c.P}",
        f"K={c.K}",
        f"batch_size_train={c.P * c.K}",
        f"batch_size_val={c.BATCH_SIZE}",
        f"lr={c.LEARNING_RATE}",
        f"weight_decay={c.WEIGHT_DECAY}",
        f"lr_scheduler={c.LR_SCHEDULER}",
        f"lr_factor={c.LR_FACTOR}",
        f"lr_patience={c.LR_PATIENCE}",
        f"min_lr={c.MIN_LR}",
        f"epochs={c.EPOCHS}",
    ]), 0)

    try:
        writer.add_graph(model, x0.to(device))
    except Exception as e:
        print("TensorBoard graph add failed:", e)

    patience = 7
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0

    # ---- TRAIN LOOP ----
    for epoch in range(c.EPOCHS):
        start = time.perf_counter()

        tr_loss = run_epoch_batchhard(
            model, train_loader, triplet_loss, optimizer, device,
            train=True, desc=f"train {epoch+1}/{c.EPOCHS}"
        )
        va_loss = run_epoch_batchhard(
            model, val_loader, triplet_loss, optimizer, device,
            train=False, desc=f"val {epoch+1}/{c.EPOCHS}"
        )

        scheduler.step(va_loss)

        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("loss/train", tr_loss, epoch + 1)
        writer.add_scalar("loss/val", va_loss, epoch + 1)
        writer.add_scalar("lr", current_lr, epoch + 1)
        writer.add_scalar("time/epoch_sec", elapsed, epoch + 1)

        save_checkpoint(
            c.LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
        )

        improved = va_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = va_loss
            bad_epochs = 0
            save_checkpoint(
                c.BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
            )
        else:
            bad_epochs += 1

        print(
            f"epoch {epoch + 1}/{c.EPOCHS} | "
            f"train triplet loss {tr_loss:.4f} | "
            f"val triplet loss {va_loss:.4f} | "
            f"lr {current_lr:.6f} | "
            f"time {elapsed:.2f}s"
        )

        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch + 1}")
            break

    writer.close()


if __name__ == "__main__":
    main()
