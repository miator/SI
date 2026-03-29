import argparse
import time
from pathlib import Path
from functools import partial
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import constants as c
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

# torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN1D triplet model with optional experiment overrides."
    )
    parser.add_argument("--run-name", type=str, default=None, help="Custom run folder name.")
    parser.add_argument("--margin", type=float, default=None, help="Triplet margin override.")
    parser.add_argument("--p", type=int, default=None, help="PK sampler P override.")
    parser.add_argument("--k", type=int, default=None, help="PK sampler K override.")
    parser.add_argument("--emb-dim", type=int, default=None, help="Embedding dimension override.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        choices=["none", "plateau", "cosine"],
        help="Learning-rate scheduler override.",
    )
    parser.add_argument("--wd", type=float, default=None, help="Weight decay override.")
    parser.add_argument(
        "--train-feature-mode",
        type=str,
        default=None,
        choices=["clean", "noise", "clean+noise", "clean+white", "clean+musan+white", "white", "musan+white"],
        help="Which precomputed train feature set(s) to use.",
    )
    return parser.parse_args()


def make_run_name(
    margin: float,
    p: int,
    k: int,
    emb_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    scheduler_name: str,
) -> str:
    margin_str = str(margin).replace(".", "")
    lr_str = str(lr).replace(".", "").replace("-", "")
    wd_str = str(weight_decay).replace(".", "").replace("-", "")
    dropout_str = str(dropout).replace(".", "")
    return f"cnn1d_emb{emb_dim}_m{margin_str}_P{p}K{k}_lr{lr_str}_wd{wd_str}_drop{dropout_str}_{scheduler_name}"


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


def build_scheduler(optimizer, scheduler_name: str, epochs: int):
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=c.LR_FACTOR,
            patience=c.LR_PATIENCE,
            min_lr=c.MIN_LR,
        )
    if scheduler_name == "cosine":
        t_max = c.COSINE_T_MAX if c.COSINE_T_MAX is not None else epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=c.COSINE_ETA_MIN,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, scheduler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def main():
    args = parse_args()

    margin = c.TRIPLET_MARGIN if args.margin is None else args.margin
    p = c.P if args.p is None else args.p
    k = c.K if args.k is None else args.k
    emb_dim = c.EMB_DIM if args.emb_dim is None else args.emb_dim
    epochs = c.EPOCHS if args.epochs is None else args.epochs
    lr = c.LEARNING_RATE if args.lr is None else args.lr
    weight_decay = c.WEIGHT_DECAY if args.wd is None else args.wd
    lr_scheduler_name = c.LR_SCHEDULER if args.lr_scheduler is None else args.lr_scheduler
    train_feature_mode = c.TRAIN_FEATURE_MODE if args.train_feature_mode is None else args.train_feature_mode

    run_name = args.run_name or make_run_name(
        margin=margin,
        p=p,
        k=k,
        emb_dim=emb_dim,
        lr=lr,
        weight_decay=weight_decay,
        dropout=c.DROPOUT,
        scheduler_name=lr_scheduler_name,
    )

    run_root = Path(c.RUNS_DIR) / run_name
    tb_dir = run_root / "tensorboard"
    ckpt_dir = run_root / "checkpoints"
    best_model_path = ckpt_dir / "best.pt"
    last_model_path = ckpt_dir / "last.pt"

    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_utts = scan_split(c.TRAIN_ROOT)
    val_utts = scan_split(c.VAL_ROOT)

    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)

    val_label_map = build_label_map(val_utts)
    val_utts = attach_labels(val_utts, val_label_map)

    num_classes = len(train_label_map)
    print("=" * 100)
    print(f"Run name: {run_name}")
    print(
        f"margin={margin} | P={p} | K={k} | emb_dim={emb_dim} | "
        f"epochs={epochs} | lr={lr} | wd={weight_decay} | dropout={c.DROPOUT} | "
        f"lr_scheduler={lr_scheduler_name}"
    )
    print("=" * 100)

    train_feat_roots = None
    if c.USE_PRECOMPUTED_FEATURES:
        train_feat_roots = c.get_train_feat_roots(train_feature_mode)
        train_ds = PrecomputedFeatureDataset(
            train_utts,
            split_root=c.TRAIN_ROOT,
            feat_root=train_feat_roots,
        )
        val_ds = PrecomputedFeatureDataset(
            val_utts,
            split_root=c.VAL_ROOT,
            feat_root=c.VAL_FEAT_ROOT,
        )
        print(
            f"Using precomputed features: "
            f"train={train_feat_roots} | val={c.VAL_FEAT_ROOT}"
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
            eps=c.EPS,
        )

        train_ds = AudioDataset(train_utts, c.SAMPLE_RATE, fe)
        val_ds = AudioDataset(val_utts, c.SAMPLE_RATE, fe)
        print("Using on-the-fly log-mel extraction.")

    train_labels = train_ds.labels
    sampler = PKSampler(train_labels, P=p, K=k, seed=37)

    collate = partial(pad_trunc_collate_fn, max_frames=c.MAX_FRAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=p * k,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=c.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(
        n_feats=c.N_MELS,
        num_classes=num_classes,
        emb_dim=emb_dim,
        dropout=c.DROPOUT,
    ).to(device)

    triplet_loss = BatchHardTripletLoss(margin=margin, normalize=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(optimizer, scheduler_name=lr_scheduler_name, epochs=epochs)

    writer = SummaryWriter(log_dir=str(tb_dir))

    writer.add_text(
        "config",
        "\n".join(
            [
                f"run_name={run_name}",
                f"sample_rate={c.SAMPLE_RATE}",
                f"n_mels={c.N_MELS}",
                f"max_frames={c.MAX_FRAMES}",
                f"emb_dim={emb_dim}",
                f"margin={margin}",
                f"P={p}",
                f"K={k}",
                f"batch_size_train={p * k}",
                f"batch_size_val={c.BATCH_SIZE}",
                f"lr={lr}",
                f"weight_decay={weight_decay}",
                f"dropout={c.DROPOUT}",
                f"lr_scheduler={lr_scheduler_name}",
                f"train_feature_mode={train_feature_mode}",
                f"train_feat_roots={','.join(str(path) for path in train_feat_roots) if c.USE_PRECOMPUTED_FEATURES else 'on_the_fly'}",
                f"lr_factor={None}",
                f"lr_patience={None}",
                f"min_lr={c.COSINE_ETA_MIN if lr_scheduler_name == 'cosine' else None}",
                f"cosine_t_max={c.COSINE_T_MAX if lr_scheduler_name == 'cosine' else None}",
                f"epochs={epochs}",
                f"run_root={run_root}",
                f"best_model_path={best_model_path}",
                f"last_model_path={last_model_path}",
            ]
        ),
        0,
    )

    patience = 5
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0

    # ---- TRAIN LOOP ----
    for epoch in range(epochs):
        start = time.perf_counter()

        tr_loss = run_epoch_batchhard(
            model,
            train_loader,
            triplet_loss,
            optimizer,
            device,
            train=True,
            desc=f"train {epoch + 1}/{epochs}",
        )
        va_loss = run_epoch_batchhard(
            model,
            val_loader,
            triplet_loss,
            optimizer,
            device,
            train=False,
            desc=f"val {epoch + 1}/{epochs}",
        )

        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if lr_scheduler_name == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()

        writer.add_scalar("loss/train", tr_loss, epoch + 1)
        writer.add_scalar("loss/val", va_loss, epoch + 1)
        writer.add_scalar("lr", current_lr, epoch + 1)
        writer.add_scalar("time/epoch_sec", elapsed, epoch + 1)

        save_checkpoint(
            last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
            scheduler=scheduler,
        )

        improved = va_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = va_loss
            bad_epochs = 0
            save_checkpoint(
                best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                scheduler=scheduler,
            )
        else:
            bad_epochs += 1

        print(
            f"epoch {epoch + 1}/{epochs} | "
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
