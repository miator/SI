import time
from pathlib import Path
from functools import partial
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import data_config as d
from src.config import experiment_config as e
from src.config import feature_config as f
from src.config import model_config as m
from src.config import train_config as t
from src.data.dataset import (
    AudioDataset,
    PrecomputedFeatureDataset,
    pad_trunc_collate_fn,
    scan_split,
    build_label_map,
    attach_labels)
from src.data.features import LogMelExtraction
from src.models.model import CNN1DNET
from src.models.triplet import BatchHardTripletLoss
from src.models.samplers import PKSampler

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def _labels_for_sampler(dataset, base_utterances) -> list[int]:
    if isinstance(dataset, PrecomputedFeatureDataset):
        return list(dataset.labels) * dataset.num_feature_roots
    return [u.label for u in base_utterances]


def run_epoch_batchhard(model, loader, criterion, optimizer, device, train: bool, desc: str):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for features, labels, _lengths in tqdm(loader, desc=desc):
            features = features.to(device)
            labels = labels.to(device)  # .view(-1)

            emb = model(features)
            loss = criterion(emb, labels)

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
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss}
    torch.save(payload, path)


def main():

    margin = t.MARGIN
    p = t.P
    k = t.K
    emb_dim = m.EMB_DIM
    epochs = t.EPOCHS
    lr = t.LEARNING_RATE
    weight_decay = t.WEIGHT_DECAY
    train_feature_mode = d.TRAIN_FEATURE_MODE

    run_name = e.EXP_NAME
    run_root = Path(e.RUNS_DIR) / run_name
    tb_dir = run_root / "tensorboard"
    ckpt_dir = run_root / "checkpoints"
    best_model_path = ckpt_dir / "best.pt"
    last_model_path = ckpt_dir / "last.pt"

    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_utts = scan_split(d.TRAIN_ROOT)
    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)

    val_utts = scan_split(d.VAL_ROOT)
    val_label_map = build_label_map(val_utts)
    val_utts = attach_labels(val_utts, val_label_map)

    print("=" * 100)
    print(f"Run name: {run_name}")
    print(
        f"margin={margin} | P={p} | K={k} | emb_dim={emb_dim} | "
        f"epochs={epochs} | lr={lr} | wd={weight_decay} | dropout={m.DROPOUT}")
    print("=" * 100)

    if d.USE_PRECOMPUTED_FEATURES:
        train_feat_roots = d.get_train_feat_roots(train_feature_mode)
        train_feat_str = ",".join(str(p) for p in train_feat_roots)
        train_ds = PrecomputedFeatureDataset(
            train_utts,
            split_root=d.TRAIN_ROOT,
            feat_root=train_feat_roots)
        val_ds = PrecomputedFeatureDataset(
            val_utts,
            split_root=d.VAL_ROOT,
            feat_root=d.VAL_FEAT_ROOT)
        print(
            f"Using precomputed features: "
            f"train={train_feat_roots} | val={d.VAL_FEAT_ROOT}")
    else:
        train_feat_str = "on_the_fly"
        fe = LogMelExtraction(
            sample_rate=f.SAMPLE_RATE,
            n_fft=f.N_FFT,
            win_length=f.WIN_LENGTH,
            hop_length=f.HOP_LENGTH,
            n_mels=f.N_MELS,
            f_min=f.FMIN,
            f_max=f.FMAX,
            eps=f.EPS)

        train_ds = AudioDataset(train_utts, f.SAMPLE_RATE, fe)
        val_ds = AudioDataset(val_utts, f.SAMPLE_RATE, fe)
        print("Using on-the-fly log-mel extraction.")

    train_labels = _labels_for_sampler(train_ds, train_utts)
    train_sampler = PKSampler(train_labels, P=p, K=k, seed=37)

    val_labels = _labels_for_sampler(val_ds, val_utts)
    val_sampler = PKSampler(val_labels, P=p, K=k, seed=37)

    collate = partial(pad_trunc_collate_fn, max_frames=f.MAX_FRAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=p * k,
        sampler=train_sampler,
        shuffle=False,
        # drop_last=True,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=p * k,
        sampler=val_sampler,
        shuffle=False,
        # drop_last=True,
        collate_fn=collate,
        num_workers=6,
        prefetch_factor=2,
        persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DNET(
        n_feats=f.N_MELS,
        emb_dim=emb_dim,
        dropout=m.DROPOUT,
    ).to(device)

    triplet_loss = BatchHardTripletLoss(margin=margin, normalize=False)  # model return normalized embeddings

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    writer = SummaryWriter(log_dir=str(tb_dir))

    writer.add_text(
        "config",
        "\n".join(
            [
                f"run_name={run_name}",
                f"sample_rate={f.SAMPLE_RATE}",
                f"n_mels={f.N_MELS}",
                f"max_frames={f.MAX_FRAMES}",
                f"emb_dim={emb_dim}",
                f"margin={margin}",
                f"P={p}",
                f"K={k}",
                f"batch_size_train={p * k}",
                f"batch_size_val={p * k}",
                f"lr={lr}",
                f"weight_decay={weight_decay}",
                f"dropout={m.DROPOUT}",
                f"train_feature_mode={train_feature_mode}",
                f"train_feat_roots={train_feat_str}",
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
            desc=f"train {epoch + 1}/{epochs}")
        va_loss = run_epoch_batchhard(
            model,
            val_loader,
            triplet_loss,
            optimizer,
            device,
            train=False,
            desc=f"val {epoch + 1}/{epochs}")

        elapsed = time.perf_counter() - start
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("loss/train", tr_loss, epoch + 1)
        writer.add_scalar("loss/val", va_loss, epoch + 1)
        writer.add_scalar("lr", current_lr, epoch + 1)
        writer.add_scalar("time/epoch_sec", elapsed, epoch + 1)

        save_checkpoint(
            last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_loss=best_val_loss)

        improved = va_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = va_loss
            bad_epochs = 0
            save_checkpoint(
                best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_loss=best_val_loss)
        else:
            bad_epochs += 1

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train triplet loss {tr_loss:.4f} | "
            f"val triplet loss {va_loss:.4f} | "
            f"lr {current_lr:.6f} | "
            f"time {elapsed:.2f}s")

        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch + 1}")
            break

    writer.close()


if __name__ == "__main__":
    main()
