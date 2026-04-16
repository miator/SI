import time
from pathlib import Path
from functools import partial
import warnings
import json
import random

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
    RandomChoicePrecomputedFeatureDataset,
    ResolvedPrecomputedFeatureDataset,
    ResolvedRandomChoicePrecomputedFeatureDataset,
    pad_trunc_collate_fn,
    scan_split,
    build_label_map,
    attach_labels)
from src.data.features import LogMelExtraction
from src.models.model import build_embedding_model
from src.models.triplet import BatchHardTripletLoss
from src.models.samplers import PKSampler

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

torch.set_num_threads(4)
# torch.set_num_interop_threads(1)


class CollapsedTrainingError(RuntimeError):
    def __init__(
        self,
        *,
        run_name: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        margin: float,
    ) -> None:
        super().__init__("early abort: loss collapsed near margin")
        self.run_name = run_name
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.margin = margin


def _emit_collapse_status(exc: CollapsedTrainingError) -> None:
    print(json.dumps(
            {"status": "collapsed",
             "run_name": exc.run_name,
             "epoch": exc.epoch,
             "train_loss": exc.train_loss,
             "val_loss": exc.val_loss,
             "margin": exc.margin}
        )
    )


def _labels_for_sampler(dataset, base_utterances) -> list[int]:
    if isinstance(dataset, RandomChoicePrecomputedFeatureDataset):
        return list(dataset.labels)
    if isinstance(dataset, ResolvedPrecomputedFeatureDataset):
        return list(dataset.labels)
    if isinstance(dataset, ResolvedRandomChoicePrecomputedFeatureDataset):
        return list(dataset.labels)
    if isinstance(dataset, PrecomputedFeatureDataset):
        return list(dataset.labels) * dataset.num_feature_roots
    return [u.label for u in base_utterances]


def _load_train_utterances() -> list:
    utterances_by_set: dict[str, list] = {}
    for split_def in d.get_train_split_definitions():
        utterances_by_set[str(split_def["set_name"])] = scan_split(Path(split_def["wav_root"]))

    data_mode = getattr(d, "TRAIN_DATA_MODE", "clean_only")
    probabilities = getattr(d, "TRAIN_DATA_PROBABILITIES", None) or {}
    use_other_as_augmentation = getattr(d, "USE_OTHER_AS_AUGMENTATION", False)

    if data_mode == "clean_only":
        train_utterances = []
        for utterances in utterances_by_set.values():
            train_utterances.extend(utterances)
        return train_utterances

    if data_mode == "clean+other_prob":
        if not use_other_as_augmentation:
            train_utterances = []
            for utterances in utterances_by_set.values():
                train_utterances.extend(utterances)
            return train_utterances

        clean_utterances = []
        other_utterances = []
        for set_name, utterances in utterances_by_set.items():
            if set_name == "other500":
                other_utterances.extend(utterances)
            else:
                clean_utterances.extend(utterances)

        clean_probability = float(probabilities.get("clean", 0.0))
        other_probability = float(probabilities.get("other", 0.0))
        if clean_probability <= 0 and other_probability <= 0:
            raise ValueError("clean+other_prob requires a positive 'clean' or 'other' probability")
        if clean_probability <= 0:
            return other_utterances
        if other_probability <= 0 or not other_utterances:
            return clean_utterances

        total_scale = min(
            len(clean_utterances) / clean_probability,
            len(other_utterances) / other_probability,
        )
        target_clean = max(1, int(total_scale * clean_probability))
        target_other = max(1, int(total_scale * other_probability))
        rng = random.Random(37)

        sampled_clean = (
            rng.sample(clean_utterances, target_clean)
            if target_clean < len(clean_utterances)
            else list(clean_utterances)
        )
        sampled_other = (
            rng.sample(other_utterances, target_other)
            if target_other < len(other_utterances)
            else list(other_utterances)
        )
        return sampled_clean + sampled_other

    if data_mode == "clean+esc+white":
        train_utterances = []
        for set_name, utterances in utterances_by_set.items():
            if set_name == "other500" and use_other_as_augmentation:
                continue
            train_utterances.extend(utterances)
        return train_utterances

    raise ValueError(f"Unsupported TRAIN_DATA_MODE: {data_mode}")


def _resolve_train_feature_roots(
    train_feature_mode: str,
    train_feature_probabilities: dict[str, float] | None,
):
    feature_keys = d.get_train_feature_root_keys(train_feature_mode)
    grouped_feature_roots = {
        key: d.get_train_feat_roots_for_key(key)
        for key in feature_keys
    }

    if d.is_probabilistic_train_feature_mode(train_feature_mode):
        mode_probabilities = d.get_train_feature_probabilities(
            train_feature_mode,
            train_feature_probabilities,
        )
        named_feature_roots: list[tuple[str, Path]] = []
        named_probabilities: dict[str, float] = {}
        for key, roots in grouped_feature_roots.items():
            if not roots:
                continue
            per_root_probability = mode_probabilities.get(key, 0.0) / len(roots)
            for idx, root in enumerate(roots):
                root_name = key if len(roots) == 1 else f"{key}_{idx + 1}"
                named_feature_roots.append((root_name, root))
                named_probabilities[root_name] = per_root_probability
        return grouped_feature_roots, named_feature_roots, named_probabilities

    flat_feature_roots = [
        (root.name or f"{key}_root", root)
        for key, roots in grouped_feature_roots.items()
        for root in roots
    ]
    return grouped_feature_roots, flat_feature_roots, None


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
    train_data_mode = getattr(d, "TRAIN_DATA_MODE", "clean_only")
    train_set_names = getattr(d, "TRAIN_SET_NAMES", ("clean100",))
    train_data_probabilities = getattr(d, "TRAIN_DATA_PROBABILITIES", None)
    model_dropout = m.CONFORMER_DROPOUT if m.MODEL_NAME.lower() == "conformer" else m.DROPOUT

    run_name = e.EXP_NAME
    run_root = Path(e.RUNS_DIR) / run_name
    tb_dir = run_root / "tensorboard"
    ckpt_dir = run_root / "checkpoints"
    best_model_path = ckpt_dir / "best.pt"
    last_model_path = ckpt_dir / "last.pt"
    collapse_patience = getattr(t, "COLLAPSE_PATIENCE", 2)
    collapse_tolerance = 5e-4
    collapse_require_train_and_val = True

    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_utts = _load_train_utterances()
    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)

    val_utts = scan_split(d.VAL_ROOT)
    val_label_map = build_label_map(val_utts)
    val_utts = attach_labels(val_utts, val_label_map)

    print("=" * 100)
    print(f"Run name: {run_name}")
    print(
        f"margin={margin} | P={p} | K={k} | emb_dim={emb_dim} | "
        f"epochs={epochs} | lr={lr} | wd={weight_decay} | dropout={model_dropout} | "
        f"train_sets={train_set_names} | data_mode={train_data_mode}")
    print("=" * 100)

    train_feature_probabilities = None

    if d.USE_PRECOMPUTED_FEATURES:
        train_feature_roots, resolved_train_feature_roots, train_feature_probabilities = (
            _resolve_train_feature_roots(
                train_feature_mode,
                d.TRAIN_FEATURE_PROBABILITIES,
            )
        )
        train_feat_str = ",".join(
            f"{key}:{list(roots)}"
            for key, roots in train_feature_roots.items()
        )
        if d.is_probabilistic_train_feature_mode(train_feature_mode):
            train_feature_paths_by_source = [
                {
                    key: d.resolve_train_feature_path(utterance.path, key)
                    for key in d.get_train_feature_root_keys(train_feature_mode)
                }
                for utterance in train_utts
            ]
            train_ds = ResolvedRandomChoicePrecomputedFeatureDataset(
                train_feature_paths_by_source,
                [utterance.label for utterance in train_utts],
                probabilities=d.get_train_feature_probabilities(
                    train_feature_mode,
                    d.TRAIN_FEATURE_PROBABILITIES,
                ),
            )
        else:
            if d.TRAIN_FEATURE_PROBABILITIES is not None:
                raise ValueError(
                    "TRAIN_FEATURE_PROBABILITIES can only be used with probabilistic "
                    "train feature modes such as 'clean|noise' or 'clean|noise|white'."
                )
            if train_feature_mode == "clean":
                train_feat_paths = [
                    d.resolve_train_clean_feature_path(utterance.path)
                    for utterance in train_utts
                ]
                train_ds = ResolvedPrecomputedFeatureDataset(
                    train_feat_paths,
                    [utterance.label for utterance in train_utts],
                )
            else:
                train_ds = PrecomputedFeatureDataset(
                    train_utts,
                    split_root=d.TRAIN_ROOT,
                    feat_root=[root for _name, root in resolved_train_feature_roots])
        val_ds = PrecomputedFeatureDataset(
            val_utts,
            split_root=d.VAL_ROOT,
            feat_root=d.VAL_FEAT_ROOT)
        print(
            f"Using precomputed features: "
            f"train={train_feature_roots} | "
            f"train_probs={train_feature_probabilities} | "
            f"val={d.VAL_FEAT_ROOT}")
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
    model = build_embedding_model(
        m.MODEL_NAME,
        n_feats=f.N_MELS,
        emb_dim=emb_dim,
        dropout=m.DROPOUT,
        conformer_d_model=m.CONFORMER_D_MODEL,
        conformer_dropout=m.CONFORMER_DROPOUT,
        conformer_num_heads=m.CONFORMER_NUM_HEADS,
        conformer_ff_mult=m.CONFORMER_FF_MULT,
        conformer_conv_kernel_size=m.CONFORMER_CONV_KERNEL_SIZE,
        conformer_num_blocks=m.CONFORMER_NUM_BLOCKS,
    ).to(device)

    triplet_loss = BatchHardTripletLoss(margin=margin, normalize=False)  # model return normalized embeddings

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    resume_checkpoint_path = getattr(t, "RESUME_CHECKPOINT_PATH", None)
    start_epoch = 0

    if resume_checkpoint_path is not None:
        resume_checkpoint_path = Path(resume_checkpoint_path)
        if resume_checkpoint_path.exists():
            ckpt = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed from checkpoint: {resume_checkpoint_path} | start_epoch={start_epoch}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")

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
                f"model_name={m.MODEL_NAME}",
                f"margin={margin}",
                f"P={p}",
                f"K={k}",
                f"batch_size_train={p * k}",
                f"batch_size_val={p * k}",
                f"lr={lr}",
                f"weight_decay={weight_decay}",
                f"dropout={model_dropout}",
                f"conformer_d_model={m.CONFORMER_D_MODEL}",
                f"conformer_dropout={m.CONFORMER_DROPOUT}",
                f"conformer_num_heads={m.CONFORMER_NUM_HEADS}",
                f"conformer_ff_mult={m.CONFORMER_FF_MULT}",
                f"conformer_conv_kernel_size={m.CONFORMER_CONV_KERNEL_SIZE}",
                f"conformer_num_blocks={m.CONFORMER_NUM_BLOCKS}",
                f"train_feature_mode={train_feature_mode}",
                f"train_data_mode={train_data_mode}",
                f"train_set_names={train_set_names}",
                f"train_data_probabilities={train_data_probabilities}",
                f"train_feat_roots={train_feat_str}",
                f"train_feature_probabilities={train_feature_probabilities}",
                f"epochs={epochs}",
                f"run_root={run_root}",
                f"best_model_path={best_model_path}",
                f"last_model_path={last_model_path}",
            ]
        ),
        0,
    )

    patience = 8
    min_delta = 1e-4
    best_val_loss = float("inf")
    bad_epochs = 0
    collapsed_epochs = 0

    # ---- TRAIN LOOP ----
    try:
        best_val_loss = locals().get("best_val_loss", float("inf"))
        for epoch in range(start_epoch, epochs):
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

            train_near_margin = abs(tr_loss - margin) <= collapse_tolerance
            val_near_margin = abs(va_loss - margin) <= collapse_tolerance
            is_collapsed_epoch = (
                train_near_margin and val_near_margin
                if collapse_require_train_and_val
                else val_near_margin)
            collapsed_epochs = collapsed_epochs + 1 if is_collapsed_epoch else 0

            if collapsed_epochs >= collapse_patience:
                print("early abort: loss collapsed near margin")
                raise CollapsedTrainingError(
                    run_name=run_name,
                    epoch=epoch + 1,
                    train_loss=tr_loss,
                    val_loss=va_loss,
                    margin=margin)

            if bad_epochs >= patience:
                print(f"early stop at epoch {epoch + 1}")
                break
    finally:
        writer.close()


if __name__ == "__main__":
    try:
        main()
    except CollapsedTrainingError as exc:
        _emit_collapse_status(exc)
        raise SystemExit(42) from exc
