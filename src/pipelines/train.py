import time
import os
from pathlib import Path
from functools import partial
import warnings
import json
import random
from collections import Counter

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import data_config as d
from src.config import experiment_config as e
from src.config import feature_config as f
from src.config import model_config as m
from src.config import train_config as t
from src.data.dataset import (
    ResolvedPrecomputedFeatureDataset,
    ResolvedRandomChoicePrecomputedFeatureDataset,
    pad_trunc_collate_fn,
    scan_split,
    build_label_map,
    attach_labels)
from src.models.model import build_embedding_model
from src.models.triplet import BatchHardTripletLoss
from src.models.samplers import PKSampler
from src.metrics import compute_roc_auc_eer
from src.pipelines import verify as verify_mod

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# torch.set_num_threads(4)
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
    if isinstance(dataset, ResolvedPrecomputedFeatureDataset):
        return list(dataset.labels)
    if isinstance(dataset, ResolvedRandomChoicePrecomputedFeatureDataset):
        return list(dataset.labels)
    return [u.label for u in base_utterances]


def _get_canonical_train_feature_key(train_feature_mode: str) -> str:
    feature_keys = d.get_train_feature_root_keys(train_feature_mode)
    if "clean" in feature_keys:
        return "clean"
    return feature_keys[0]


def _load_train_utterances(train_feature_mode: str) -> list:
    utterances_by_set: dict[str, list] = {}
    canonical_key = _get_canonical_train_feature_key(train_feature_mode)
    for split_def in d.get_train_split_definitions():
        feat_root = d.get_train_split_feature_root(split_def, canonical_key)
        utterances_by_set[str(split_def["set_name"])] = scan_split(feat_root, pattern="*.pt")

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


def run_epoch_batchhard(model, loader, criterion, optimizer, device, train: bool, desc: str, scaler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_batches = 0
    use_amp = scaler is not None and torch.cuda.is_available() and str(device).startswith("cuda")

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for features, labels, _lengths in tqdm(loader, desc=desc):
            features = features.to(device, non_blocking=torch.cuda.is_available())
            labels = labels.to(device, non_blocking=torch.cuda.is_available())  # .view(-1)

            with autocast("cuda", enabled=use_amp):
                emb = model(features)
                loss = criterion(emb, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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


def _build_loader_kwargs(*, num_workers: int) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = t.PREFETCH_FACTOR
        kwargs["persistent_workers"] = True
    return kwargs


def _resolve_train_feature_path_for_utterance(utterance, source_key: str, target_key: str) -> Path:
    if utterance.split_root is not None and utterance.relative_path is not None:
        return d.resolve_train_feature_path_from_relative_path(
            split_root=utterance.split_root,
            relative_path=utterance.relative_path,
            source_key=source_key,
            target_key=target_key,
        )
    return d.resolve_train_feature_path_from_source_path(
        utterance.path,
        source_key,
        target_key,
    )


def build_lightweight_verify_loader(split_name: str) -> DataLoader:
    eval_splits = d.get_eval_split_definitions()
    if split_name not in eval_splits:
        raise ValueError(f"Unknown lightweight verification split: {split_name}")

    feat_root = Path(eval_splits[split_name]["feat_root"])
    utterances = scan_split(feat_root, pattern="*.pt")
    if not utterances:
        raise RuntimeError(f"No utterances found in lightweight verification split: {split_name}")

    dataset = verify_mod.VerificationDataset(utterances)
    collate = partial(verify_mod.pad_trunc_collate_verify, max_frames=f.MAX_FRAMES)
    return DataLoader(
        dataset,
        batch_size=t.P * t.K,
        shuffle=False,
        collate_fn=collate,
        **_build_loader_kwargs(num_workers=t.LIGHTWEIGHT_VERIFY_NUM_WORKERS),
    )


def run_lightweight_verification(
    model,
    loader: DataLoader,
    device: str,
    *,
    same_pairs: int,
    diff_pairs: int,
    seed: int = 37,
) -> dict[str, float]:
    all_embeddings: list[torch.Tensor] = []
    all_speaker_ids: list[str] = []

    model.eval()
    try:
        with torch.no_grad():
            for features, speaker_ids, _paths, _lengths in loader:
                features = features.to(device, non_blocking=True)
                embeddings = model(features)
                all_embeddings.append(embeddings.cpu())
                all_speaker_ids.extend(speaker_ids)
    finally:
        model.train()

    embeddings = torch.cat(all_embeddings, dim=0)
    indices_by_speaker: dict[str, list[int]] = {}
    for idx, speaker_id in enumerate(all_speaker_ids):
        indices_by_speaker.setdefault(speaker_id, []).append(idx)

    same = verify_mod.sample_pairs(
        indices_by_speaker,
        max_pairs=same_pairs,
        pair_kind="same",
        seed=seed,
    )
    diff = verify_mod.sample_pairs(
        indices_by_speaker,
        max_pairs=diff_pairs,
        pair_kind="diff",
        seed=seed + 1,
    )
    same_scores = verify_mod.cosine_scores_from_pairs(embeddings, same)
    diff_scores = verify_mod.cosine_scores_from_pairs(embeddings, diff)

    if same_scores.numel() == 0 or diff_scores.numel() == 0:
        raise RuntimeError(
            "Could not build enough lightweight verification pairs "
            f"(same={same_scores.numel()}, diff={diff_scores.numel()})."
        )

    metrics = compute_roc_auc_eer(same_scores.numpy(), diff_scores.numpy())
    return {
        "auc": float(metrics["auc"]),
        "eer": float(metrics["eer"]),
        "same_mean": float(same_scores.mean().item()),
        "diff_mean": float(diff_scores.mean().item()),
    }


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
    normalized_model_name = m.MODEL_NAME.lower().replace("-", "_")
    if normalized_model_name == "conformer":
        model_dropout = m.CONFORMER_DROPOUT
    elif normalized_model_name in {"ecapa", "ecapa_tdnn"}:
        model_dropout = m.ECAPA_DROPOUT
    else:
        model_dropout = m.DROPOUT

    run_name = e.EXP_NAME
    run_root = Path(e.RUNS_DIR) / run_name
    tb_dir = run_root / "tensorboard"
    ckpt_dir = run_root / "checkpoints"
    run_dir = ckpt_dir
    best_model_path = ckpt_dir / "best.pt"
    last_model_path = ckpt_dir / "last.pt"
    collapse_patience = getattr(t, "COLLAPSE_PATIENCE", 2)
    collapse_tolerance = 5e-4
    collapse_require_train_and_val = True
    lightweight_verify_every = int(getattr(t, "LIGHTWEIGHT_VERIFY_EVERY_N_EPOCHS", 0))
    lightweight_verify_split = str(getattr(t, "LIGHTWEIGHT_VERIFY_SPLIT", "dev_clean"))
    lightweight_verify_same_pairs = int(getattr(t, "LIGHTWEIGHT_VERIFY_SAME_PAIRS", 2000))
    lightweight_verify_diff_pairs = int(getattr(t, "LIGHTWEIGHT_VERIFY_DIFF_PAIRS", 2000))

    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_utts = _load_train_utterances(train_feature_mode)
    train_label_map = build_label_map(train_utts)
    train_utts = attach_labels(train_utts, train_label_map)
    label_counts = Counter(u.label for u in train_utts)
    train_utts = [u for u in train_utts if label_counts[u.label] >= 8]

    val_utts = scan_split(d.VAL_FEAT_ROOT, pattern="*.pt")
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

    if not d.USE_PRECOMPUTED_FEATURES:
        raise RuntimeError("Training requires precomputed features. On-the-fly WAV loading is no longer supported.")

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
    canonical_feature_key = _get_canonical_train_feature_key(train_feature_mode)
    if d.is_probabilistic_train_feature_mode(train_feature_mode):
        train_feature_paths_by_source = [
            {
                key: _resolve_train_feature_path_for_utterance(
                    utterance,
                    canonical_feature_key,
                    key,
                )
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
        train_feature_keys = d.get_train_feature_root_keys(train_feature_mode)
        train_feat_paths: list[Path] = []
        train_labels: list[int] = []
        for utterance in train_utts:
            for key in train_feature_keys:
                train_feat_paths.append(
                    _resolve_train_feature_path_for_utterance(
                        utterance,
                        canonical_feature_key,
                        key,
                    )
                )
                train_labels.append(int(utterance.label))
        train_ds = ResolvedPrecomputedFeatureDataset(
            train_feat_paths,
            train_labels,
        )

    val_ds = ResolvedPrecomputedFeatureDataset(
        [utterance.path for utterance in val_utts],
        [utterance.label for utterance in val_utts],
    )
    print(
        f"Using precomputed features: "
        f"train={train_feature_roots} | "
        f"train_probs={train_feature_probabilities} | "
        f"val={d.VAL_FEAT_ROOT}")

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
        **_build_loader_kwargs(num_workers=t.TRAIN_NUM_WORKERS))

    val_loader = DataLoader(
        val_ds,
        batch_size=p * k,
        sampler=val_sampler,
        shuffle=False,
        # drop_last=True,
        collate_fn=collate,
        **_build_loader_kwargs(num_workers=t.VAL_NUM_WORKERS))

    lightweight_verify_loader = None
    if lightweight_verify_every > 0:
        lightweight_verify_loader = build_lightweight_verify_loader(lightweight_verify_split)

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
        ecapa_channels=m.ECAPA_CHANNELS,
        ecapa_mfa_channels=m.ECAPA_MFA_CHANNELS,
        ecapa_attention_channels=m.ECAPA_ATTENTION_CHANNELS,
        ecapa_scale=m.ECAPA_SCALE,
        ecapa_se_bottleneck=m.ECAPA_SE_BOTTLENECK,
        ecapa_dropout=m.ECAPA_DROPOUT,
    ).to(device)

    triplet_loss = BatchHardTripletLoss(margin=margin, normalize=False)  # model return normalized embeddings

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)

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
                f"ecapa_channels={m.ECAPA_CHANNELS}",
                f"ecapa_mfa_channels={m.ECAPA_MFA_CHANNELS}",
                f"ecapa_attention_channels={m.ECAPA_ATTENTION_CHANNELS}",
                f"ecapa_scale={m.ECAPA_SCALE}",
                f"ecapa_se_bottleneck={m.ECAPA_SE_BOTTLENECK}",
                f"ecapa_dropout={m.ECAPA_DROPOUT}",
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
    best_eer = float("inf")
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
                desc=f"train {epoch + 1}/{epochs}",
                scaler=scaler)
            va_loss = run_epoch_batchhard(
                model,
                val_loader,
                triplet_loss,
                optimizer,
                device,
                train=False,
                desc=f"val {epoch + 1}/{epochs}",
                scaler=None)

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

            if (
                lightweight_verify_loader is not None
                and (epoch + 1) % lightweight_verify_every == 0
            ):
                verify_metrics = run_lightweight_verification(
                    model,
                    lightweight_verify_loader,
                    device,
                    same_pairs=lightweight_verify_same_pairs,
                    diff_pairs=lightweight_verify_diff_pairs,
                )
                auc = verify_metrics["auc"]
                eer = verify_metrics["eer"]
                print(f"epoch {epoch + 1} | val_loss {va_loss:.4f} | AUC {auc:.6f} | EER {eer:.6f}")
                print(
                    f"epoch {epoch + 1} | "
                    f"AUC {auc:.6f} | "
                    f"EER {eer:.6f} | "
                    f"same_mean {verify_metrics['same_mean']:.6f} | "
                    f"diff_mean {verify_metrics['diff_mean']:.6f}"
                )
                if eer < best_eer:
                    best_eer = eer
                    print(f"\nNew best EER: {best_eer:.6f} -> saving checkpoint\n")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "eer": eer,
                            "auc": auc,
                        },
                        os.path.join(run_dir, "best_eer.pt"),
                    )

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
