from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COOLDOWN_SECONDS = 10
DEFAULT_VERIFY_SPLITS = ("val", "val_noisy", "test", "test_noisy")
NOISY_VERIFY_SPLITS = ("val_noisy", "test_noisy")
CLEAN_VERIFY_SPLITS = ("val", "test")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    run_name: str
    train_feature_mode: Optional[str] = None
    train_snr_min: Optional[float] = None
    train_snr_max: Optional[float] = None
    train_noise_cache_name: Optional[str] = None
    emb_dim: int = 192
    margin: float = 0.22
    p: int = 12
    k: int = 5
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    dropout: float = 0.3
    verify_splits: tuple[str, ...] = DEFAULT_VERIFY_SPLITS
    verify_checkpoint_types: tuple[str, ...] = ("best",)
    global_summary_name: str = "verify_summary_new.csv"
    save_verify_artifacts: bool = False
    run_train: bool = True


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "run 1": ExperimentSpec(
        name="run 1",
        run_name="cnn1d_emb192_m022_P12K5_lr00005",
        run_train=False,
        verify_splits=CLEAN_VERIFY_SPLITS,
        verify_checkpoint_types=("best",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run precomputed-feature training/verification experiments, "
            "including evaluation-only presets for existing checkpoints."
        )
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS.keys(), "all"],
        default=["all"],
        help="Which experiments to run.",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip feature precomputation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Recompute feature caches even if they already exist.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=COOLDOWN_SECONDS,
        help="Pause between experiments.",
    )
    return parser.parse_args()


def select_experiments(names: list[str]) -> list[ExperimentSpec]:
    if "all" in names:
        return list(EXPERIMENTS.values())
    return [EXPERIMENTS[name] for name in names]


def run_python_code(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def get_last_checkpoint_path(spec: ExperimentSpec) -> Path:
    return PROJECT_ROOT / "runs" / spec.run_name / "checkpoints" / "last.pt"


def precompute_feature_sets(
    *,
    overwrite: bool,
    need_train_clean: bool,
    train_noise_variants: tuple[tuple[str, float, float], ...],
    need_noisy_eval: bool,
) -> subprocess.CompletedProcess:
    code = f"""
from pathlib import Path

from src.config import data_config as d
from src.config import feature_config as f
from src.data.augment import AdditiveNoise
from src.data.features import LogMelExtraction
from src.pipelines import precompute_logmels as p

fe = LogMelExtraction(
    sample_rate=f.SAMPLE_RATE,
    n_fft=f.N_FFT,
    win_length=f.WIN_LENGTH,
    hop_length=f.HOP_LENGTH,
    n_mels=f.N_MELS,
    f_min=f.FMIN,
    f_max=f.FMAX,
    eps=f.EPS,
)

for path in (
    d.VAL_FEAT_ROOT,
    d.VAL_NOISY_FEAT_ROOT,
    d.TEST_FEAT_ROOT,
    d.TEST_NOISY_FEAT_ROOT,
):
    path.mkdir(parents=True, exist_ok=True)

if {need_train_clean!r}:
    d.TRAIN_CLEAN_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
    p._precompute_split(
        split_name="train_clean",
        wav_root=Path(d.TRAIN_ROOT),
        feat_root=d.TRAIN_CLEAN_FEAT_ROOT,
        fe=fe,
        overwrite={overwrite!r},
    )
p._precompute_split(
    split_name="val",
    wav_root=Path(d.VAL_ROOT),
    feat_root=d.VAL_FEAT_ROOT,
    fe=fe,
    overwrite={overwrite!r},
)
p._precompute_split(
    split_name="test",
    wav_root=Path(d.TEST_ROOT),
    feat_root=d.TEST_FEAT_ROOT,
    fe=fe,
    overwrite={overwrite!r},
)

for cache_name, snr_min, snr_max in {list(train_noise_variants)!r}:
    feat_root = Path(d.PRECOMPUTED_ROOT) / cache_name
    feat_root.mkdir(parents=True, exist_ok=True)
    train_augmenter = AdditiveNoise(
        sample_rate=f.SAMPLE_RATE,
        noise_root=d.ESC50_TRAIN_NOISE_ROOT,
        prob=1.0,
        snr_min=snr_min,
        snr_max=snr_max,
    )
    p._precompute_split(
        split_name=f"train_noise:{{cache_name}}",
        wav_root=Path(d.TRAIN_ROOT),
        feat_root=feat_root,
        fe=fe,
        augmenter=train_augmenter,
        overwrite={overwrite!r},
    )

if {need_noisy_eval!r}:
    p._precompute_noisy_eval_splits(fe, overwrite={overwrite!r})
"""
    return run_python_code(code)


def train_experiment(spec: ExperimentSpec) -> subprocess.CompletedProcess:
    code = f"""
from pathlib import Path

from src.config import data_config as d
from src.config import experiment_config as e
from src.config import model_config as m
from src.config import train_config as t

e.EXP_NAME = {spec.run_name!r}
e.EXP_DIR = e.RUNS_DIR / e.EXP_NAME
e.TB_DIR = e.EXP_DIR / "tensorboard"
e.CKPT_DIR = e.EXP_DIR / "checkpoints"
e.RESULTS_DIR = e.EXP_DIR / "results"
e.BEST_MODEL_PATH = e.CKPT_DIR / "best.pt"
e.LAST_MODEL_PATH = e.CKPT_DIR / "last.pt"

d.TRAIN_FEATURE_MODE = {spec.train_feature_mode!r}
if {spec.train_noise_cache_name!r} is not None:
    d.TRAIN_NOISE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / {spec.train_noise_cache_name!r}

t.MARGIN = {spec.margin!r}
t.P = {spec.p!r}
t.K = {spec.k!r}
t.EPOCHS = {spec.epochs!r}
t.LEARNING_RATE = {spec.lr!r}
t.WEIGHT_DECAY = {spec.weight_decay!r}

m.EMB_DIM = {spec.emb_dim!r}
m.DROPOUT = {spec.dropout!r}

import src.pipelines.train as train_mod
train_mod.main()
"""
    return run_python_code(code)


def verify_experiment(spec: ExperimentSpec) -> subprocess.CompletedProcess:
    code = f"""
from pathlib import Path

import torch

from src.config import data_config as d
from src.config import experiment_config as e
from src.config import feature_config as f
from src.config import model_config as m
from src.models.model import CNN1DNET
from src.pipelines import verify as verify_mod

e.EXP_NAME = {spec.run_name!r}
e.EXP_DIR = e.RUNS_DIR / e.EXP_NAME
e.TB_DIR = e.EXP_DIR / "tensorboard"
e.CKPT_DIR = e.EXP_DIR / "checkpoints"
e.RESULTS_DIR = e.EXP_DIR / "results"
e.BEST_MODEL_PATH = e.CKPT_DIR / "best.pt"
e.LAST_MODEL_PATH = e.CKPT_DIR / "last.pt"

m.EMB_DIM = {spec.emb_dim!r}
m.DROPOUT = {spec.dropout!r}
if {spec.train_noise_cache_name!r} is not None:
    d.TRAIN_NOISE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / {spec.train_noise_cache_name!r}

run_root = Path(e.RUNS_DIR) / e.EXP_NAME
verify_dir = run_root / "results" / "verify"
per_run_csv = verify_dir / "metrics_summary.csv"
global_csv = Path(e.RUNS_DIR) / {spec.global_summary_name!r}
selected_splits = set({list(spec.verify_splits)!r})
checkpoint_types = {list(spec.verify_checkpoint_types)!r}

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []

for checkpoint_type in checkpoint_types:
    ckpt_path = verify_mod.resolve_checkpoint_path(run_root, checkpoint_type)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CNN1DNET(
        n_feats=f.N_MELS,
        emb_dim=m.EMB_DIM,
        dropout=m.DROPOUT,
    ).to(device)
    verify_mod.load_checkpoint_into_model(model, ckpt["model_state_dict"])

    eval_definitions = d.get_eval_split_definitions()
    for split_name, split_def in eval_definitions.items():
        if split_name not in selected_splits:
            continue

        row = verify_mod.evaluate_split(
            model=model,
            split_name=split_name,
            checkpoint_type=checkpoint_type,
            split_root=Path(split_def["wav_root"]),
            feat_root=Path(split_def["feat_root"]),
            device=device,
            output_dir=verify_dir,
            experiment_name=run_root.name,
            save_artifacts={spec.save_verify_artifacts!r},
        )
        rows.append(row)

verify_mod.upsert_metrics_rows(per_run_csv, rows)
verify_mod.upsert_metrics_rows(global_csv, rows)

print(f"Per-run summary saved to: {{per_run_csv}}")
print(f"Global summary updated at: {{global_csv}}")
"""
    return run_python_code(code)


def main() -> None:
    args = parse_args()
    experiments = select_experiments(args.experiments)
    experiments_needing_training = [
        spec
        for spec in experiments
        if spec.run_train
        and not args.skip_train
        and not get_last_checkpoint_path(spec).exists()
    ]
    need_train_clean = any(
        spec.train_feature_mode in {"clean", "both"}
        for spec in experiments_needing_training
    )
    train_noise_variants = tuple(
        dict.fromkeys(
            (
                spec.train_noise_cache_name,
                spec.train_snr_min,
                spec.train_snr_max,
            )
            for spec in experiments_needing_training
            if spec.train_feature_mode in {"noise", "both"}
            and spec.train_noise_cache_name is not None
            and spec.train_snr_min is not None
            and spec.train_snr_max is not None
        )
    )
    need_noisy_eval = any(
        split_name.endswith("_noisy")
        for spec in experiments
        for split_name in spec.verify_splits
    )

    if not args.skip_precompute:
        print("=" * 80)
        print("Precomputing required feature sets...")
        print("=" * 80)
        result = precompute_feature_sets(
            overwrite=args.overwrite_features,
            need_train_clean=need_train_clean,
            train_noise_variants=train_noise_variants,
            need_noisy_eval=need_noisy_eval,
        )
        if result.returncode != 0:
            print("Feature precomputation failed. Stopping.")
            return

    for idx, spec in enumerate(experiments, start=1):
        print("=" * 80)
        print(f"Experiment {idx}/{len(experiments)}: {spec.name} -> {spec.run_name}")
        print("=" * 80)

        if spec.run_train and not args.skip_train:
            last_checkpoint = get_last_checkpoint_path(spec)
            if last_checkpoint.exists():
                print(f"Skipping training because checkpoint already exists: {last_checkpoint}")
            else:
                print(
                    f"Training {spec.run_name} with precomputed mode "
                    f"{spec.train_feature_mode!r}..."
                )
                result = train_experiment(spec)
                if result.returncode != 0:
                    print("Training failed. Stopping.")
                    return
        elif not spec.run_train:
            print(f"Skipping training for evaluation-only experiment {spec.name}.")

        if not args.skip_verify:
            print(f"Verifying {spec.run_name} on {', '.join(spec.verify_splits)}...")
            result = verify_experiment(spec)
            if result.returncode != 0:
                print("Verification failed. Stopping.")
                return

        if idx < len(experiments) and args.cooldown_seconds > 0:
            print(f"Cooling down for {args.cooldown_seconds} seconds...")
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
