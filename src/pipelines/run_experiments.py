from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import data_config as d
from src.config import experiment_config as e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COOLDOWN_SECONDS = 1
COLLAPSED_TRAINING_EXIT_CODE = 42
DEFAULT_VERIFY_SPLITS = (
    "val", "val_noise", "val_white",
    "test", "test_noise", "test_white")
NOISY_VERIFY_SPLITS = (
    "val_noise", "val_white",
    "test_noise", "test_white")
WHITE_VERIFY_SPLITS = ("val_white", "test_white")
CLEAN_VERIFY_SPLITS = ("val", "test")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    run_name: str
    train_feature_mode: Optional[str] = None
    train_feature_probabilities: Optional[dict[str, float]] = None
    train_augments: tuple[tuple[str, str, float, float, float], ...] = ()
    train_augment_kind: Optional[str] = None
    noise_prob: float = 1.0
    snr_min: Optional[float] = None
    snr_max: Optional[float] = None
    train_feature_subdir: Optional[str] = None
    emb_dim: int = 192
    model_name: str = "conformer"  # conformer, cnn
    margin: float = 0.22
    p: int = 12
    k: int = 5
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    dropout: float = 0.3
    conformer_d_model: int = 144
    conformer_num_heads: int = 4
    conformer_ff_mult: int = 4
    conformer_conv_kernel_size: int = 15
    conformer_num_blocks: int = 2
    verify_splits: tuple[str, ...] = DEFAULT_VERIFY_SPLITS
    verify_checkpoint_types: tuple[str, ...] = ("best",)  # best, last
    global_summary_name: str = "verify_summary_new.csv"
    save_verify_artifacts: bool = False
    run_train: bool = True


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "1": ExperimentSpec(
        name="conf_baseline",
        run_name="conf_baseline",
        train_feature_mode="clean",
        verify_splits=DEFAULT_VERIFY_SPLITS,
    ),
    "2": ExperimentSpec(
        name="conf_clean+esc50_snr20+white25",
        run_name="conf_clean+esc50_snr20+white25  0.5 0.3 0.2",
        train_feature_mode="clean|noise|white",
        train_feature_probabilities={"clean": 0.5, "noise": 0.3, "white": 0.2},
        train_augments=(
            ("noise", "train_noise", 1.0, 20.0, 20.0),
            ("white", "train_white_snr25", 1.0, 25.0, 25.0)),
        verify_splits=DEFAULT_VERIFY_SPLITS,
    ),
    "3": ExperimentSpec(
        name="clean+esc50_snr20",
        run_name="clean+esc50_snr20  0.5 0.5",
        train_feature_mode="clean|noise|white",
        train_feature_probabilities={"clean": 0.5, "noise": 0.5, "white": 0.0},
        train_augments=(
            ("noise", "train_noise", 1.0, 20.0, 20.0),
            ("white", "train_white_snr25", 1.0, 25.0, 25.0)),
        verify_splits=DEFAULT_VERIFY_SPLITS,
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
    return Path(e.RUNS_DIR) / spec.run_name / "checkpoints" / "last.pt"


def infer_train_augment_kind(spec: ExperimentSpec) -> Optional[str]:
    if spec.train_augment_kind is not None:
        return spec.train_augment_kind

    mode_keys = [key for key in d.get_train_feature_root_keys(spec.train_feature_mode) if key != "clean"]
    if len(mode_keys) == 1:
        return mode_keys[0]
    return None


def format_cache_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def get_train_feature_subdir(spec: ExperimentSpec) -> Optional[str]:
    if spec.train_feature_subdir is not None:
        return spec.train_feature_subdir

    augment_kind = infer_train_augment_kind(spec)
    if augment_kind is None:
        return None
    if spec.snr_min is None or spec.snr_max is None:
        return f"train_{augment_kind}"

    return (
        f"train_{augment_kind}_snr"
        f"{format_cache_value(spec.snr_min)}_{format_cache_value(spec.snr_max)}"
    )


def get_train_feature_overrides(spec: ExperimentSpec) -> dict[str, str]:
    return {
        augment_kind: feature_subdir
        for augment_kind, feature_subdir, _noise_prob, _snr_min, _snr_max in get_train_augments(spec)
    }


def get_train_augments(spec: ExperimentSpec) -> tuple[tuple[str, str, float, float, float], ...]:
    if spec.train_augments:
        return spec.train_augments

    augment_kind = infer_train_augment_kind(spec)
    feature_subdir = get_train_feature_subdir(spec)
    if (
        augment_kind is None
        or feature_subdir is None
        or spec.snr_min is None
        or spec.snr_max is None
    ):
        return ()

    return (
        (
            augment_kind,
            feature_subdir,
            spec.noise_prob,
            spec.snr_min,
            spec.snr_max,
        ),
    )


def collect_train_precompute_requests(
    experiments: list[ExperimentSpec],
) -> tuple[bool, tuple[tuple[str, str, float, float, float], ...]]:
    need_train_clean = False
    train_variants: list[tuple[str, str, float, float, float]] = []

    for spec in experiments:
        mode_keys = d.get_train_feature_root_keys(spec.train_feature_mode)
        if "clean" in mode_keys:
            need_train_clean = True

        for augment_kind, feature_subdir, noise_prob, snr_min, snr_max in get_train_augments(spec):
            if augment_kind not in mode_keys:
                raise ValueError(
                    f"Experiment {spec.name} defines train augment {augment_kind!r} "
                    f"but train_feature_mode is {spec.train_feature_mode!r}."
                )
            train_variants.append(
                (
                    augment_kind,
                    feature_subdir,
                    noise_prob,
                    snr_min,
                    snr_max,
                )
            )

    return need_train_clean, tuple(dict.fromkeys(train_variants))


def precompute_feature_sets(
    *,
    overwrite: bool,
    need_train_clean: bool,
    train_variants: tuple[tuple[str, str, float, float, float], ...],
) -> subprocess.CompletedProcess:
    code = f"""
from pathlib import Path

from src.config import data_config as d
from src.config import feature_config as f
from src.data.augment import build_waveform_augmenter
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
    d.VAL_WHITE_FEAT_ROOT,
    d.TEST_FEAT_ROOT,
    d.TEST_NOISY_FEAT_ROOT,
    d.TEST_WHITE_FEAT_ROOT,
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

for augment_kind, feature_subdir, noise_prob, snr_min, snr_max in {list(train_variants)!r}:
    feat_root = Path(d.PRECOMPUTED_ROOT) / feature_subdir
    feat_root.mkdir(parents=True, exist_ok=True)
    augmenter = build_waveform_augmenter(
        kind=augment_kind,
        sample_rate=f.SAMPLE_RATE,
        noise_root=d.ESC50_TRAIN_NOISE_ROOT if augment_kind == "noise" else None,
        prob=noise_prob,
        snr_min=snr_min,
        snr_max=snr_max,
    )
    p._precompute_split(
        split_name=f"train_{{augment_kind}}:{{feature_subdir}}",
        wav_root=Path(d.TRAIN_ROOT),
        feat_root=feat_root,
        fe=fe,
        augmenter=augmenter,
        overwrite={overwrite!r},
    )

p._precompute_augmented_eval_splits(fe, overwrite={overwrite!r})
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
d.TRAIN_FEATURE_PROBABILITIES = {spec.train_feature_probabilities!r}
train_feature_overrides = {get_train_feature_overrides(spec)!r}
if "noise" in train_feature_overrides:
    d.TRAIN_NOISE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / train_feature_overrides["noise"]
if "white" in train_feature_overrides:
    d.TRAIN_WHITE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / train_feature_overrides["white"]

t.MARGIN = {spec.margin!r}
t.P = {spec.p!r}
t.K = {spec.k!r}
t.EPOCHS = {spec.epochs!r}
t.LEARNING_RATE = {spec.lr!r}
t.WEIGHT_DECAY = {spec.weight_decay!r}

m.EMB_DIM = {spec.emb_dim!r}
m.MODEL_NAME = {spec.model_name!r}
m.DROPOUT = {spec.dropout!r}
m.CONFORMER_D_MODEL = {spec.conformer_d_model!r}
m.CONFORMER_NUM_HEADS = {spec.conformer_num_heads!r}
m.CONFORMER_FF_MULT = {spec.conformer_ff_mult!r}
m.CONFORMER_CONV_KERNEL_SIZE = {spec.conformer_conv_kernel_size!r}
m.CONFORMER_NUM_BLOCKS = {spec.conformer_num_blocks!r}

import src.pipelines.train as train_mod
try:
    train_mod.main()
except train_mod.CollapsedTrainingError as exc:
    train_mod._emit_collapse_status(exc)
    raise SystemExit({COLLAPSED_TRAINING_EXIT_CODE})
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
from src.models.model import build_embedding_model
from src.pipelines import verify as verify_mod

e.EXP_NAME = {spec.run_name!r}
e.EXP_DIR = e.RUNS_DIR / e.EXP_NAME
e.TB_DIR = e.EXP_DIR / "tensorboard"
e.CKPT_DIR = e.EXP_DIR / "checkpoints"
e.RESULTS_DIR = e.EXP_DIR / "results"
e.BEST_MODEL_PATH = e.CKPT_DIR / "best.pt"
e.LAST_MODEL_PATH = e.CKPT_DIR / "last.pt"

m.EMB_DIM = {spec.emb_dim!r}
m.MODEL_NAME = {spec.model_name!r}
m.DROPOUT = {spec.dropout!r}
m.CONFORMER_D_MODEL = {spec.conformer_d_model!r}
m.CONFORMER_NUM_HEADS = {spec.conformer_num_heads!r}
m.CONFORMER_FF_MULT = {spec.conformer_ff_mult!r}
m.CONFORMER_CONV_KERNEL_SIZE = {spec.conformer_conv_kernel_size!r}
m.CONFORMER_NUM_BLOCKS = {spec.conformer_num_blocks!r}
train_feature_overrides = {get_train_feature_overrides(spec)!r}
if "noise" in train_feature_overrides:
    d.TRAIN_NOISE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / train_feature_overrides["noise"]
if "white" in train_feature_overrides:
    d.TRAIN_WHITE_FEAT_ROOT = Path(d.PRECOMPUTED_ROOT) / train_feature_overrides["white"]

run_root = Path(e.RUNS_DIR) / e.EXP_NAME
verify_dir = run_root / "results" / "verify"
per_run_csv = verify_dir / "metrics_summary.csv"
global_csv_paths = verify_mod.get_global_summary_paths(
    runs_dir=Path(e.RUNS_DIR),
    primary_summary_name={spec.global_summary_name!r},
    model_name=m.MODEL_NAME,
)
selected_splits = set({list(spec.verify_splits)!r})
checkpoint_types = {list(spec.verify_checkpoint_types)!r}

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []

for checkpoint_type in checkpoint_types:
    ckpt_path = verify_mod.resolve_checkpoint_path(run_root, checkpoint_type)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_embedding_model(
        m.MODEL_NAME,
        n_feats=f.N_MELS,
        emb_dim=m.EMB_DIM,
        dropout=m.DROPOUT,
        conformer_d_model=m.CONFORMER_D_MODEL,
        conformer_num_heads=m.CONFORMER_NUM_HEADS,
        conformer_ff_mult=m.CONFORMER_FF_MULT,
        conformer_conv_kernel_size=m.CONFORMER_CONV_KERNEL_SIZE,
        conformer_num_blocks=m.CONFORMER_NUM_BLOCKS,
    ).to(device)
    verify_mod.load_checkpoint_into_model(model, ckpt["model_state_dict"])

    for split_name, split_def in d.get_eval_split_definitions().items():
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
for global_csv in global_csv_paths:
    verify_mod.upsert_metrics_rows(global_csv, rows)

print(f"Per-run summary saved to: {{per_run_csv}}")
for global_csv in global_csv_paths:
    print(f"Global summary updated at: {{global_csv}}")
"""
    return run_python_code(code)


def main() -> None:
    args = parse_args()
    experiments = select_experiments(args.experiments)
    train_experiments = [
        spec
        for spec in experiments
        if spec.run_train
        and not args.skip_train
        and not get_last_checkpoint_path(spec).exists()
    ]
    need_train_clean, train_variants = collect_train_precompute_requests(train_experiments)

    if not args.skip_precompute:
        print("=" * 80)
        print("Precomputing required feature sets...")
        print("=" * 80)
        result = precompute_feature_sets(
            overwrite=args.overwrite_features,
            need_train_clean=need_train_clean,
            train_variants=train_variants,
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
            training_collapsed = False
            if last_checkpoint.exists():
                print(f"Skipping training because checkpoint already exists: {last_checkpoint}")
            else:
                print(
                    f"Training {spec.run_name} with precomputed mode "
                    f"{spec.train_feature_mode!r}..."
                )
                result = train_experiment(spec)
                if result.returncode == COLLAPSED_TRAINING_EXIT_CODE:
                    training_collapsed = True
                    print(
                        f"Training aborted early due to collapse for {spec.run_name}. "
                        "Skipping verification and continuing."
                    )
                elif result.returncode != 0:
                    print("Training failed. Stopping.")
                    return
        elif not spec.run_train:
            training_collapsed = False
            print(f"Skipping training for evaluation-only experiment {spec.name}.")
        else:
            training_collapsed = False

        if training_collapsed:
            pass
        elif not args.skip_verify:
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
