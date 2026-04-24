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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COOLDOWN_SECONDS = 20
COLLAPSED_TRAINING_EXIT_CODE = 42
DEFAULT_VERIFY_SPLITS = ("dev_clean", "dev_other", "test_clean", "test_other")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    run_name: str
    data_mode: str = "clean_only"  # "clean_only", "clean+other_prob", "clean+esc+white"
    train_sets: tuple[str, ...] = ("clean100",)  # ("clean100", "clean360", "other500")
    probabilities: Optional[dict[str, float]] = None  # {"clean": x, "other": y, "esc50": z, "white": w}
    use_other_as_augmentation: bool = False
    train_feature_mode: Optional[str] = None
    train_feature_probabilities: Optional[dict[str, float]] = None
    train_augments: tuple[tuple[str, str, float, float, float], ...] = ()
    train_augment_kind: Optional[str] = None
    noise_prob: float = 1.0
    snr_min: Optional[float] = None
    snr_max: Optional[float] = None
    train_feature_subdir: Optional[str] = None
    emb_dim: int = 256
    model_name: str = "cnn"  # cnn, rescnn, conformer, ecapa
    margin: float = 0.22
    p: int = 12
    k: int = 5
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    dropout: float = 0.3
    conformer_dropout: float = 0.4
    conformer_d_model: int = 144
    conformer_num_heads: int = 4
    conformer_ff_mult: int = 4
    conformer_conv_kernel_size: int = 31
    conformer_num_blocks: int = 3
    ecapa_channels: int = 512
    ecapa_mfa_channels: int = 1536
    ecapa_attention_channels: int = 256
    ecapa_scale: int = 8
    ecapa_se_bottleneck: int = 128
    ecapa_dropout: float = 0.0
    collapse_patience: int = 3
    resume_from_run: Optional[str] = None
    resume_checkpoint_type: str = "best"
    verify_splits: tuple[str, ...] = DEFAULT_VERIFY_SPLITS
    verify_checkpoint_types: tuple[str, ...] = ("last", "best", "best_eer")  # best, best_eer, last
    global_summary_name: str = "verify_summary_more_dataset.csv"
    save_verify_artifacts: bool = False
    skip_precompute: bool = True
    run_train: bool = True


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "1": ExperimentSpec(
        name="rescnn_mincnn_emb256_m022_P16K8_full960",
        run_name="rescnn_mincnn_emb256_m022_P16K8_full960",
        model_name="rescnn",
        p=16,
        k=8,
        lr=5e-5,
        data_mode="clean_only",
        train_sets=("clean100", "clean360", "other500"),
        train_feature_mode="clean",
        skip_precompute=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run precomputed-feature training/verification experiments, "
            "including evaluation-only presets for existing checkpoints."))
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[*EXPERIMENTS.keys(), "all"],
        default=["all"],
        help="Which experiments to run.")
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip feature precomputation.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training.")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification.")
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Recompute feature caches even if they already exist.")
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=COOLDOWN_SECONDS,
        help="Pause between experiments.")
    return parser.parse_args()


def select_experiments(names: list[str]) -> list[ExperimentSpec]:
    if "all" in names:
        return list(EXPERIMENTS.values())
    return [EXPERIMENTS[name] for name in names]


def run_python_code(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        check=False)


def get_last_checkpoint_path(spec: ExperimentSpec) -> Path:
    return e.DEFAULT_RUNS_DIR / spec.run_name / "checkpoints" / "last.pt"


def get_resume_checkpoint_path(spec: ExperimentSpec) -> Optional[Path]:
    if spec.resume_from_run is None:
        return None
    checkpoint_name_by_type = {
        "best": "best.pt",
        "best_eer": "best_eer.pt",
        "last": "last.pt",
    }
    if spec.resume_checkpoint_type not in checkpoint_name_by_type:
        raise ValueError(f"Unsupported resume_checkpoint_type: {spec.resume_checkpoint_type}")
    checkpoint_name = checkpoint_name_by_type[spec.resume_checkpoint_type]
    return Path(e.RUNS_DIR) / spec.resume_from_run / "checkpoints" / checkpoint_name


def get_effective_train_feature_mode(spec: ExperimentSpec) -> str:
    if spec.train_feature_mode is not None:
        return spec.train_feature_mode
    if spec.data_mode == "clean+esc+white":
        return "clean|noise|white"
    return "clean"


def get_effective_train_feature_probabilities(spec: ExperimentSpec) -> Optional[dict[str, float]]:
    if spec.train_feature_probabilities is not None:
        return spec.train_feature_probabilities
    if spec.data_mode != "clean+esc+white":
        return None
    probabilities = spec.probabilities or {}
    return {
        "clean": float(probabilities.get("clean", 0.0)),
        "noise": float(probabilities.get("esc50", 0.0)),
        "white": float(probabilities.get("white", 0.0)),
    }


def get_resolved_train_feature_probabilities(spec: ExperimentSpec) -> dict[str, float]:
    return d.get_train_feature_probabilities(
        get_effective_train_feature_mode(spec),
        get_effective_train_feature_probabilities(spec))


def is_train_feature_enabled(spec: ExperimentSpec, key: str) -> bool:
    mode_keys = d.get_train_feature_root_keys(get_effective_train_feature_mode(spec))
    if key not in mode_keys:
        return False
    if not d.is_probabilistic_train_feature_mode(spec.train_feature_mode):
        return True
    return get_resolved_train_feature_probabilities(spec).get(key, 0.0) > 0.0


def infer_train_augment_kind(spec: ExperimentSpec) -> Optional[str]:
    if spec.train_augment_kind is not None:
        return spec.train_augment_kind

    mode_keys = [key for key in d.get_train_feature_root_keys(get_effective_train_feature_mode(spec)) if key != "clean"]
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
        return f"{d.TRAIN_SPLIT_NAME}_{augment_kind}"

    return (
        f"{d.TRAIN_SPLIT_NAME}_{augment_kind}_snr"
        f"{format_cache_value(spec.snr_min)}_{format_cache_value(spec.snr_max)}")


def get_train_feature_overrides(spec: ExperimentSpec) -> dict[str, str]:
    return {
        augment_kind: feature_subdir
        for augment_kind, feature_subdir, _noise_prob, _snr_min, _snr_max in get_train_augments(spec)
        if is_train_feature_enabled(spec, augment_kind)}


def get_train_augments(spec: ExperimentSpec) -> tuple[tuple[str, str, float, float, float], ...]:
    if spec.train_augments:
        return spec.train_augments

    augment_kind = infer_train_augment_kind(spec)
    feature_subdir = get_train_feature_subdir(spec)
    if (
        augment_kind is None
        or feature_subdir is None
        or spec.snr_min is None
        or spec.snr_max is None):
        return ()

    return (
        (augment_kind,
         feature_subdir,
         spec.noise_prob,
         spec.snr_min,
         spec.snr_max),
    )


def collect_train_precompute_requests(
    experiments: list[ExperimentSpec],
) -> tuple[bool, tuple[tuple[str, str, float, float, float], ...]]:
    need_train_clean = False
    train_variants: list[tuple[str, str, float, float, float]] = []

    for spec in experiments:
        mode_keys = d.get_train_feature_root_keys(get_effective_train_feature_mode(spec))
        if "clean" in mode_keys and is_train_feature_enabled(spec, "clean"):
            need_train_clean = True

        for augment_kind, feature_subdir, noise_prob, snr_min, snr_max in get_train_augments(spec):
            if not is_train_feature_enabled(spec, augment_kind):
                continue
            if augment_kind not in mode_keys:
                raise ValueError(
                    f"Experiment {spec.name} defines train augment {augment_kind!r} "
                    f"but train_feature_mode is {get_effective_train_feature_mode(spec)!r}.")
            train_variants.append(
                (augment_kind,
                 feature_subdir,
                 noise_prob,
                 snr_min,
                 snr_max))

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
    eps=f.EPS)

for path in (
    *[split_def["feat_root"] for split_def in d.get_eval_split_definitions().values()],):
    path.mkdir(parents=True, exist_ok=True)

if {need_train_clean!r}:
    for split_def in d.get_train_split_definitions():
        feat_root = Path(split_def["clean_feat_root"])
        feat_root.mkdir(parents=True, exist_ok=True)
        p._precompute_split(
            split_name=f"train_clean:{{split_def['set_name']}}",
            wav_root=Path(split_def["wav_root"]),
            feat_root=feat_root,
            fe=fe,
            overwrite={overwrite!r})

for split_name, split_def in d.get_eval_split_definitions().items():
    if split_def["is_noisy"]:
        continue
    p._precompute_split(
        split_name=split_name,
        wav_root=Path(split_def["wav_root"]),
        feat_root=Path(split_def["feat_root"]),
        fe=fe,
        overwrite={overwrite!r})

for augment_kind, feature_subdir, noise_prob, snr_min, snr_max in {list(train_variants)!r}:
    for split_def in d.get_train_split_definitions():
        feat_root = Path(split_def["precomputed_root"]) / feature_subdir
        feat_root.mkdir(parents=True, exist_ok=True)
        augmenter = build_waveform_augmenter(
            kind=augment_kind,
            sample_rate=f.SAMPLE_RATE,
            noise_root=d.ESC50_TRAIN_NOISE_ROOT if augment_kind == "noise" else None,
            prob=noise_prob,
            snr_min=snr_min,
            snr_max=snr_max)
        p._precompute_split(
            split_name=f"train_{{augment_kind}}:{{split_def['set_name']}}:{{feature_subdir}}",
            wav_root=Path(split_def["wav_root"]),
            feat_root=feat_root,
            fe=fe,
            augmenter=augmenter,
            overwrite={overwrite!r})

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

e.configure_experiment(
    exp_name={spec.run_name!r},
    runs_dir=e.DEFAULT_RUNS_DIR),

d.TRAIN_SET_NAMES = {spec.train_sets!r}
d.TRAIN_DATA_MODE = {spec.data_mode!r}
d.TRAIN_DATA_PROBABILITIES = {spec.probabilities!r}
d.USE_OTHER_AS_AUGMENTATION = {spec.use_other_as_augmentation!r}
d.TRAIN_FEATURE_MODE = {get_effective_train_feature_mode(spec)!r}
d.TRAIN_FEATURE_PROBABILITIES = {get_effective_train_feature_probabilities(spec)!r}
train_feature_overrides = {get_train_feature_overrides(spec)!r}
for split_def in d.get_train_split_definitions():
    set_name = str(split_def["set_name"])
    precomputed_root = Path(split_def["precomputed_root"])
    if "noise" in train_feature_overrides:
        d.TRAIN_NOISE_FEAT_ROOTS_BY_SET[set_name] = precomputed_root / train_feature_overrides["noise"]
    if "white" in train_feature_overrides:
        d.TRAIN_WHITE_FEAT_ROOTS_BY_SET[set_name] = precomputed_root / train_feature_overrides["white"]

t.MARGIN = {spec.margin!r}
t.P = {spec.p!r}
t.K = {spec.k!r}
t.EPOCHS = {spec.epochs!r}
t.LEARNING_RATE = {spec.lr!r}
t.WEIGHT_DECAY = {spec.weight_decay!r}

t.COLLAPSE_PATIENCE = {spec.collapse_patience!r}
t.LIGHTWEIGHT_VERIFY_EVERY_N_EPOCHS = {1 if spec.model_name.lower().replace("-", "_") in {"cnn", "conformer", "ecapa", "ecapa_tdnn", "rescnn"} else 0!r}
t.LIGHTWEIGHT_VERIFY_SPLIT = {"dev_clean"!r}
t.LIGHTWEIGHT_VERIFY_SAME_PAIRS = {4000!r}
t.LIGHTWEIGHT_VERIFY_DIFF_PAIRS = {4000!r}
resume_checkpoint_path = {str(get_resume_checkpoint_path(spec)) if get_resume_checkpoint_path(spec) is not None else None!r}
t.RESUME_CHECKPOINT_PATH = resume_checkpoint_path

m.EMB_DIM = {spec.emb_dim!r}
m.MODEL_NAME = {spec.model_name!r}
m.DROPOUT = {spec.dropout!r}
m.CONFORMER_DROPOUT = {spec.conformer_dropout!r}
m.CONFORMER_D_MODEL = {spec.conformer_d_model!r}
m.CONFORMER_NUM_HEADS = {spec.conformer_num_heads!r}
m.CONFORMER_FF_MULT = {spec.conformer_ff_mult!r}
m.CONFORMER_CONV_KERNEL_SIZE = {spec.conformer_conv_kernel_size!r}
m.CONFORMER_NUM_BLOCKS = {spec.conformer_num_blocks!r}
m.ECAPA_CHANNELS = {spec.ecapa_channels!r}
m.ECAPA_MFA_CHANNELS = {spec.ecapa_mfa_channels!r}
m.ECAPA_ATTENTION_CHANNELS = {spec.ecapa_attention_channels!r}
m.ECAPA_SCALE = {spec.ecapa_scale!r}
m.ECAPA_SE_BOTTLENECK = {spec.ecapa_se_bottleneck!r}
m.ECAPA_DROPOUT = {spec.ecapa_dropout!r}

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
from src.config import train_config as t
from src.models.model import build_embedding_model
from src.pipelines import verify as verify_mod

e.configure_experiment(
    exp_name={spec.run_name!r},
    runs_dir=e.DEFAULT_RUNS_DIR),

d.TRAIN_SET_NAMES = {spec.train_sets!r}
d.TRAIN_DATA_MODE = {spec.data_mode!r}
d.TRAIN_DATA_PROBABILITIES = {spec.probabilities!r}
d.USE_OTHER_AS_AUGMENTATION = {spec.use_other_as_augmentation!r}
d.TRAIN_FEATURE_MODE = {get_effective_train_feature_mode(spec)!r}
d.TRAIN_FEATURE_PROBABILITIES = {get_effective_train_feature_probabilities(spec)!r}
m.EMB_DIM = {spec.emb_dim!r}
m.MODEL_NAME = {spec.model_name!r}
m.DROPOUT = {spec.dropout!r}
m.CONFORMER_DROPOUT = {spec.conformer_dropout!r}
m.CONFORMER_D_MODEL = {spec.conformer_d_model!r}
m.CONFORMER_NUM_HEADS = {spec.conformer_num_heads!r}
m.CONFORMER_FF_MULT = {spec.conformer_ff_mult!r}
m.CONFORMER_CONV_KERNEL_SIZE = {spec.conformer_conv_kernel_size!r}
m.CONFORMER_NUM_BLOCKS = {spec.conformer_num_blocks!r}
m.ECAPA_CHANNELS = {spec.ecapa_channels!r}
m.ECAPA_MFA_CHANNELS = {spec.ecapa_mfa_channels!r}
m.ECAPA_ATTENTION_CHANNELS = {spec.ecapa_attention_channels!r}
m.ECAPA_SCALE = {spec.ecapa_scale!r}
m.ECAPA_SE_BOTTLENECK = {spec.ecapa_se_bottleneck!r}
m.ECAPA_DROPOUT = {spec.ecapa_dropout!r}
t.P = {spec.p!r}
t.K = {spec.k!r}
train_feature_overrides = {get_train_feature_overrides(spec)!r}
for split_def in d.get_train_split_definitions():
    set_name = str(split_def["set_name"])
    precomputed_root = Path(split_def["precomputed_root"])
    if "noise" in train_feature_overrides:
        d.TRAIN_NOISE_FEAT_ROOTS_BY_SET[set_name] = precomputed_root / train_feature_overrides["noise"]
    if "white" in train_feature_overrides:
        d.TRAIN_WHITE_FEAT_ROOTS_BY_SET[set_name] = precomputed_root / train_feature_overrides["white"]

run_root = Path(e.EXP_DIR)
verify_dir = run_root / "results" / "verify"
per_run_csv = verify_dir / "metrics_summary.csv"
global_csv_paths = verify_mod.get_global_summary_paths(
    runs_dir=Path(e.RUNS_DIR),
    primary_summary_name={spec.global_summary_name!r},
    model_name=m.MODEL_NAME)
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
    verify_mod.load_checkpoint_into_model(model, ckpt["model_state_dict"])

    for split_name, split_def in d.get_eval_split_definitions().items():
        if split_name not in selected_splits:
            continue

        row = verify_mod.evaluate_split(
            model=model,
            split_name=split_name,
            checkpoint_type=checkpoint_type,
            feat_root=Path(split_def["feat_root"]),
            device=device,
            output_dir=verify_dir,
            experiment_name=run_root.name,
            save_artifacts={spec.save_verify_artifacts!r})
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
        and (spec.resume_from_run is not None or not get_last_checkpoint_path(spec).exists())]
    need_train_clean, train_variants = collect_train_precompute_requests(train_experiments)

    should_skip_precompute = args.skip_precompute or all(spec.skip_precompute for spec in experiments)

    if not should_skip_precompute:
        print("=" * 80)
        print("Precomputing required feature sets...")
        print("=" * 80)
        result = precompute_feature_sets(
            overwrite=args.overwrite_features,
            need_train_clean=need_train_clean,
            train_variants=train_variants)
        if result.returncode != 0:
            print("Feature precomputation failed. Stopping.")
            return

    for idx, spec in enumerate(experiments, start=1):
        print("=" * 80)
        print(
            f"Experiment {idx}/{len(experiments)}: {spec.name} -> "
            f"{e.DEFAULT_RUNS_DIR / spec.run_name}")
        print("=" * 80)

        if spec.run_train and not args.skip_train:
            last_checkpoint = get_last_checkpoint_path(spec)
            should_resume = spec.resume_from_run is not None
            training_collapsed = False
            if last_checkpoint.exists() and not should_resume:
                print(f"Skipping training because checkpoint already exists: {last_checkpoint}")
            else:
                print(
                    f"Training {spec.run_name} with precomputed mode "
                    f"{get_effective_train_feature_mode(spec)!r}...")
                result = train_experiment(spec)
                if result.returncode == COLLAPSED_TRAINING_EXIT_CODE:
                    training_collapsed = True
                    print(
                        f"Training aborted early due to collapse for {spec.run_name}. "
                        "Skipping verification and continuing.")
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
