from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
COOL_DOWN_SECONDS = 1
DEFAULT_EVAL_SPLITS = [
    "val",
    "val_noisy_snr15",
    "test",
    "test_noisy_snr15",
    "test_noisy_snr10",
]
DEFAULT_NOISY_EVAL_ONLY_RUNS = [
    "cnn1d_emb192_noise_snr10",
    "cnn1d_emb192_noise_snr15",
    "cnn1d_emb192_noise_snr20",
    "cnn1d_emb192_noise_snr25",
    "cnn1d_noise+clean_01",
]
DEFAULT_NOISY_EVAL_SPLITS = [
    "val_noisy_snr15",
    "test_noisy_snr15",
    "test_noisy_snr10",
]
DEFAULT_CLEAN_LAST_SPLITS = [
    "val",
    "test",
]

BASE_TRAIN_ARGS = [
    "--margin", "0.22",
    "--p", "12",
    "--k", "5",
    "--emb-dim", "192",
    "--lr", "5e-4",
    "--wd", "1e-4",
    "--lr-scheduler", "none",
]


@dataclass(frozen=True)
class TrainNoiseConfig:
    label: str
    use_musan: bool
    snr_min: Optional[float] = None
    snr_max: Optional[float] = None
    noise_prob: float = 1.0

    @property
    def uses_noise(self) -> bool:
        return self.use_musan

    @property
    def feature_mode(self) -> str:
        return "clean+noise" if self.uses_noise else "clean"

    @property
    def train_noise_root_name(self) -> Optional[str]:
        if not self.uses_noise:
            return None
        return f"train_noise_{self.label}"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_name: str
    train_noise: TrainNoiseConfig


EXPERIMENTS = {
    "clean_baseline": ExperimentConfig(
        name="clean_baseline",
        run_name="cnn1d_emb192_m022_P12K5_clean_baseline",
        train_noise=TrainNoiseConfig(
            label="clean",
            use_musan=False,
            noise_prob=0.0,
        ),
    ),
    "musan_snr10_20": ExperimentConfig(
        name="musan_snr10_20",
        run_name="cnn1d_emb192_m022_P12K5_clean_plus_musan_snr10_20",
        train_noise=TrainNoiseConfig(
            label="musan_snr10_20",
            use_musan=True,
            snr_min=10.0,
            snr_max=20.0,
            noise_prob=1.0,
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute features, train, and verify the configured speaker-verification experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="overnight_eval",
        choices=["full", "overnight_eval"],
        help="Run the overnight evaluation-only queue or the normal pipeline.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
        help="Which named experiments to run. Defaults to all.",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=DEFAULT_EVAL_SPLITS,
        help="Evaluation splits to verify for each experiment.",
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Recompute existing feature caches instead of reusing them.",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Skip feature precomputation.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run remaining stages.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=COOL_DOWN_SECONDS,
        help="Cooldown between experiments.",
    )
    parser.add_argument(
        "--existing-run-names",
        nargs="+",
        default=DEFAULT_NOISY_EVAL_ONLY_RUNS,
        help="Already-trained run folder names to evaluate in overnight_eval mode.",
    )
    parser.add_argument(
        "--clean-last-experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
        help="Current run_experiments entries whose last.pt should be evaluated on clean splits in overnight_eval mode.",
    )
    return parser.parse_args()


def select_experiments(selected_names: List[str]) -> List[ExperimentConfig]:
    if "all" in selected_names:
        return list(EXPERIMENTS.values())
    return [EXPERIMENTS[name] for name in selected_names]


def run_python_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SCRIPT_DIR),
        check=False,
    )


def precompute_shared_features(overwrite: bool = False) -> subprocess.CompletedProcess:
    overwrite_flag = "True" if overwrite else "False"
    code = dedent(
        f"""
        from pathlib import Path
        import constants as c
        from features import LogMelExtraction
        from precompute_logmels import precompute_split, precompute_noisy_eval_splits

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

        c.TRAIN_CLEAN_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
        c.VAL_FEAT_ROOT.mkdir(parents=True, exist_ok=True)
        c.TEST_FEAT_ROOT.mkdir(parents=True, exist_ok=True)

        precompute_split(
            split_name="train_clean",
            wav_root=Path(c.TRAIN_ROOT),
            feat_root=c.TRAIN_CLEAN_FEAT_ROOT,
            fe=fe,
            overwrite={overwrite_flag},
            seed=101,
        )
        precompute_split(
            split_name="val",
            wav_root=Path(c.VAL_ROOT),
            feat_root=c.VAL_FEAT_ROOT,
            fe=fe,
            overwrite={overwrite_flag},
            seed=102,
        )
        precompute_split(
            split_name="test",
            wav_root=Path(c.TEST_ROOT),
            feat_root=c.TEST_FEAT_ROOT,
            fe=fe,
            overwrite={overwrite_flag},
            seed=103,
        )
        precompute_noisy_eval_splits(fe=fe, overwrite={overwrite_flag})
        """
    )
    return run_python_snippet(code)


def precompute_train_noise_features(config: ExperimentConfig, overwrite: bool = False) -> subprocess.CompletedProcess:
    if not config.train_noise.uses_noise:
        return subprocess.CompletedProcess(args=[], returncode=0)

    overwrite_flag = "True" if overwrite else "False"
    code = dedent(
        f"""
        from pathlib import Path
        import constants as c
        from augment import AdditiveNoise
        from features import LogMelExtraction
        from precompute_logmels import build_train_eval_noise_file_lists, precompute_split

        train_noise_paths, _eval_noise_paths = build_train_eval_noise_file_lists()
        c.TRAIN_NOISE_FEAT_ROOT = c.PRECOMPUTED_ROOT / "{config.train_noise.train_noise_root_name}"
        c.TRAIN_NOISE_FEAT_ROOT.mkdir(parents=True, exist_ok=True)

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

        augmenter = AdditiveNoise(
            sample_rate=c.SAMPLE_RATE,
            noise_paths=train_noise_paths,
            prob={float(config.train_noise.noise_prob)},
            snr_min={float(config.train_noise.snr_min)},
            snr_max={float(config.train_noise.snr_max)},
            min_noise_seconds=c.MIN_NOISE_SECONDS,
        )

        precompute_split(
            split_name="{config.train_noise.train_noise_root_name}",
            wav_root=Path(c.TRAIN_ROOT),
            feat_root=c.TRAIN_NOISE_FEAT_ROOT,
            fe=fe,
            augmenter=augmenter,
            overwrite={overwrite_flag},
            seed=201,
        )
        """
    )
    return run_python_snippet(code)


def train_experiment(config: ExperimentConfig) -> subprocess.CompletedProcess:
    extra_args = [
        "--run-name", config.run_name,
        "--train-feature-mode", config.train_noise.feature_mode,
    ]
    args_repr = repr(BASE_TRAIN_ARGS + extra_args)
    train_noise_root_name = config.train_noise.train_noise_root_name
    noise_root_assignment = (
        f'c.TRAIN_NOISE_FEAT_ROOT = c.PRECOMPUTED_ROOT / "{train_noise_root_name}"'
        if train_noise_root_name is not None
        else "c.TRAIN_NOISE_FEAT_ROOT = c.PRECOMPUTED_ROOT / 'train_noise'"
    )
    code = dedent(
        f"""
        import sys
        import constants as c

        {noise_root_assignment}
        c.TRAIN_FEATURE_MODE = "{config.train_noise.feature_mode}"

        import train

        sys.argv = ["train.py", *{args_repr}]
        train.main()
        """
    )
    return run_python_snippet(code)


def verify_run_name(
    run_name: str,
    eval_splits: List[str],
    checkpoint_type: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "verify.py",
            "--experiment",
            run_name,
            "--checkpoint-type",
            checkpoint_type,
            "--eval-splits",
            *eval_splits,
        ],
        cwd=str(SCRIPT_DIR),
        check=False,
    )


def verify_experiment(config: ExperimentConfig, eval_splits: List[str], checkpoint_type: str = "best") -> subprocess.CompletedProcess:
    return verify_run_name(
        run_name=config.run_name,
        eval_splits=eval_splits,
        checkpoint_type=checkpoint_type,
    )


def run_overnight_eval(args) -> None:
    existing_run_names = args.existing_run_names
    clean_last_configs = select_experiments(args.clean_last_experiments)

    jobs = []
    for run_name in existing_run_names:
        for checkpoint_type in ("best", "last"):
            jobs.append(
                {
                    "run_name": run_name,
                    "checkpoint_type": checkpoint_type,
                    "eval_splits": DEFAULT_NOISY_EVAL_SPLITS,
                }
            )

    for config in clean_last_configs:
        jobs.append(
            {
                "run_name": config.run_name,
                "checkpoint_type": "last",
                "eval_splits": DEFAULT_CLEAN_LAST_SPLITS,
            }
        )

    for i, job in enumerate(jobs, start=1):
        print("=" * 80)
        print(
            f"Overnight eval job {i}/{len(jobs)} | "
            f"run_name={job['run_name']} | "
            f"checkpoint={job['checkpoint_type']} | "
            f"splits={', '.join(job['eval_splits'])}"
        )
        print("=" * 80)

        verify_result = verify_run_name(
            run_name=job["run_name"],
            eval_splits=job["eval_splits"],
            checkpoint_type=job["checkpoint_type"],
        )
        if verify_result.returncode != 0:
            print("Verification failed. Stopping.")
            break

        if i < len(jobs) and args.cooldown_seconds > 0:
            print(f"Cooling CPU for {args.cooldown_seconds} seconds...")
            time.sleep(args.cooldown_seconds)


def main() -> None:
    args = parse_args()

    if args.mode == "overnight_eval":
        run_overnight_eval(args)
        return

    experiments = select_experiments(args.experiments)

    if not args.skip_precompute:
        print("=" * 80)
        print("Precomputing shared clean/noisy evaluation features...")
        print("=" * 80)
        shared_precompute_result = precompute_shared_features(overwrite=args.overwrite_features)
        if shared_precompute_result.returncode != 0:
            print("Shared feature precomputation failed. Stopping.")
            return

    for i, config in enumerate(experiments, start=1):
        print("=" * 80)
        print(
            f"Starting experiment {i}/{len(experiments)} | "
            f"name={config.name} | run_name={config.run_name}"
        )
        print("=" * 80)

        if not args.skip_precompute and config.train_noise.uses_noise:
            print(
                f"Precomputing train-noise features for {config.name} | "
                f"musan={config.train_noise.use_musan} | "
                f"snr={config.train_noise.snr_min:.0f}-{config.train_noise.snr_max:.0f} dB"
            )
            precompute_result = precompute_train_noise_features(
                config,
                overwrite=args.overwrite_features,
            )
            if precompute_result.returncode != 0:
                print("Train-noise feature precomputation failed. Stopping.")
                break

        if not args.skip_train:
            print(f"Launching training for {config.run_name}...")
            train_result = train_experiment(config)
            if train_result.returncode != 0:
                print("Training failed. Stopping.")
                break

        if not args.skip_verify:
            print(f"Running verification for {config.run_name} on {', '.join(args.eval_splits)}...")
            verify_result = verify_experiment(config, args.eval_splits, checkpoint_type="best")
            if verify_result.returncode != 0:
                print("Verification failed. Stopping.")
                break

        if i < len(experiments) and args.cooldown_seconds > 0:
            print(f"Cooling CPU for {args.cooldown_seconds} seconds...")
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
