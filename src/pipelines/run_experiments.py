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
COOL_DOWN_SECONDS = 90

DEFAULT_FULL_EXPERIMENTS = [
    "musan_snr20_p050",
    "musan_snr20_p035",
]

DEFAULT_EVAL_SPLITS = [
    "val",
    "val_noisy_snr15",
    "test",
    "test_noisy_snr15",
    "test_noisy_snr10",
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
    noise_kind: str  # "clean", "musan", "white", "musan+white"
    snr_min: Optional[float] = None
    snr_max: Optional[float] = None
    noise_prob: float = 1.0

    @property
    def uses_noise(self) -> bool:
        return self.noise_kind != "clean"

    @property
    def feature_mode(self) -> str:
        mapping = {
            "clean": "clean",
            "musan": "clean+noise",
            "white": "clean+white",
            "white_mild": "clean+white_mild",
            "musan+white": "clean+musan+white",
        }
        return mapping[self.noise_kind]

    @property
    def train_noise_root_name(self) -> Optional[str]:
        mapping = {
            "clean": None,
            "musan": self.label,
            "white": self.label,
            "white_mild": self.label,
            "musan+white": self.label,
        }
        return mapping[self.noise_kind]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_name: str
    train_noise: TrainNoiseConfig


EXPERIMENTS = {
    "musan_snr20_p050": ExperimentConfig(
        name="musan_snr20_p050",
        run_name="cnn1d_emb192_m022_P12K5_clean+musan_snr20_p050",
        train_noise=TrainNoiseConfig(
            label="train_noise_musan_snr20_p050",
            noise_kind="musan",
            snr_min=20.0,
            snr_max=20.0,
            noise_prob=0.5,
        ),
    ),
    "musan_snr20_p035": ExperimentConfig(
        name="musan_snr20_p035",
        run_name="cnn1d_emb192_m022_P12K5_clean+musan_snr20_p035",
        train_noise=TrainNoiseConfig(
            label="train_noise_musan_snr20_p035",
            noise_kind="musan",
            snr_min=20.0,
            snr_max=20.0,
            noise_prob=0.35,
        ),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute features, train, and verify the configured speaker-verification experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=DEFAULT_FULL_EXPERIMENTS,
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
        from augment import AdditiveNoise, WhiteNoise, RandomChoiceAugment
        from features import LogMelExtraction
        from precompute_logmels import build_train_eval_noise_file_lists, precompute_split

        train_noise_paths, _ = build_train_eval_noise_file_lists()

        feat_root = c.PRECOMPUTED_ROOT / "{config.train_noise.train_noise_root_name}"
        feat_root.mkdir(parents=True, exist_ok=True)

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

        musan_augmenter = AdditiveNoise(
            sample_rate=c.SAMPLE_RATE,
            noise_paths=train_noise_paths,
            prob={float(config.train_noise.noise_prob)},
            snr_min={float(config.train_noise.snr_min)},
            snr_max={float(config.train_noise.snr_max)},
            min_noise_seconds=c.MIN_NOISE_SECONDS,
        )

        white_augmenter = WhiteNoise(
            prob={float(config.train_noise.noise_prob)},
            snr_min={float(config.train_noise.snr_min)},
            snr_max={float(config.train_noise.snr_max)},
        )

        noise_kind = "{config.train_noise.noise_kind}"
        if noise_kind == "musan":
            augmenter = musan_augmenter
        elif noise_kind in ("white", "white_mild"):
            augmenter = white_augmenter
        elif noise_kind == "musan+white":
            augmenter = RandomChoiceAugment([musan_augmenter, white_augmenter])
        else:
            raise ValueError(f"Unsupported noise_kind: {{noise_kind}}")

        precompute_split(
            split_name="{config.train_noise.train_noise_root_name}",
            wav_root=Path(c.TRAIN_ROOT),
            feat_root=feat_root,
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

    extra_root_assignments = []
    if config.train_noise.feature_mode == "clean+white":
        extra_root_assignments.append(
            f'c.TRAIN_WHITE_FEAT_ROOT = c.PRECOMPUTED_ROOT / "{train_noise_root_name}"'
        )
    elif config.train_noise.feature_mode == "clean+white_mild":
        extra_root_assignments.append(
            f'c.TRAIN_WHITE_MILD_FEAT_ROOT = c.PRECOMPUTED_ROOT / "{train_noise_root_name}"'
        )
    elif config.train_noise.feature_mode == "clean+musan+white":
        extra_root_assignments.append(
            f'c.TRAIN_MUSAN_WHITE_FEAT_ROOT = c.PRECOMPUTED_ROOT / "{train_noise_root_name}"'
        )

    extra_root_assignments_str = "\n".join(extra_root_assignments)
    code = dedent(
        f"""
        import sys
        import constants as c

        {noise_root_assignment}
        {extra_root_assignments_str}
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


def main() -> None:
    args = parse_args()

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
                f"noise_kind={config.train_noise.noise_kind} | "
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
            verify_failed = False
            for checkpoint_type in ("best", "last"):
                print(
                    f"Running verification for {config.run_name} "
                    f"[{checkpoint_type}] on {', '.join(args.eval_splits)}..."
                )
                verify_result = verify_experiment(
                    config,
                    args.eval_splits,
                    checkpoint_type=checkpoint_type,
                )
                if verify_result.returncode != 0:
                    print("Verification failed. Stopping.")
                    verify_failed = True
                    break
            if verify_failed:
                break

        if i < len(experiments) and args.cooldown_seconds > 0:
            print(f"Cooling CPU for {args.cooldown_seconds} seconds...")
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()
