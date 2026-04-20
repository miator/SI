from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import torch
import torchaudio
from tqdm import tqdm


TARGET_SAMPLE_RATE = 16_000
DEFAULT_NO_SPEECH_THRESHOLD_SEC = 0.3
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac")
TQDM_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} files [{elapsed}<{remaining}, {rate_fmt}]"
)


@dataclass(frozen=True)
class DatasetSpec:
    display_name: str
    root: Path


@dataclass
class DatasetStats:
    total: int = 0
    no_speech: int = 0
    failed: int = 0

    @property
    def filtered_percent(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.no_speech / self.total) * 100.0


@dataclass(frozen=True)
class FireRedVadSettings:
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000


class FireRedVadDetector:
    def __init__(
        self,
        model_dir: Path,
        settings: FireRedVadSettings,
        fireredvad_root: Path | None = None,
    ) -> None:
        if fireredvad_root is not None:
            root = fireredvad_root.expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"FireRedVAD source root does not exist: {root}")
            sys.path.insert(0, str(root))

        try:
            from fireredvad import FireRedVad, FireRedVadConfig  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FireRedVAD is not available. Install it with `pip install fireredvad`, "
                "or pass `--fireredvad-root` pointing to a local FireRedVAD source checkout."
            ) from exc

        model_dir = model_dir.expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(
                "FireRedVAD model directory does not exist: "
                f"{model_dir}\nDownload/copy the VAD weights locally and pass "
                "`--fireredvad-model-dir <path-to-pretrained_models/FireRedVAD/VAD>`."
            )

        vad_config = FireRedVadConfig(
            use_gpu=settings.use_gpu,
            smooth_window_size=settings.smooth_window_size,
            speech_threshold=settings.speech_threshold,
            min_speech_frame=settings.min_speech_frame,
            max_speech_frame=settings.max_speech_frame,
            min_silence_frame=settings.min_silence_frame,
            merge_silence_frame=settings.merge_silence_frame,
            extend_speech_frame=settings.extend_speech_frame,
            chunk_max_frame=settings.chunk_max_frame,
        )
        self._vad = FireRedVad.from_pretrained(str(model_dir), vad_config)

    def speech_duration_sec(self, path: Path, temp_dir: Path | None = None) -> float:
        with prepared_fireredvad_wav(path, temp_dir=temp_dir) as wav_path:
            result, _probs = self._vad.detect(str(wav_path))

        timestamps = result.get("timestamps", [])
        return sum(
            max(0.0, float(end) - float(start))
            for start, end in timestamps
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_fireredvad_model_dir() -> Path:
    env_model_dir = os.environ.get("FIREREDVAD_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir)
    return _project_root() / "pretrained_models" / "FireRedVAD" / "VAD"


def _env_data_root_default() -> Path:
    if sys.platform == "win32":
        return Path(r"C:\Users\User\Desktop\Data")
    return Path("/workspace/data")


def _resolve_existing_path(candidates: Iterable[Path]) -> Path:
    candidates = tuple(candidates)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _candidate_roots(data_root: Path, split_name: str) -> list[Path]:
    candidates = [
        data_root / split_name,
        data_root / split_name / "LibriSpeech" / split_name,
        data_root / "LibriSpeech" / split_name,
        data_root / "wav" / split_name,
    ]

    if split_name == "train-clean-100":
        candidates.extend(
            [
                data_root / "LibriSpeech_standardized_chunks_3s" / "wav" / "train_clean100",
                data_root / "LibriSpeech_standardized_chunks_3s" / "wav" / split_name,
            ]
        )
    else:
        candidates.extend(
            [
                data_root
                / "librispeech_train_360_500_standardized_chunks_3s"
                / "wav"
                / split_name,
                data_root / "librispeech_train_360_500_standardized" / split_name,
            ]
        )

    return candidates


def default_dataset_specs(data_root: Path) -> list[DatasetSpec]:
    config_roots: dict[str, Path] = {}
    try:
        from src.config.data_config import (  # type: ignore
            TRAIN_CLEAN100_WAV_ROOT,
            TRAIN_CLEAN360_WAV_ROOT,
            TRAIN_OTHER500_WAV_ROOT,
        )

        config_roots = {
            "train-clean-100": Path(TRAIN_CLEAN100_WAV_ROOT),
            "train-clean-360": Path(TRAIN_CLEAN360_WAV_ROOT),
            "train-other-500": Path(TRAIN_OTHER500_WAV_ROOT),
        }
    except Exception:
        config_roots = {}

    specs: list[DatasetSpec] = []
    for split_name in ("train-clean-100", "train-clean-360", "train-other-500"):
        candidates = []
        if split_name in config_roots:
            candidates.append(config_roots[split_name])
        candidates.extend(_candidate_roots(data_root, split_name))
        specs.append(DatasetSpec(split_name, _resolve_existing_path(candidates)))

    return specs


def parse_extensions(raw_extensions: str) -> tuple[str, ...]:
    extensions = []
    for raw_ext in raw_extensions.split(","):
        ext = raw_ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.append(ext)
    if not extensions:
        raise ValueError("At least one audio extension is required.")
    return tuple(dict.fromkeys(extensions))


def iter_audio_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        print(f"Warning: dataset root does not exist: {root}", file=sys.stderr)
        return []

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 2:
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform[:1]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform.to(torch.float32)


def load_waveform_16k_mono(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = to_mono(waveform)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )
    return waveform


def is_fireredvad_ready_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        info = torchaudio.info(str(path))
    except Exception:
        return True
    return int(info.sample_rate) == TARGET_SAMPLE_RATE and int(info.num_channels) == 1


@contextmanager
def prepared_fireredvad_wav(
    path: Path,
    temp_dir: Path | None = None,
) -> Iterator[Path]:
    if is_fireredvad_ready_wav(path):
        yield path
        return

    waveform = load_waveform_16k_mono(path)
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=str(temp_dir) if temp_dir is not None else None,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        torchaudio.save(
            str(temp_path),
            waveform.clamp(-1.0, 1.0),
            TARGET_SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def analyze_dataset(
    spec: DatasetSpec,
    extensions: tuple[str, ...],
    detector: FireRedVadDetector,
    no_speech_threshold_sec: float,
    show_progress: bool,
    temp_dir: Path | None,
    print_no_speech: bool,
    no_speech_file: TextIO | None,
) -> DatasetStats:
    files = iter_audio_files(spec.root, extensions)
    stats = DatasetStats(total=len(files))

    iterator = (
        tqdm(
            files,
            desc=spec.display_name,
            unit="file",
            total=len(files),
            dynamic_ncols=True,
            ascii=True,
            bar_format=TQDM_BAR_FORMAT,
        )
        if show_progress
        else files
    )
    for path in iterator:
        try:
            duration = detector.speech_duration_sec(path, temp_dir=temp_dir)
        except Exception as exc:
            stats.failed += 1
            print(f"Warning: failed to analyze {path}: {exc!r}", file=sys.stderr)
            continue

        if duration < no_speech_threshold_sec:
            stats.no_speech += 1
            line = (
                f"{spec.display_name}\t{duration:.6f}\t{path}"
            )
            if print_no_speech:
                print(
                    f"NO_SPEECH Dataset: {spec.display_name} "
                    f"Speech: {duration:.3f}s "
                    f"Path: {path}",
                    flush=True,
                )
            if no_speech_file is not None:
                no_speech_file.write(f"{line}\n")
                no_speech_file.flush()

    return stats


def print_stats(spec: DatasetSpec, stats: DatasetStats) -> None:
    print(f"Dataset: {spec.display_name}")
    print(f"Total: {stats.total}")
    print(f"No speech: {stats.no_speech}")
    print(f"Filtered: {stats.filtered_percent:.2f}%")
    if stats.failed:
        print(f"Failed: {stats.failed}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LibriSpeech train splits with FireRedVAD and "
            "print the percentage of files with less than 0.3 seconds of speech."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_env_data_root_default(),
        help="Base data directory used when split-specific roots are not provided.",
    )
    parser.add_argument(
        "--train-clean-100-root",
        type=Path,
        default=None,
        help="Override root for train-clean-100 audio files.",
    )
    parser.add_argument(
        "--train-clean-360-root",
        type=Path,
        default=None,
        help="Override root for train-clean-360 audio files.",
    )
    parser.add_argument(
        "--train-other-500-root",
        type=Path,
        default=None,
        help="Override root for train-other-500 audio files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("train-clean-100", "train-clean-360", "train-other-500"),
        default=("train-clean-100", "train-clean-360", "train-other-500"),
        help="Dataset splits to analyze, in the order provided.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_AUDIO_EXTENSIONS),
        help="Comma-separated audio extensions to scan, for example .wav,.flac.",
    )
    parser.add_argument(
        "--fireredvad-root",
        type=Path,
        default=None,
        help="Optional local FireRedVAD source checkout root. Not needed if fireredvad is installed.",
    )
    parser.add_argument(
        "--fireredvad-model-dir",
        type=Path,
        default=_default_fireredvad_model_dir(),
        help="Local path to FireRedVAD VAD model directory.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run FireRedVAD on GPU.",
    )
    parser.add_argument("--speech-threshold", type=float, default=0.4)
    parser.add_argument("--smooth-window-size", type=int, default=5)
    parser.add_argument("--min-speech-frame", type=int, default=20)
    parser.add_argument("--max-speech-frame", type=int, default=2000)
    parser.add_argument("--min-silence-frame", type=int, default=20)
    parser.add_argument("--merge-silence-frame", type=int, default=0)
    parser.add_argument("--extend-speech-frame", type=int, default=0)
    parser.add_argument("--chunk-max-frame", type=int, default=30000)
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Optional directory for temporary 16kHz mono WAV conversions.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=DEFAULT_NO_SPEECH_THRESHOLD_SEC,
        help="Mark an audio file as no speech when detected speech is below this many seconds.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--print-no-speech",
        action="store_true",
        help="Print each audio path whose detected speech duration is below the no-speech threshold.",
    )
    parser.add_argument(
        "--save-no-speech-paths",
        type=Path,
        default=None,
        help="Optional text file path for no-speech rows: dataset, speech seconds, audio path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_root = args.data_root.expanduser()
    specs = default_dataset_specs(data_root)
    overrides = {
        "train-clean-100": args.train_clean_100_root,
        "train-clean-360": args.train_clean_360_root,
        "train-other-500": args.train_other_500_root,
    }
    specs = [
        DatasetSpec(spec.display_name, Path(overrides[spec.display_name]).expanduser())
        if overrides[spec.display_name] is not None
        else DatasetSpec(spec.display_name, spec.root.expanduser())
        for spec in specs
    ]
    specs_by_name = {spec.display_name: spec for spec in specs}
    specs = [specs_by_name[name] for name in args.datasets]

    extensions = parse_extensions(args.extensions)
    show_progress = not args.no_progress
    temp_dir = args.temp_dir.expanduser() if args.temp_dir is not None else None
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)

    settings = FireRedVadSettings(
        use_gpu=bool(args.use_gpu),
        smooth_window_size=int(args.smooth_window_size),
        speech_threshold=float(args.speech_threshold),
        min_speech_frame=int(args.min_speech_frame),
        max_speech_frame=int(args.max_speech_frame),
        min_silence_frame=int(args.min_silence_frame),
        merge_silence_frame=int(args.merge_silence_frame),
        extend_speech_frame=int(args.extend_speech_frame),
        chunk_max_frame=int(args.chunk_max_frame),
    )
    try:
        detector = FireRedVadDetector(
            model_dir=args.fireredvad_model_dir,
            settings=settings,
            fireredvad_root=args.fireredvad_root,
        )
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    all_stats: list[DatasetStats] = []

    no_speech_file_path = (
        args.save_no_speech_paths.expanduser()
        if args.save_no_speech_paths is not None
        else None
    )
    if no_speech_file_path is not None:
        no_speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        no_speech_file_path.open("w", encoding="utf-8")
        if no_speech_file_path is not None
        else nullcontext(None)
    ) as no_speech_file:
        if no_speech_file is not None:
            no_speech_file.write("dataset\tspeech_duration_sec\tpath\n")
            no_speech_file.flush()

        for index, spec in enumerate(specs):
            print(f"Scanning {spec.display_name}: {spec.root}", file=sys.stderr)
            stats = analyze_dataset(
                spec=spec,
                extensions=extensions,
                detector=detector,
                no_speech_threshold_sec=float(args.no_speech_threshold),
                show_progress=show_progress,
                temp_dir=temp_dir,
                print_no_speech=bool(args.print_no_speech),
                no_speech_file=no_speech_file,
            )
            if index:
                print()
            print_stats(spec, stats)
            all_stats.append(stats)

    overall = DatasetStats(
        total=sum(stats.total for stats in all_stats),
        no_speech=sum(stats.no_speech for stats in all_stats),
        failed=sum(stats.failed for stats in all_stats),
    )
    print()
    print("Dataset: overall")
    print(f"Total: {overall.total}")
    print(f"No speech: {overall.no_speech}")
    print(f"Filtered: {overall.filtered_percent:.2f}%")
    if overall.failed:
        print(f"Failed: {overall.failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
