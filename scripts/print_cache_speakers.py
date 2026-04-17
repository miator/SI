from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOTS = [
    Path(
        # r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\logmel_cache\train-clean-360"
    ),
    Path(
        r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\logmel_cache\train-other-500"
    ),
    # Path(
    #     r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_clean100"
    # ),
    Path(
        # r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_noise_snr20"
    ),
    # Path(
    #     r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_white_snr20-25"
    # ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print speaker IDs found in one or more log-mel cache roots."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Cache roots to scan. Defaults to the LibriSpeech cache roots used in this project.",
    )
    return parser.parse_args()


def collect_speaker_ids(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Cache root is not a directory: {root}")

    speaker_dirs = sorted(
        child.name for child in root.iterdir() if child.is_dir() and any(child.iterdir())
    )
    if speaker_dirs:
        return speaker_dirs

    speaker_ids = {
        path.stem.split("-", 1)[0]
        for path in root.rglob("*.pt")
        if "-" in path.stem
    }
    return sorted(speaker_ids)


def print_speakers(root: Path, speaker_ids: list[str]) -> None:
    print(f"Root: {root}")
    print(f"Speakers ({len(speaker_ids)}):")

    if not speaker_ids:
        print("  <none found>")
        print()
        return

    for speaker_id in speaker_ids:
        print(f"{speaker_id}")
    print()


def summarize_items(items: list[str], limit: int = 10) -> str:
    if not items:
        return "<none>"
    if len(items) <= limit:
        return ", ".join(items)
    head = ", ".join(items[:limit])
    return f"{head}, ... (+{len(items) - limit} more)"


def print_comparison(root_to_speakers: dict[Path, list[str]]) -> None:
    if len(root_to_speakers) < 2:
        return

    print("Comparison:")
    root_items = list(root_to_speakers.items())
    first_root, first_speakers = root_items[0]
    first_set = set(first_speakers)

    for root, speakers in root_items[1:]:
        speaker_set = set(speakers)
        if speaker_set == first_set:
            print(f"- {root.name}: same speaker set as {first_root.name}")
            continue

        missing = sorted(first_set - speaker_set)
        extra = sorted(speaker_set - first_set)
        print(f"- {root.name}: differs from {first_root.name}")
        print(f"  missing vs {first_root.name} ({len(missing)}): {summarize_items(missing)}")
        print(f"  extra vs {first_root.name} ({len(extra)}): {summarize_items(extra)}")


def main() -> None:
    args = parse_args()
    root_to_speakers: dict[Path, list[str]] = {}

    for root in args.roots:
        speaker_ids = collect_speaker_ids(root)
        root_to_speakers[root] = speaker_ids
        print_speakers(root, speaker_ids)

    print_comparison(root_to_speakers)


if __name__ == "__main__":
    main()
