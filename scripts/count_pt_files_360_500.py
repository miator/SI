from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ROOTS = {
    "dev-clean": Path(
        r"C:\Users\User\Desktop\Data\Librispeech_eval_standardized_chunks_3s\logmel_cache\dev-clean"
    ),
    "dev-other": Path(
        r"C:\Users\User\Desktop\Data\Librispeech_eval_standardized_chunks_3s\logmel_cache\dev-other"
    ),
    "test-clean": Path(
        r"C:\Users\User\Desktop\Data\Librispeech_eval_standardized_chunks_3s\logmel_cache\test-clean"
    ),
    "test-other": Path(
        r"C:\Users\User\Desktop\Data\Librispeech_eval_standardized_chunks_3s\logmel_cache\test-other"
    ),
    "train-clean-360": Path(
        r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\logmel_cache\train-clean-360"
    ),
    "train-other-500": Path(
        r"C:\Users\User\Desktop\Data\librispeech_train_360_500_standardized_chunks_3s\logmel_cache\train-other-500"
    ),
    "train_clean100": Path(
        r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_clean100"
    ),
    "train_noise_snr20": Path(
        r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_noise_snr20"
    ),
    "train_white_snr20-25": Path(
        r"C:\Users\User\Desktop\Data\LibriSpeech_standardized_chunks_3s\logmel_cache\train_white_snr20-25"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the total size on disk for selected log-mel cache roots."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Optional custom cache roots. If omitted, uses the default nine dataset folders.",
    )
    return parser.parse_args()


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def get_directory_size(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Cache root is not a directory: {root}")

    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_file():
                total += path.stat().st_size
    return total


def main() -> None:
    args = parse_args()

    if args.roots:
        roots = {root.name: root for root in args.roots}
    else:
        roots = DEFAULT_ROOTS

    grand_total = 0
    for name, root in roots.items():
        size_bytes = get_directory_size(root)
        grand_total += size_bytes
        print(f"{name}/ {format_bytes(size_bytes)}")
        print(f"  path: {root}")

    print(f"total/ {format_bytes(grand_total)}")


if __name__ == "__main__":
    main()
