import math
import random
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

import constants as c


def scan_noise_files(
    noise_root,
    sample_rate: int,
    min_noise_seconds: float = 3.0,
) -> List[Path]:
    noise_root = Path(noise_root)
    min_noise_samples = int(round(sample_rate * min_noise_seconds))

    if not noise_root.exists():
        raise FileNotFoundError(f"MUSAN noise root does not exist: {noise_root}")

    valid_paths: List[Path] = []
    for path in sorted(noise_root.rglob("*.wav")):
        try:
            info = torchaudio.info(str(path))
        except RuntimeError:
            continue

        if info.sample_rate != sample_rate:
            continue
        if info.num_frames < min_noise_samples:
            continue
        valid_paths.append(path)

    return valid_paths


def split_noise_paths(
    noise_paths: List[Path],
    train_fraction: float = c.NOISE_TRAIN_FILES_FRACTION,
    seed: int = c.NOISE_SPLIT_SEED,
) -> tuple[List[Path], List[Path]]:
    if not noise_paths:
        return [], []

    rng = random.Random(seed)
    shuffled = list(noise_paths)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    split_idx = int(round(len(shuffled) * train_fraction))
    split_idx = max(1, min(len(shuffled) - 1, split_idx))
    return shuffled[:split_idx], shuffled[split_idx:]


class AdditiveNoise:
    def __init__(
        self,
        sample_rate: int,
        noise_root=None,
        prob: float = 0.5,
        snr_min: float = 10.0,
        snr_max: float = 20.0,
        min_noise_seconds: float = 3.0,
        noise_paths: Optional[List[Path]] = None,
        eps: float = 1e-8,
    ):
        self.noise_root = Path(noise_root) if noise_root is not None else None
        self.sample_rate = sample_rate
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.min_noise_seconds = min_noise_seconds
        self.eps = eps

        if noise_paths is not None:
            self.noise_paths = [Path(path) for path in noise_paths]
        elif self.noise_root is not None:
            self.noise_paths = scan_noise_files(
                self.noise_root,
                sample_rate=self.sample_rate,
                min_noise_seconds=self.min_noise_seconds,
            )
        else:
            self.noise_paths = []

        if not self.noise_paths:
            raise RuntimeError(
                "At least one external noise source must be enabled."
            )

    def _load_noise(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} for noise file {path}")
        if wav.dim() != 2:
            raise ValueError(f"Expected 2D tensor from torchaudio.load for {path}, got {wav.dim()}D")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).to(torch.float32)

    def _fit_noise_length(self, noise: torch.Tensor, target_length: int) -> torch.Tensor:
        noise_length = noise.numel()
        if noise_length == target_length:
            return noise

        if noise_length < target_length:
            repeats = math.ceil(target_length / max(noise_length, 1))
            noise = noise.repeat(repeats)
            noise_length = noise.numel()

        max_offset = noise_length - target_length
        start = random.randint(0, max_offset) if max_offset > 0 else 0
        return noise[start:start + target_length]

    def _sample_noise(self, target_length: int) -> torch.Tensor:
        path = random.choice(self.noise_paths)
        noise = self._load_noise(path)
        return self._fit_noise_length(noise, target_length)

    def _mix(self, speech: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        snr_db = random.uniform(self.snr_min, self.snr_max)

        speech_rms = speech.pow(2).mean().sqrt().clamp_min(self.eps)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(self.eps)
        desired_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (desired_noise_rms / noise_rms)

        mixed = speech + scaled_noise
        return mixed.clamp_(-1.0, 1.0)

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        speech = speech.to(torch.float32)
        if random.random() > self.prob:
            return speech.clamp(-1.0, 1.0)

        noise = self._sample_noise(target_length=speech.numel())
        return self._mix(speech, noise)


class WhiteNoise:
    def __init__(
        self,
        prob: float = 1.0,
        snr_min: float = 10.0,
        snr_max: float = 20.0,
        eps: float = 1e-8,
    ):
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.eps = eps

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        speech = speech.to(torch.float32)
        if random.random() > self.prob:
            return speech.clamp(-1.0, 1.0)

        noise = torch.randn_like(speech)
        snr_db = random.uniform(self.snr_min, self.snr_max)

        speech_rms = speech.pow(2).mean().sqrt().clamp_min(self.eps)
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(self.eps)
        desired_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise * (desired_noise_rms / noise_rms)

        mixed = speech + scaled_noise
        return mixed.clamp_(-1.0, 1.0)


class RandomChoiceAugment:
    def __init__(self, augmenters):
        self.augmenters = list(augmenters)
        if not self.augmenters:
            raise ValueError("augmenters must not be empty")

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        augmenter = random.choice(self.augmenters)
        return augmenter(speech)
