import random
from pathlib import Path
from typing import Optional

import torch
import torchaudio


def scan_noise_files(
    noise_root,
    sample_rate: int,
) -> list[Path]:
    noise_root = Path(noise_root)

    if not noise_root.exists():
        raise FileNotFoundError(f"Noise root does not exist: {noise_root}")

    valid_paths: list[Path] = []
    for path in sorted(noise_root.rglob("*.wav")):
        try:
            info = torchaudio.info(str(path))
        except RuntimeError:
            continue

        if info.sample_rate != sample_rate:
            continue
        valid_paths.append(path)

    return valid_paths


def _mix_with_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    *,
    snr_min: float,
    snr_max: float,
    eps: float,
) -> torch.Tensor:
    snr_db = random.uniform(snr_min, snr_max)

    speech_rms = speech.pow(2).mean().sqrt().clamp_min(eps)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(eps)
    desired_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / noise_rms)

    mixed = speech + scaled_noise
    return mixed.clamp_(-1.0, 1.0)


class AdditiveNoise:
    def __init__(
        self,
        sample_rate: int,
        noise_root=None,
        prob: float = 1.0,
        snr_min: float = 20.0,
        snr_max: float = 20.0,
        noise_paths: Optional[list[Path]] = None,
        eps: float = 1e-8,
    ):
        self.noise_root = Path(noise_root) if noise_root is not None else None
        self.sample_rate = sample_rate
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.eps = eps

        if noise_paths is not None:
            self.noise_paths = [Path(path) for path in noise_paths]
        elif self.noise_root is not None:
            self.noise_paths = scan_noise_files(
                self.noise_root,
                sample_rate=self.sample_rate)
        else:
            self.noise_paths = []

        if not self.noise_paths:
            raise RuntimeError(
                "At least one external noise source must be enabled.")

    def _load_noise(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            raise ValueError(f"Unexpected sample rate {sr} for noise file {path}")
        if wav.dim() != 2:
            raise ValueError(f"Expected 2D tensor from torchaudio.load for {path}, got {wav.dim()}D")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).to(torch.float32)

    def _crop_noise_length(self, noise: torch.Tensor, target_length: int) -> torch.Tensor:
        noise_length = noise.numel()
        if noise_length == target_length:
            return noise

        if noise_length < target_length:
            raise ValueError(
                f"Noise chunk is shorter than target speech length: "
                f"noise={noise_length} samples, speech={target_length} samples"
            )

        max_offset = noise_length - target_length
        start = random.randint(0, max_offset) if max_offset > 0 else 0
        return noise[start:start + target_length]

    def _sample_noise(self, target_length: int) -> torch.Tensor:
        path = random.choice(self.noise_paths)
        noise = self._load_noise(path)
        return self._crop_noise_length(noise, target_length)

    def _mix(self, speech: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _mix_with_snr(
            speech,
            noise,
            snr_min=self.snr_min,
            snr_max=self.snr_max,
            eps=self.eps)

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        speech = speech.to(torch.float32)
        if random.random() > self.prob:
            return speech.clamp(-1.0, 1.0)

        noise = self._sample_noise(target_length=speech.numel())
        return self._mix(speech, noise)


class WhiteNoise:
    def __init__(
        self,
        sample_rate: int,
        prob: float = 1.0,
        snr_min: float = 20.0,
        snr_max: float = 20.0,
        eps: float = 1e-8,
    ):
        self.sample_rate = sample_rate
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.eps = eps

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        speech = speech.to(torch.float32)
        if random.random() > self.prob:
            return speech.clamp(-1.0, 1.0)

        return _mix_with_snr(
            speech,
            torch.randn_like(speech),
            snr_min=self.snr_min,
            snr_max=self.snr_max,
            eps=self.eps)


def build_waveform_augmenter(
    kind: str,
    sample_rate: int,
    prob: float = 1.0,
    snr_min: float = 20.0,
    snr_max: float = 20.0,
    noise_root=None,
    noise_paths: Optional[list[Path]] = None,
    eps: float = 1e-8,
):
    if kind == "noise":
        return AdditiveNoise(
            sample_rate=sample_rate,
            noise_root=noise_root,
            prob=prob,
            snr_min=snr_min,
            snr_max=snr_max,
            noise_paths=noise_paths,
            eps=eps)
    if kind == "white":
        return WhiteNoise(
            sample_rate=sample_rate,
            prob=prob,
            snr_min=snr_min,
            snr_max=snr_max,
            eps=eps)
    raise ValueError(f"Unsupported waveform augmenter kind: {kind}")
