import torch
import torchaudio


class LogMelExtraction:
    def __init__(self, sample_rate: int, n_fft: int,
                 win_length: int, hop_length: int, n_mels: int,
                 f_min: float, f_max: float, eps: float = 1e-6):
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    @torch.inference_mode()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        wav = wav.to(torch.float32)

        mel = self.mel(wav)  # (n_mels, frames)
        logmel = torch.log(mel + self.eps)
        logmel = (logmel - logmel.mean()) / (logmel.std() + self.eps)

        return logmel.transpose(0, 1)  # (frames, n_mels)
