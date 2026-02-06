import torch
import torchaudio


class MFCCExtraction:
    def __init__(self, sample_rate: int, n_mfcc: int, n_mels: int,
                 n_fft: int, win_length: int, hop_length: int,
                 f_min: float, f_max: float):
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": f_min,
                "f_max": f_max,
                "center": True,
                "power": 2.0
            },
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # wav: (T, ) float
        x = wav.unsqueeze(0)  # (1, T)
        mfcc = self.mfcc(x)  # (1, n_mfcc, frames)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (frames, n_mfcc)
        return mfcc
