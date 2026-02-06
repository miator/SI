import torch
import torchaudio


class MFCCExtraction:
    def __init__(self, sample_rate: int, n_mfcc: int, n_mels: int,
                 n_fft: int, win_length: int, hop_length: int,
                 f_min: float, f_max: float):
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,  # MFCC-level parameter
            n_mfcc=n_mfcc,            # MFCC-level parameter
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": f_min,
                "f_max": f_max,
                "center": True,
                "power": 2.0
            },                        # parameters for the internal MelSpectrogram
        )

    # @torch.inference_mode() decorator disables gradient tracking, faster, less memory, efficient
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # wav: (T, ) float
        x = wav.unsqueeze(0)  # (1, T)
        mfcc = self.mfcc(x)  # (1, n_mfcc, frames)
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # (frames, n_mfcc)
        return mfcc


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
        # wav: (N, ) preferred; also accept (1, N) or (C, N)
        if wav.dim() == 2:  # average to mono and squeeze
            wav = wav.mean(dim=0)
        wav = wav.to(torch.float32)

        m = self.mel(wav)  # (n_mels, frames)
        x = torch.log(m + self.eps)  # convert to log
        x = (x - x.mean()) / (x.std() + self.eps)  # per-utterance normalization CMVN

        return x.transpose(0, 1)  # (frames, n_mels)
