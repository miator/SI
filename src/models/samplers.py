import random
from collections import defaultdict
from typing import Iterator
from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """Samples batches with P speakers and K utterances per speaker (batch size = P*K)."""

    def __init__(self, labels: list[int], P: int, K: int, seed: int = 37):
        super().__init__()
        self.labels = list(map(int, labels))
        self.P = int(P)
        self.K = int(K)
        self.seed = int(seed)

        self.by_spk = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.by_spk[y].append(idx)

        self.speakers = sorted(self.by_spk.keys())
        for spk in self.speakers:
            if len(self.by_spk[spk]) < self.K:
                raise ValueError(f"Speaker {spk} has {len(self.by_spk[spk])} samples, need K={self.K}")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        pools = {spk: self.by_spk[spk][:] for spk in self.speakers}  # shallow copy
        for spk in self.speakers:
            rng.shuffle(pools[spk])

        next_idx_per_spk = {spk: 0 for spk in self.speakers}

        buffer = []
        while True:
            speaker_order = self.speakers[:]
            rng.shuffle(speaker_order)
            chosen = speaker_order[: self.P]

            made_any = False
            for spk in chosen:
                c = next_idx_per_spk[spk]
                if c + self.K <= len(pools[spk]):
                    buffer.extend(pools[spk][c: c + self.K])
                    next_idx_per_spk[spk] += self.K
                    made_any = True

            if not made_any:
                break

            if len(buffer) >= self.P * self.K:
                out = buffer[: self.P * self.K]
                buffer = buffer[self.P * self.K:]
                yield from out

    def __len__(self) -> int:
        # approximate number of yielded indices (used by DataLoader for length/progress)
        per_spk = [len(v) // self.K for v in self.by_spk.values()]
        steps = sum(per_spk) // self.P
        return steps * self.P * self.K
