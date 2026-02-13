# import math
import random
from collections import defaultdict
from typing import Iterator, List

# import torch
from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """
    Samples batches with P speakers and K utterances per speaker (batch size = P*K).
    Expects dataset items to have speaker labels accessible as dataset.labels (list[int]).
    """

    def __init__(self, labels: List[int], P: int, K: int, seed: int = 37, drop_last: bool = True):
        super().__init__()
        self.labels = list(map(int, labels))
        self.P = int(P)
        self.K = int(K)
        self.seed = int(seed)
        self.drop_last = drop_last

        self.by_spk = defaultdict(list)
        for idx, y in enumerate(self.labels):
            self.by_spk[y].append(idx)

        self.speakers = sorted(self.by_spk.keys())
        for spk in self.speakers:
            if len(self.by_spk[spk]) < self.K:
                raise ValueError(f"Speaker {spk} has {len(self.by_spk[spk])} samples, need K={self.K}")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        # shuffle indices per speaker
        pools = {spk: self.by_spk[spk][:] for spk in self.speakers}
        for spk in self.speakers:
            rng.shuffle(pools[spk])

        # cursor per speaker
        curs = {spk: 0 for spk in self.speakers}

        batch = []
        while True:
            rng.shuffle(self.speakers)
            chosen = self.speakers[: self.P]

            made_any = False
            for spk in chosen:
                c = curs[spk]
                if c + self.K <= len(pools[spk]):
                    batch.extend(pools[spk][c: c + self.K])
                    curs[spk] += self.K
                    made_any = True

            if not made_any:
                break

            if len(batch) >= self.P * self.K:
                out = batch[: self.P * self.K]
                batch = batch[self.P * self.K:]
                yield from out

        # optionally drop_last=True means ignore leftovers

    def __len__(self) -> int:
        # rough lower bound; DataLoader doesn't strictly need exact
        per_spk = [len(v) // self.K for v in self.by_spk.values()]
        steps = sum(per_spk) // self.P
        return steps * self.P * self.K
