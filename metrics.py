from typing import List


class Metrics:
    __slots__ = "num_compositions", "num_chains", "composition_total_loss", "composition_total_samples", "confusion"

    def __init__(self, num_compositions: int, num_chains: int) -> None:
        self.num_compositions = num_compositions
        self.num_chains = num_chains
        self.composition_total_loss: List[float] = [0 for i in range(num_compositions)]
        self.composition_total_samples: List[int] = [0 for i in range(num_compositions)]

    def reset(self) -> None:
        for i in range(self.num_compositions):
            self.composition_total_samples[i] = 0
            self.composition_total_loss[i] = 0

    def flush(self) -> None:
        self.reset()
