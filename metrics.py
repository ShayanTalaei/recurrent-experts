from typing import List
import numpy as np
import torch

class Metrics:
    # __slots__ = "num_compositions", "num_chains", "composition_total_loss", "composition_total_samples", "confusion", "scores", "winners", "composition_scores_D"

    def __init__(self, num_compositions: int, num_chains: int) -> None:
        self.num_compositions = num_compositions
        self.num_chains = num_chains
        self.composition_total_loss: List[float] = [0 for i in range(num_compositions)]
        self.composition_total_samples: List[int] = [0 for i in range(num_compositions)]
        self.composition_scores_D: List[float] = [0 for i in range(num_compositions)]
        self.scores = [[] for _ in range(num_chains)]
        self.losses = None
        self.winners: np.ndarray = np.zeros((num_chains, num_compositions))

    def reset(self) -> None:
        for i in range(self.num_compositions):
            self.composition_total_samples[i] = 0
            self.composition_total_loss[i] = 0

    def flush(self) -> None:
        self.reset()

    def get_loss_desc(self):
        text = ""
        for i in range(self.num_compositions):
            loss = self.composition_total_loss[i] / self.composition_total_samples[i] if self.composition_total_samples[i] else -1
            text += f"Composition {i+1} loss: {round(loss, 4)}\n"
        return text

    def get_score_desc(self):
        text = ""
        for i in range(self.num_compositions):
            score = self.composition_scores_D[i] / self.composition_total_samples[i] if self.composition_total_samples[i] else -1
            text += f"Composition {i+1} score: {round(score, 4)}\n"
        return text

    def append_scores(self, comp_scores, chosen_chain_index):
      # comp_scores: (batch_size, num_compositions)
      # chosen_chain_index: (batch_size), entries ranging from 0 to num_chains-1
        for ind in range(self.num_chains):
            chain_indices = torch.where(chosen_chain_index == ind)[0]
            avg_scores = torch.mean(comp_scores[chain_indices], axis=0)
            self.scores[ind].append(avg_scores.detach().cpu().numpy())
      
