from typing import List, Any, Dict

import numpy as np
import torch
import wandb
from torch import Tensor


class Metrics:
    def __init__(
        self,
        composition_names: List[str],
        chain_names: List[str],
    ) -> None:
        self.composition_names = composition_names
        self.chain_names = chain_names
        self.num_compositions = len(composition_names)
        self.num_chains = len(chain_names)
        self.composition_loss: List[float] = [0 for i in range(self.num_compositions)]
        self.composition_won_samples: List[int] = [
            1 for i in range(self.num_compositions)
        ]
        self.composition_scores_D: List[float] = [
            0 for i in range(self.num_compositions)
        ]
        self.scores: List[List[np.ndarray]] = [[] for _ in range(self.num_chains)]
        self.winners: np.ndarray = np.zeros((self.num_chains, self.num_compositions))

    def reset(self) -> None:
        for i in range(self.num_compositions):
            self.composition_won_samples[i] = 0
            self.composition_loss[i] = 0

    def flush(self, iteration: int, epoch: int) -> None:
        wandb.log(self._get_wandb_log(epoch, iteration))

    def get_loss_desc(self):
        text = ""
        for i in range(self.num_compositions):
            loss = (
                self.composition_loss[i] / self.composition_won_samples[i]
                if self.composition_won_samples[i]
                else -1
            )
            text += f"Composition {i + 1} loss: {round(loss, 4)}\n"
        return text

    def get_score_desc(self):
        text = ""
        for i in range(self.num_compositions):
            score = (
                self.composition_scores_D[i] / self.composition_won_samples[i]
                if self.composition_won_samples[i]
                else -1
            )
            text += f"Composition {i + 1} score: {round(score, 4)}\n"
        return text

    def log_chain_win(self, chosen_chain_index: Tensor, winner_indices: List[int]) -> None:
        self.winners[chosen_chain_index, winner_indices] += 1

    def log_composition_win(self, composition_index: int, won_samples: int) -> None:
        self.composition_won_samples[composition_index] += won_samples

    def log_composition_loss(self, composition_index: int, loss_amount: float) -> None:
        self.composition_loss[composition_index] += loss_amount

    def log_composition_score(self, composition_index: int, score: float) -> None:
        self.composition_scores_D[composition_index] += score

    def append_scores(self, comp_scores, chosen_chain_index):
        # comp_scores: (batch_size, num_compositions)
        # chosen_chain_index: (batch_size), entries ranging from 0 to num_chains-1
        for ind in range(self.num_chains):
            chain_indices = torch.where(chosen_chain_index == ind)[0]
            avg_scores = torch.mean(comp_scores[chain_indices], axis=0)
            self.scores[ind].append(avg_scores.detach().cpu().numpy())

    def _get_wandb_log(self, epoch: int, iteration: int) -> Dict[str, Any]:
        """
        self.composition_total_loss: List[float] = [0 for i in range(num_compositions)]
        self.composition_total_samples: List[int] = [0 for i in range(num_compositions)]
        self.composition_scores_D: List[float] = [0 for i in range(num_compositions)]
        self.scores: List[List[np.ndarray]] = [[] for _ in range(num_chains)]
        self.winners: np.ndarray = np.zeros((num_chains, num_compositions))
        """
        log = {
            "iteration": iteration,
            "epoch": epoch,
        }

        for comp_index in range(self.num_compositions):
            log[f"{self.composition_names[comp_index]}_loss"] = (
                self.composition_loss[comp_index]
                / self.composition_won_samples[comp_index]
            )

        return log
