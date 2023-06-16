from typing import Protocol, List, Callable

import torch
from torch import Tensor

from metrics import Metrics
from model import Composition


class Selector(Protocol):
    def select_experts(
        self,
        compositions: List[Composition],
        canon: Tensor,
        transformed: Tensor,
        metrics: Metrics,
    ) -> List[Tensor]:
        ...


class GreedySelector(Selector):
    __slots__ = ("device", "criterion")

    def __init__(self, device: str, criterion: Callable[[Tensor, Tensor], Tensor]) -> None:
        self.device = device
        self.criterion = criterion

    def select_experts(
        self,
        compositions: List[Composition],
        canon: Tensor,
        transformed: Tensor,
        metrics: Metrics,
    ) -> List[Tensor]:
        batch_size = canon.size(0)

        # batch size x input_dim
        x_canon = canon.view(batch_size, -1).to(self.device)
        x_transf = transformed.view(batch_size, -1).to(self.device)

        # batch x len(compositions)
        losses = torch.full((batch_size, len(compositions)), float("inf"), device=self.device)

        # batch x 2
        for composition_index, composition in enumerate(compositions):
            # batch_size x 1
            loss = self.criterion(composition.forward(x_transf), x_canon).mean(dim=1)
            losses[:, composition_index] = loss

        # batch x 1
        winner_indices = torch.argmin(losses, dim=1)
        winner_losses = losses[:, winner_indices]
        average_composition_losses = []

        for composition_index, composition in enumerate(compositions):
            mask_winners = winner_indices == composition_index
            won_losses = winner_losses[mask_winners]
            won_samples_count = torch.sum(mask_winners).item()
            avg_loss = torch.mean(won_losses)
            metrics.composition_total_samples[composition_index] += won_samples_count
            metrics.composition_total_loss[composition_index] += avg_loss.item() * won_samples_count
            average_composition_losses.append(avg_loss)

        return average_composition_losses
