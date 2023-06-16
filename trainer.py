from typing import List, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from metrics import Metrics
from model import Expert, Composition
from selector import Selector


def initialize_expert(
    epochs: int,
    expert: Expert,
    expert_index: int,
    loss: Callable[[Tensor, Tensor], Tensor],
    train_loader: DataLoader,
    device: torch.device,
    checkpt_dir: str,
) -> None:
    print("Initializing expert [{}] as identity on preturbed data".format(expert_index + 1))
    expert.train()

    for epoch in tqdm(range(epochs), desc="Epoch"):
        total_loss = 0
        n_samples = 0
        data_tqdm = tqdm(train_loader, desc="Batch")
        last_loss = np.inf
        for batch in data_tqdm:
            x_canonical, x_transf, _ = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            x_transf = x_transf.view(x_transf.size(0), -1).to(device)
            x_hat = expert(x_transf)
            loss_rec = loss(x_hat, x_transf)
            total_loss += loss_rec.item() * batch_size
            expert.optimizer.zero_grad()
            loss_rec.backward()
            expert.optimizer.step()

            # Loss
            last_loss = total_loss / n_samples
            data_tqdm.set_description(
                f"initialization epoch [{epoch + 1}] expert [{expert_index + 1}] loss {last_loss:.4f}"
            )

        if last_loss < 0.002:
            break

    torch.save(expert.state_dict(), checkpt_dir + f"/expert_{expert_index + 1}_init.pth")


def train_compositions(
    compositions: List[Composition],
    experts: List[Expert],
    selector: Selector,
    data_train: DataLoader,
    metrics: Metrics,
) -> None:
    for i, expert in enumerate(experts):
        expert.train()

    # Iterate through data
    for idx, batch in enumerate(data_train):
        canon, transformed, chain_indices = batch
        # reset expert gradients
        for expert in experts:
            expert.optimizer.zero_grad()

        losses = selector.select_experts(compositions, canon, transformed, metrics)
        for composition_index, composition_loss in enumerate(losses):
            composition_loss.backward(retain_graph=True)

        for expert in experts:
            expert.optimizer.step()

    metrics.flush()
