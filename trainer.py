from typing import List, Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from metrics import Metrics
from model import Expert, Composition, Discriminator
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


def train_compositions_with_selector(
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


def train_compositions_without_selector(
    input_size: int,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: str,
    discriminator: Discriminator,
    experts: List[Expert],
    data_train: DataLoader,
) -> None:
    discriminator.train()
    for i, expert in enumerate(experts):
        expert.train()

    # Labels for canonical vs transformed samples
    canonical_label = 1.0
    transformed_label = 0.0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    expert_scores_D = [0 for i in range(len(experts))]
    expert_winning_samples_idx = [[] for i in range(len(experts))]
    num_experts = len(experts)

    # Iterate through data
    for idx, batch in enumerate(data_train):
        x_canon, x_transf, _ = batch
        batch_size = x_canon.size(0)
        n_samples += batch_size
        x_canon = x_canon.view(batch_size, -1).to(device)
        x_transf = x_transf.view(batch_size, -1).to(device)

        # Train Discriminator on canonical distribution
        scores_canon = discriminator(x_canon.to(device))
        labels = torch.full((batch_size,), canonical_label, device=device).unsqueeze(dim=1)
        loss_D_canon = criterion(scores_canon, labels)
        total_loss_D_canon += loss_D_canon.item() * batch_size
        discriminator.optimizer.zero_grad()
        loss_D_canon.backward()

        # Train Discriminator on experts output
        labels.fill_(transformed_label)
        loss_D_transformed = 0
        exp_outputs = []
        expert_scores = []
        for i, expert in enumerate(experts):
            exp_output = expert(x_transf)
            exp_outputs.append(exp_output.view(batch_size, 1, input_size))
            exp_scores = discriminator(exp_output.detach().to(device))
            expert_scores.append(exp_scores)
            loss_D_transformed += criterion(exp_scores, labels)
        loss_D_transformed = loss_D_transformed / num_experts
        total_loss_D_transformed += loss_D_transformed.item() * batch_size
        loss_D_transformed.backward()
        discriminator.optimizer.step()

        # Train experts
        exp_outputs = torch.cat(exp_outputs, dim=1)
        expert_scores = torch.cat(expert_scores, dim=1)
        mask_winners = expert_scores.argmax(dim=1)

        # Update each expert on samples it won
        for i, expert in enumerate(experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            accrue = 0 if idx == 0 else 1
            expert_winning_samples_idx[i] += (winning_indexes + accrue * n_samples).tolist()
            n_expert_samples = winning_indexes.size(0)
            if n_expert_samples > 0:
                total_samples_expert[i] += n_expert_samples
                exp_samples = exp_outputs[winning_indexes, i]
                D_E_x_transf = discriminator(exp_samples)
                labels = torch.full((n_expert_samples,), canonical_label, device=device).unsqueeze(dim=1)
                loss_E = criterion(D_E_x_transf, labels)
                total_loss_expert[i] += loss_E.item() * n_expert_samples
                expert.optimizer.zero_grad()
                loss_E.backward(retain_graph=True)
                expert.optimizer.step()
                expert_scores_D[i] += D_E_x_transf.squeeze().sum().item()
