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
    model_name: str,
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
            x_transf = x_transf.to(device)  # view(x_transf.size(0), -1).to(device)
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
    path = checkpt_dir + f"/expert_{expert_index + 1}_{model_name}.pth"
    torch.save(expert.state_dict(), path)
    return last_loss


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
    epochs: int,
    input_size: int,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: str,
    discriminator: Discriminator,
    experts: List[Expert],
    compositions: List[Composition],
    data_train: DataLoader,
    metrics: Metrics,
) -> None:
    iteration = 0

    for epoch in tqdm(range(epochs)):
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
        num_compositions = len(compositions)

        # Iterate through data
        pbar = tqdm(data_train)
        for idx, batch in enumerate(pbar):
            iteration += 1
            x_canon, x_transf, chosen_chain_index = batch
            batch_size = x_canon.size(0)
            n_samples += batch_size
            x_canon = x_canon.to(device)
            x_transf = x_transf.to(device)

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
            comp_outputs = []
            comp_scores = []
            for i, composition in enumerate(compositions):
                comp_output = composition(x_transf)
                comp_outputs.append(comp_output)
                comp_score = discriminator(comp_output.detach().to(device))
                comp_scores.append(comp_score)
                loss_D_transformed += criterion(comp_score, labels)
            loss_D_transformed = loss_D_transformed / num_compositions
            total_loss_D_transformed += loss_D_transformed.item() * batch_size
            loss_D_transformed.backward()
            discriminator.optimizer.step()

            # Train compositions
            comp_outputs = torch.cat(comp_outputs, dim=1)
            comp_scores = torch.cat(comp_scores, dim=1)
            mask_winners = comp_scores.argmax(dim=1)
            winner_indices = mask_winners.tolist()
            metrics.log_chain_win(chosen_chain_index, winner_indices)

            # zero grad experts
            for expert in experts:
                expert.optimizer.zero_grad()

            # Update each composition on samples it won
            for i, comp in enumerate(compositions):
                winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
                n_comp_samples = winning_indexes.size(0)
                if n_comp_samples > 0:
                    metrics.log_composition_win(composition_index=i, won_samples=n_comp_samples)
                    comp_samples = comp_outputs[winning_indexes, i].unsqueeze(1)
                    D_E_x_transf = discriminator(comp_samples)
                    labels = torch.full((n_comp_samples,), canonical_label, device=device).unsqueeze(dim=1)
                    loss_E = criterion(D_E_x_transf, labels)
                    metrics.log_composition_loss(i, loss_E.item() * n_comp_samples)
                    loss_E.backward(retain_graph=True)
                    metrics.log_composition_score(i, D_E_x_transf.squeeze().sum().item())

            # step experts
            for expert in experts:
                expert.optimizer.step()

            if idx % 10 == 0:
                pbar.set_description(metrics.get_loss_desc() + metrics.get_score_desc())
                pbar.refresh()

            metrics.flush(iteration=iteration, epoch=epoch)
