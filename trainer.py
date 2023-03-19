
import torch
from torch.utils.data import DataLoader, Dataset

from model import Expert, Discriminator

from tqdm.notebook import tqdm


def initialize_expert(
    epochs: int,
    architecture_name: str,
    expert: Expert,
    i: int,
    loss,
    data_train: DataLoader,
    device: str,
    checkpt_dir: str
) -> None:
    print("Initializing expert [{}] as identity on preturbed data".format(i+1))
    expert.train()

    epoch_tqdm = tqdm(range(epochs), desc="Epoch")
    for epoch in epoch_tqdm:
        total_loss = 0
        n_samples = 0
        data_tqdm = tqdm(data_train, desc="Batch")
        for batch in data_tqdm:
            x_canonical, x_transf = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            x_transf = x_transf.view(x_transf.size(0), -1).to(device)
            x_hat = expert(x_transf)
            loss_rec = loss(x_hat, x_transf)
            total_loss += loss_rec.item()*batch_size
            expert.optimizer.zero_grad()
            loss_rec.backward()
            expert.optimizer.step()

            # Loss
            mean_loss = total_loss/n_samples
            data_tqdm.set_description(f"initialization epoch [{epoch+1}] expert [{i+1}] loss {mean_loss:.4f}")

        if mean_loss < 0.002:
            break

    # torch.save(expert.state_dict(), checkpt_dir + f'/{name}_E_{i + 1}_init.pth')


def train_system(epoch, experts, discriminator, criterion, data_train, input_size, device):
    discriminator.train()
    for i, expert in enumerate(experts):
        expert.train()
    num_experts = len(experts)

    # Labels for canonical vs transformed samples
    canonical_label = 1.0
    transformed_label = 0.0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [[0 for i in range(num_experts)] for _ in range(num_experts)]
    total_samples_expert = [[0 for i in range(num_experts)] for _ in range(num_experts)]
    total_expert_scores_D = [[0 for _ in range(num_experts)] for _ in range(num_experts)]
    # expert_winning_samples_idx = [[[] for i in range(num_experts)] for _ in range(num_experts)]

    # Iterate through data
    for idx, batch in enumerate(data_train):
        x_canon, x_transf = batch
        # x_transf = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
        batch_size = x_canon.size(0)
        n_samples += batch_size
        x_canon = x_canon.view(batch_size, -1).to(device)
        x_transf = x_transf.view(batch_size, -1).to(device)

        # Train Discriminator on canonical distribution
        scores_canon = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label, device=device).unsqueeze(dim=1)
        loss_D_canon = criterion(scores_canon, labels)
        total_loss_D_canon += loss_D_canon.item() * batch_size
        discriminator.optimizer.zero_grad()
        loss_D_canon.backward()

        # # Train Discriminator on experts output
        labels.fill_(transformed_label)
        loss_D_transformed = 0
        exp_outputs = []
        exp_scores = torch.zeros(num_experts, num_experts, batch_size, 1).to(device)
        for i, expert_1 in enumerate(experts):
            exp_outputs.append([])
            e1_out = expert_1(x_transf)
            for j, expert_2 in enumerate(experts):
                e2_out = expert_2(e1_out)
                exp_outputs[i].append(e2_out.view(batch_size, 1, input_size))
                score = discriminator(e2_out.detach())
                exp_scores[i, j] = score
                loss_D_transformed += criterion(score, labels)
            exp_outputs[i] = torch.stack(exp_outputs[i], dim=0)
        exp_outputs = torch.stack(exp_outputs, dim=0)
        loss_D_transformed = loss_D_transformed / (num_experts*num_experts)
        total_loss_D_transformed += loss_D_transformed.item() * batch_size
        loss_D_transformed.backward()
        discriminator.optimizer.step()

        # Train experts
        exp_scores = exp_scores.reshape(-1, batch_size).detach()
        flat_indexes = exp_scores.argmax(0)
        mask_winners = torch.tensor([divmod(idx.item(), num_experts) for idx in flat_indexes]).detach()

        # Update each expert on samples it won
        for expert in experts:
            expert.optimizer.zero_grad()
        for i, expert_1 in enumerate(experts):
            for j, expert_2 in enumerate(experts):
                winning_indices = mask_winners.eq(torch.tensor([i, j])).all(dim=1)
                winning_samples = exp_outputs[i, j, winning_indices.detach()]
                accrue = 0 if idx == 0 else 1
                # expert_winning_samples_idx[i][j] += (winning_indices+accrue*n_samples).tolist()
                won_samples_count = winning_samples.size(0)
                if won_samples_count > 0:
                    total_samples_expert[i][j] += won_samples_count
                    D_E_x_transf = discriminator(winning_samples)
                    labels = torch.full((won_samples_count,), canonical_label,
                                        device=device).unsqueeze(dim=1)
                    loss_E = criterion(D_E_x_transf, labels)
                    total_loss_expert[i][j] += loss_E.item() * won_samples_count
                    loss_E.backward(retain_graph=True) # TODO figure out why retain graph is necessary
                    total_expert_scores_D[i][j] += D_E_x_transf.squeeze().sum().item()
        for expert in experts:
            expert.optimizer.step()
        # Logging
        mean_loss_D_generated = total_loss_D_transformed / n_samples
        mean_loss_D_canon = total_loss_D_canon / n_samples
        print("epoch [{}] loss_D_transformed {:.4f}".format(epoch + 1, mean_loss_D_generated))
        print("epoch [{}] loss_D_canon {:.4f}".format(epoch + 1, mean_loss_D_canon))
        # writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch + 1)
        # writer.add_scalar('loss_D_transformed', mean_loss_D_generated, epoch + 1)
        for i in range(num_experts):
            for j in range(num_experts):
                # print("epoch [{}] expert [{}][{}] n_samples {}".format(epoch + 1, i + 1, j + 1, total_samples_expert[i][j]))
                # writer.add_scalar('expert_{}_n_samples'.format(i + 1), total_samples_expert[i], epoch + 1)
                # writer.add_text('expert_{}_winning_samples'.format(i + 1),
                #                    ":".join([str(j) for j in expert_winning_samples_idx[i]]), epoch + 1)
                if total_samples_expert[i][j] > 0:
                    mean_loss_expert = total_loss_expert[i][j] / total_samples_expert[i][j]
                    mean_expert_scores = total_expert_scores_D[i][j] / total_samples_expert[i][j]
                    # print("epoch [{}] expert [{}][{}] loss {:.4f}".format(epoch + 1, i + 1, j + 1, mean_loss_expert))
                    # print("epoch [{}] expert [{}][{}] scores {:.4f}".format(epoch + 1, i + 1, j + 1, mean_expert_scores))
                    # writer.add_scalar('expert_{}_loss'.format(i + 1), mean_loss_expert, epoch + 1)
                    # writer.add_scalar('expert_{}_scores'.format(i + 1), mean_expert_scores, epoch + 1)
