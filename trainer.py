
import torch
from torch.utils.data import DataLoader, Dataset

from model import Expert, Discriminator

from tqdm import tqdm


def initialize_expert(
    epochs: int,
    architecture_name: str,
    expert: Expert,
    i: int,
    optimizer,
    loss,
    data_train: DataLoader,
    device: str,
    checkpt_dir: str,
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
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # Loss
        mean_loss = total_loss/n_samples
        print("initialization epoch [{}] expert [{}] loss {:.4f}".format(epoch+1, i+1, mean_loss))
        if mean_loss < 0.002:
            break

    # torch.save(expert.state_dict(), checkpt_dir + f'/{name}_E_{i + 1}_init.pth')


def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args, writer):
    discriminator.train()
    for i, expert in enumerate(experts):
        expert.train()

    # Labels for canonical vs transformed samples
    canonical_label = 1
    transformed_label = 0

    # Keep track of losses
    total_loss_D_canon = 0
    total_loss_D_transformed = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    expert_scores_D = [0 for i in range(len(experts))]
    expert_winning_samples_idx = [[] for i in range(len(experts))]

    # Iterate through data
    for idx, batch in enumerate(data_train):
        x_canon, x_transf = batch
        # x_transf = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
        batch_size = x_canon.size(0)
        n_samples += batch_size
        x_canon = x_canon.view(batch_size, -1).to(args.device)
        x_transf = x_transf.view(batch_size, -1).to(args.device)

        # Train Discriminator on canonical distribution
        scores_canon = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label, device=args.device).unsqueeze(dim=1)
        loss_D_canon = criterion(scores_canon, labels)
        total_loss_D_canon += loss_D_canon.item() * batch_size
        optimizer_D.zero_grad()
        loss_D_canon.backward()

        # # Train Discriminator on experts output
        # labels.fill_(transformed_label)
        # loss_D_transformed = 0
        # exp_outputs = []
        # expert_scores = []
        # for i, expert in enumerate(experts):
        #     exp_output = expert(x_transf)
        #     exp_outputs.append(exp_output.view(batch_size, 1, args.input_size))
        #     exp_scores = discriminator(exp_output.detach())
        #     expert_scores.append(exp_scores)
        #     loss_D_transformed += criterion(exp_scores, labels)
        # loss_D_transformed = loss_D_transformed / args.num_experts
        # total_loss_D_transformed += loss_D_transformed.item() * batch_size
        # loss_D_transformed.backward()
        # optimizer_D.step()

        # # Train experts
        # exp_outputs = torch.cat(exp_outputs, dim=1)
        # expert_scores = torch.cat(expert_scores, dim=1)
        # mask_winners = expert_scores.argmax(dim=1)

        # Update each expert on samples it won
        # for i, expert in enumerate(experts):
        #     winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
        #     accrue = 0 if idx == 0 else 1
        #     expert_winning_samples_idx[i] += (winning_indexes+accrue*n_samples).tolist()
        #     n_expert_samples = winning_indexes.size(0)
        #     if n_expert_samples > 0:
        #         total_samples_expert[i] += n_expert_samples
        #         exp_samples = exp_outputs[winning_indexes, i]
        #         D_E_x_transf = discriminator(exp_samples)
        #         labels = torch.full((n_expert_samples,), canonical_label,
        #                             device=args.device).unsqueeze(dim=1)
        #         loss_E = criterion(D_E_x_transf, labels)
        #         total_loss_expert[i] += loss_E.item() * n_expert_samples
        #         optimizers_E[i].zero_grad()
        #         loss_E.backward(retain_graph=True) # TODO figure out why retain graph is necessary
        #         optimizers_E[i].step()
        #         expert_scores_D[i] += D_E_x_transf.squeeze().sum().item()

    # Logging
    # mean_loss_D_generated = total_loss_D_transformed / n_samples
    # mean_loss_D_canon = total_loss_D_canon / n_samples
    # print("epoch [{}] loss_D_transformed {:.4f}".format(epoch + 1, mean_loss_D_generated))
    # print("epoch [{}] loss_D_canon {:.4f}".format(epoch + 1, mean_loss_D_canon))
    # writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch + 1)
    # writer.add_scalar('loss_D_transformed', mean_loss_D_generated, epoch + 1)
    # for i in range(len(experts)):
    #     print("epoch [{}] expert [{}] n_samples {}".format(epoch + 1, i + 1, total_samples_expert[i]))
    #     writer.add_scalar('expert_{}_n_samples'.format(i + 1), total_samples_expert[i], epoch + 1)
    #     writer.add_text('expert_{}_winning_samples'.format(i + 1),
    #                        ":".join([str(j) for j in expert_winning_samples_idx[i]]), epoch + 1)
    #     if total_samples_expert[i]> 0:
    #         mean_loss_expert = total_loss_expert[i] / total_samples_expert[i]
    #         mean_expert_scores = expert_scores_D[i] / total_samples_expert[i]
    #         print("epoch [{}] expert [{}] loss {:.4f}".format(epoch + 1, i + 1, mean_loss_expert))
    #         print("epoch [{}] expert [{}] scores {:.4f}".format(epoch + 1, i + 1, mean_expert_scores))
    #         writer.add_scalar('expert_{}_loss'.format(i + 1), mean_loss_expert, epoch + 1)
    #         writer.add_scalar('expert_{}_scores'.format(i + 1), mean_expert_scores, epoch + 1)
