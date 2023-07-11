import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import wandb

from dataset import *
from model import *
from optim import *
from trainer import *


def init_weights(model: Expert, path: str) -> None:
    pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)

    for layer in pre_trained_dict.keys():
        model.state_dict()[layer].copy_(pre_trained_dict[layer])

    for param in model.parameters():
        param.requires_grad = True

def show_img(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    plt.show()


def show_expert_demo(experts, dataset: ChainedMNIST, sample_index: int, chain: Chain, device) -> None:
    canon, _, _ = dataset[sample_index]
    print(canon.shape)
    transformed = chain.forward(canon)
    size = canon.numel()
    width = int(size**0.5)
    print(width)
    num_experts = len(experts)

    f, axarr = plt.subplots((num_experts + 2) // 2, 2)
    f.tight_layout()
    axarr[0, 0].imshow(canon.reshape(width, width).cpu().detach().numpy())
    axarr[0, 0].text(2, width + 10 * (num_experts // 4), "Canon")

    axarr[0, 1].imshow(transformed.reshape(width, width).cpu().detach().numpy())
    axarr[0, 1].text(2, width + 10 * (num_experts // 4), f"Trans: {chain.abbreviation}")

    transformed = transformed.to(device).unsqueeze(0)

    train_status = [e.training for e in experts]
    for e in experts:
        e.train(False)

    for i, expert in enumerate(experts):
        axarr[i // 2 + 1, i % 2].imshow(experts[0](transformed).reshape(width, width).cpu().detach().numpy())
        axarr[i // 2 + 1, i % 2].text(2, width + 10 * (num_experts // 4), f"Expert - {i + 1}")

    for i, e in enumerate(experts):
        e.train(train_status[i])


def show_compound(
    dataset: ChainedMNIST,
    sample: int,
    chains: list[Chain],
    compositions: list[Composition],
    device,
    name=None,
) -> None:
    canon, trans, chain_index = dataset[sample]
    width = int(canon.numel() ** 0.5)
    chain = chains[chain_index]
    num_compositions = len(compositions)

    f, axarr = plt.subplots((num_compositions + 2) // 2, 2)
    f.tight_layout()
    axarr[0, 0].imshow(canon.reshape(width, width))
    axarr[0, 0].text(9, width + 10 * (num_compositions // 4), "Canon")

    axarr[0, 1].imshow(trans.reshape(width, width))
    axarr[0, 1].text(5, width + 10 * (num_compositions // 4), f"Trans {chain.abbreviation}")

    trans = trans.to(device).unsqueeze(0)  # .reshape(1, 784)
    for i, comp in enumerate(compositions):
        recovered = comp.forward(trans).reshape(width, width).cpu().detach().numpy()
        axarr[i // 2 + 1, i % 2].imshow(recovered, vmin=-0.424212918, vmax=2.82148653)
        axarr[i // 2 + 1, i % 2].text(0, width + 10 * (num_compositions // 4), f"Recovered {comp.name}")
    if name is not None:
        if not os.path.exists("./results"):
            os.mkdir("./results")
        plt.savefig(f"./results/compounds_{name}.pdf")
    wandb.log({"results": f, "data_index": i})
    # plt.show()
    plt.close(f)


def create_checkpoints_dir(checkpt_dir):
    if not os.path.exists(checkpt_dir):
        os.mkdir(checkpt_dir)


def create_optimizer(params: Iterator[Parameter], t: str, lr: float, wd: float) -> Optimizer:
    if t == "adam":
        return new_adam(params, lr=lr, weight_decay=wd)
    elif t == "sgd":
        return new_sgd(params, lr=lr, weight_decay=wd)
    raise NotImplementedError(f"{t} is not a known optimizer")


def init_identity_experts(num_experts, checkpt_dir, model_name, load_initialized_experts, device, train_loader):
    from trainer import initialize_expert
    experts = []
    loss_initial = torch.nn.MSELoss(reduction="mean")
    init_optimizer = "adam"
    init_learning_rate = 0.1
    init_weight_decay = 0
    for expert_index in range(num_experts):
        if load_initialized_experts:
            expert = initialize_expert_model(model_name).to(device)
            expert.optimizer = create_optimizer(
                expert.parameters(), init_optimizer, lr=init_learning_rate, wd=init_weight_decay
            )
            path = checkpt_dir + f"/expert_{expert_index + 1}_{model_name}.pth"
            init_weights(expert, path)

        else:
            init_epochs = 3
            last_loss = 1
            while last_loss > 0.002:
                expert = initialize_expert_model(model_name).to(device)
                expert.optimizer = create_optimizer(
                    expert.parameters(), init_optimizer, lr=init_learning_rate, wd=init_weight_decay
                )

                last_loss = initialize_expert(
                    epochs=init_epochs,
                    expert=expert,
                    expert_index=expert_index,
                    loss=loss_initial,
                    train_loader=train_loader,
                    device=device,
                    model_name=model_name,
                    checkpt_dir=checkpt_dir,
                )
        experts.append(expert)
    return experts


def load_datasets(seed, data_dir, chains, batch_size):
    train_dataset = ChainedMNIST(seed=seed, data_dir=data_dir, chains=chains, train=True, max_samples=50_000)
    test_dataset = ChainedMNIST(seed=seed, data_dir=data_dir, chains=chains, train=False, max_samples=50_000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


def plot_hitmap(values, title, x_label, y_label, x_ticks, y_ticks, name):
    # Calculate the figure size based on the dimensions of "values"
    fig_width = len(x_ticks) / 2
    fig_height = len(y_ticks) / 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(y_ticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_ticks)):
        for j in range(len(x_ticks)):
            ax.text(j, i, values[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    if name is not None:
        plt.savefig(f"./results/hitmap_{name}.pdf")
    plt.show()
    return plt, fig
