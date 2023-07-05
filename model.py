import itertools
from dataclasses import dataclass
from typing import List, Any, Optional, Set, Callable, Iterator

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh, BatchNorm1d, LeakyReLU, Parameter, Sigmoid
from torch.optim import Optimizer
from torch import nn

def block_expert(input_size: int) -> Module:
    def block(in_feat: int, out_feat: int, normalize=True) -> List[Module]:
        layers: List[Module] = [Linear(in_feat, out_feat)]
        if normalize:
            layers.append(BatchNorm1d(out_feat, 0.8))
        layers.append(LeakyReLU(0.2, inplace=True))
        return layers

    return Sequential(
        *block(input_size, 128, normalize=False),
        *block(128, 256),
        *block(256, 512),
        *block(512, 1024),
        Linear(1024, int(input_size)),
        Tanh(),
    )

class MnistDiscriminator(nn.Module):
    def __init__(self):
        super(MnistDiscriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ELU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.optimizer = None
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class MnistExpert(nn.Module):
    def __init__(self):
        super(MnistExpert, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        
        self.fc_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.optimizer = None
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layer(x)
        return x


def single_neuron_expert(input_size: int) -> Module:
    return Linear(input_size, input_size)


def leaky_discriminator(input_size: int) -> Module:
    return Sequential(
        Linear(input_size, 512),
        LeakyReLU(0.2, inplace=True),
        Linear(512, 256),
        LeakyReLU(0.2, inplace=True),
        Linear(256, 1),
        Sigmoid(),
    )


class Discriminator(Module):
    def __init__(self, nn: Optional[Module], optim: Optional[Optimizer]) -> None:
        super(Discriminator, self).__init__()
        if nn is not None:
            assert isinstance(nn, Module), "nn must be a torch.nn.Module"
        if optim is not None:
            assert isinstance(optim, Optimizer), "optim must be a torch.optim.Optimizer"
        self.model = nn
        self.optimizer = optim

    def forward(self, inp: Tensor) -> Tensor:
        return self.model(inp.reshape(inp.shape[0], -1))


class Expert(Module):
    def __init__(self, nn: Optional[Module], optim: Optional[Optimizer]) -> None:
        super(Expert, self).__init__()
        if nn is not None:
            assert isinstance(nn, Module), "nn must be a torch.nn.Module"
        if optim is not None:
            assert isinstance(optim, Optimizer), "optim must be a torch.optim.Optimizer"
        self.model = nn
        self.optimizer = optim

    def forward(self, inp: Tensor) -> Tensor:
        return self.model(inp.reshape(inp.shape[0], -1))


@dataclass
class Composition:

    def __init__(self, name, experts, indices):
        self.name = name
        self.experts = experts
        self.indices = indices
        self.father = None
        self.last_X = None
        self.last_eval = None


    def forward(self, x: Tensor) -> Tensor:
        for expert in self.experts:
            x = expert(x)
        return x

    def reverse(self, x: Tensor) -> Tensor:
        for expert in reversed(self.experts):
            x = expert(x)
        return x

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Composition):
            return False
        return self.name == other.name

    def __len__(self) -> int:
        return len(self.experts)

    def __call__(self, x):
        if (self.father != None) and (self.father.last_X != None) and (self.father.last_X.data_ptr() == x.data_ptr()):
            res = self.experts[-1](self.father.last_eval)
        else:
            res = self.forward(x)
        self.last_X = x
        self.last_eval = res
        return res

    def set_father(self, compositions):
        for comp in compositions:
            if len(self.indices) == len(comp.indices) + 1:
                father = True
                for i, ind in enumerate(comp.indices):
                    if ind != self.indices[i]:
                      father = False
                      break
                if father:
                    self.father = comp
                    break
    


def create_composition_with_indices(all_experts: List[Expert], indices: List[int]) -> Composition:
    name = "".join([f"e{i + 1}(" for i in reversed(indices)] + ["x"] + [")"] * len(indices))
    experts = [all_experts[i] for i in indices]
    return Composition(name, experts, indices)


def create_all_compositions(
    all_experts: List[Expert],
    of_sizes: Set[int],
    avoid_repetition: bool = False,
) -> List[Composition]:
    composition_indices: List[List[int]] = []
    indices = range(len(all_experts))
    if avoid_repetition:
        composition_indices = [
            list(permutation)
            for size in of_sizes
            for combo in itertools.combinations(indices, size)
            for permutation in itertools.permutations(combo)
        ]
    else:
        composition_indices = [
            list(product) for size in of_sizes for product in itertools.product(indices, repeat=size)
        ]
    compositions = [create_composition_with_indices(all_experts, indices) for indices in composition_indices]
    for comp in compositions:
        comp.set_father(compositions)
    return compositions


def create_expert(
    input_size: int,
    lr: float,
    weight_decay: float,
    model_constructor: Callable[[int], Module],
    optim_constructor: Callable[[Iterator[Parameter], float, float], Optimizer],
) -> Expert:
    model = model_constructor(input_size)
    optim = optim_constructor(model.parameters(), lr, weight_decay)
    expert = Expert(model, optim)
    return expert
