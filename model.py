import itertools
from dataclasses import dataclass
from typing import List, Any, Optional, Set, Callable, Iterator

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh, BatchNorm1d, LeakyReLU, Parameter
from torch.optim import Optimizer


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


def single_neuron_expert(input_size: int) -> Module:
    return Linear(input_size, input_size)


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
    name: str
    experts: List[Expert]
    indices: List[int]

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

    return [create_composition_with_indices(all_experts, indices) for indices in composition_indices]


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
