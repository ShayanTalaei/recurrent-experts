from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, List, Set, Any, Optional

import torch.cuda
from torch import Tensor, rot90, roll
from torchvision.transforms.functional import hflip


class Transformer(ABC):
    def get_name(self) -> str:
        ...

    def get_abbreviation(self) -> str:
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...

    def reverse(self, x: Tensor) -> Tensor:
        ...

    def __hash__(self) -> int:
        return self.get_name().__hash__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Transformer):
            return False
        return self.get_name() == other.get_name()


@dataclass(slots=True, eq=False)
class LambdaTransformer(Transformer):
    name: str
    abbreviation: str = field(init=False)
    forward_func: Callable[[Tensor], Tensor]
    reverse_func: Callable[[Tensor], Tensor]

    def __post_init__(self) -> None:
        self.abbreviation = "".join([part[0] for part in self.name.split("_")])

    def get_name(self) -> str:
        return self.name

    def get_abbreviation(self) -> str:
        return self.abbreviation

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_func(x)

    def reverse(self, x: Tensor) -> Tensor:
        return self.reverse_func(x)

    def __repr__(self) -> str:
        return f"LambdaTransformer({self.name})"

    def __str__(self) -> str:
        return f"LambdaTransformer({self.abbreviation})"


@dataclass(slots=True, eq=False)
class Involution(Transformer):
    func: Callable[[Tensor], Tensor]
    name: str
    abbreviation: str = field(init=False)

    def __post_init__(self) -> None:
        self.abbreviation = "".join([part[0] for part in self.name.split("_")])

    def get_name(self) -> str:
        return self.name

    def get_abbreviation(self) -> str:
        return self.abbreviation

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)

    def reverse(self, x: Tensor) -> Tensor:
        return self.func(x)

    def __repr__(self) -> str:
        return f"Involution({self.name})"

    def __str__(self) -> str:
        return f"Involution({self.abbreviation})"


horizontal_flip = Involution(func=hflip, name="horizontal_flip")
rotate_left = LambdaTransformer(forward_func=rot90, reverse_func=lambda x: rot90(x, 1, [1, 0]), name="rotate_left")
horizontal_shift = LambdaTransformer(
    forward_func=lambda x: roll(x, x.size(1) // 2, 1),
    reverse_func=lambda x: roll(x, (x.size(1) + 1) // 2, 1),
    name="horizontal_shift",
)
vertical_shift = LambdaTransformer(
    forward_func=lambda x: roll(x, x.size(0) // 2, 0),
    reverse_func=lambda x: roll(x, (x.size(0) + 1) // 2, 0),
    name="vertical_shift",
)
# all_transformers = [horizontal_flip, rotate_left, horizontal_shift, vertical_shift]

lambda_up    = lambda x: roll(x, -4, 1)
lambda_left  = lambda x: roll(x, -4, 2)
lambda_right = lambda x: roll(x, 4, 2)
lambda_down  = lambda x: roll(x, 4, 1)


shift_up = LambdaTransformer(
  forward_func=lambda_up,
  reverse_func= lambda_down,
  name="shift_up"
)
shift_left = LambdaTransformer(
  forward_func=lambda_left,
  reverse_func= lambda_right,
  name="shift_left"
)
shift_right = LambdaTransformer(
  forward_func=lambda_right,
  reverse_func= lambda_left,
  name="shift_right"
)
shift_down = LambdaTransformer(
  forward_func=lambda_down,
  reverse_func= lambda_up,
  name="shift_down"
)
shift_up_left = LambdaTransformer(
  forward_func=lambda x:lambda_up(lambda_left(x)),
  reverse_func= lambda x:lambda_down(lambda_right(x)),
  name="shift_up_left"
)
shift_up_right = LambdaTransformer(
  forward_func=lambda x:lambda_up(lambda_right(x)),
  reverse_func= lambda x:lambda_down(lambda_left(x)),
  name="shift_up_right"
)
shift_down_left = LambdaTransformer(
  forward_func=lambda x:lambda_down(lambda_left(x)),
  reverse_func= lambda x:lambda_up(lambda_right(x)),
  name="shift_down_left"
)
shift_down_right = LambdaTransformer(
  forward_func=lambda x:lambda_down(lambda_right(x)),
  reverse_func= lambda x:lambda_up(lambda_left(x)),
  name="shift_down_right"
)

# all_transformers = [shift_up, shift_down, shift_right, shift_left,
#                     shift_up_left, shift_up_right, 
#                     shift_down_left, shift_down_right]
all_transformers = [shift_up, shift_right, shift_up_right]


class Chain(Transformer):
    """
    This class represents a pipeline of transformations that are applied in order.
    """

    __slots__ = "transformers", "name", "abbreviation"

    def __init__(self, transformers: List[Transformer]) -> None:
        self.transformers = transformers
        self.name = "".join([f"{t.get_name()}(" for t in reversed(transformers)] + ["x"] + [")"] * len(transformers))
        self.abbreviation = "".join(
            [f"{t.get_abbreviation()}(" for t in reversed(transformers)] + ["x"] + [")"] * len(transformers)
        )

    def get_name(self) -> str:
        return self.name

    def get_abbreviation(self) -> str:
        return self.abbreviation

    def forward(self, x: Tensor) -> Tensor:
        for transformer in self.transformers:
            x = transformer.forward(x)
        return x

    def reverse(self, x: Tensor) -> Tensor:
        for transformer in reversed(self.transformers):
            x = transformer.reverse(x)
        return x

    def __repr__(self) -> str:
        return f"Chain({self.abbreviation})"

    def __str__(self) -> str:
        return f"Chain({self.abbreviation})"

    def __len__(self) -> int:
        return len(self.transformers)

    def __getitem__(self, index: int) -> Transformer:
        return self.transformers[index]

    def __iter__(self):
        return iter(self.transformers)

    def __reversed__(self):
        return reversed(self.transformers)

    def __contains__(self, transformer: Transformer) -> bool:
        return transformer in self.transformers


class IdentityDetector:
    """
    We use this class to identify identity transformations in a chain.
    """

    __slots__ = ("random_keys",)

    def __init__(self, seed: int, shape: List[int], samples: int) -> None:
        tensor_size = int(torch.prod(Tensor(shape), 0).item())
        tensor_generator = torch.Generator().manual_seed(seed)
        self.random_keys = [torch.rand(tensor_size, generator=tensor_generator).view(shape) for _ in range(samples)]

    def is_identity(self, t: Transformer | Callable[[Tensor], Tensor]) -> bool:
        def func(x: Tensor) -> Tensor:
            if isinstance(t, Transformer):
                return t.forward(x)
            elif isinstance(t, Callable):
                return t(x)
            raise NotImplementedError("t must be a Transformer or a Callable")

        for key in self.random_keys:
            transformed = func(key)
            if transformed.shape != key.shape or not torch.allclose(transformed, key):
                return False

        return True


def make_chains_of_size(candidates: List[Transformer], size: int, avoid_repetition: bool) -> List[Chain]:
    """
    make all possible chains that have a unique permutation and a certain size.
    """
    if size == 1:
        return [Chain([transformer]) for transformer in candidates]

    chains: List[Chain] = []
    for _ in range(size):
        for candidate_idx, selected_candidate in enumerate(candidates):
            other_candidates = (
                candidates[:candidate_idx] + candidates[candidate_idx + 1 :] if avoid_repetition else candidates
            )
            further_sub_chains = make_chains_of_size(other_candidates, size - 1, avoid_repetition)
            for sub_chain in further_sub_chains:
                chains.append(Chain([selected_candidate] + sub_chain.transformers))

    return chains


def make_chains(
    candidates: List[Transformer],
    of_sizes: Set[int],
    avoid_repetition: bool = True,
    identity_detector: Optional[IdentityDetector] = None,
) -> List[Chain]:
    chains = []
    for size in of_sizes:
        new_chains = make_chains_of_size(candidates, size, avoid_repetition)
        if identity_detector is not None:
            # we should eliminate chains that are identity transformations
            new_chains = [chain for chain in new_chains if not identity_detector.is_identity(chain)]
        chains.extend(new_chains)
    return chains
