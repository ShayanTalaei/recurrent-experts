from typing import Iterator

from torch.nn import Parameter
from torch.optim import Optimizer, Adam, SGD


def new_adam(params: Iterator[Parameter], lr: float, weight_decay: float) -> Optimizer:
    return Adam(params=params, lr=lr, weight_decay=weight_decay)


def new_sgd(params: Iterator[Parameter], lr: float, weight_decay: float) -> Optimizer:
    return SGD(params=params, lr=lr, weight_decay=weight_decay)
