from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST

from transformer import Chain


class ChainedMNIST(Dataset):
    def __init__(
        self,
        seed: int,
        data_dir: str,
        chains: List[Chain],
        train: bool = True,
        max_samples: int = 50_000,
    ) -> None:
        self.chains = chains
        self.train = train

        mnist = MNIST(root=data_dir, train=train, download=True)
        self.X = mnist.data[:max_samples]
        self.padder = nn.ZeroPad2d(2)

        self.chain_index = torch.randint(
            low=0,
            high=len(chains),
            size=(max_samples,),
            generator=torch.Generator().manual_seed(seed),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        canonical: Tensor = (self.X[index] / 255).type(torch.FloatTensor).unsqueeze(0)
        canonical = self.padder(canonical)
        chosen_chain_index = self.chain_index[index].item()
        transformed: Tensor = self.chains[chosen_chain_index].forward(canonical).type(torch.FloatTensor)
        return canonical, transformed, chosen_chain_index
