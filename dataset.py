from typing import List, Tuple

import torch
from torch import Tensor
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

        if train:
            # generate random data
            self.X = torch.randn(size=(max_samples, 28, 28), generator=torch.Generator().manual_seed(seed))
        else:
            mnist = MNIST(root=data_dir, train=train, download=True)
            self.X = mnist.data[:max_samples]

        self.chain_index = torch.randint(
            low=0,
            high=len(chains),
            size=(max_samples,),
            generator=torch.Generator().manual_seed(seed),
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        if not self.train:
            canonical: Tensor = ((self.X[index] / 255 - 0.1307) / 0.3081).type(torch.FloatTensor)
        else:
            canonical: Tensor = self.X[index].type(torch.FloatTensor)
        chosen_chain_index = self.chain_index[index].item()
        transformed: Tensor = self.chains[chosen_chain_index].forward(canonical).type(torch.FloatTensor)
        return canonical, transformed, chosen_chain_index
