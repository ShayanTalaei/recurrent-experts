from typing import Dict
from typing import Callable, List
import pandas as pd
import argparse
import os
import numpy as np
import torch

from torch.utils.data.dataset import Dataset

import torchvision
from torchvision import transforms

DATA_DIR = "./data"

transformers: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "horizontal_flip": lambda x : transforms.functional.hflip(x),
    "rotate_left": lambda x : torch.rot90(x),
    "horizontal_shift": lambda x : torch.roll(x, x.size(1) // 2, 1),
    "vertical_shift": lambda x : torch.roll(x, x.size(0) // 2, 0),
}

np.random.seed(4363562)


# create a class for MNIST dataset with different transformations
# we need to combine pairs of transformations
# list of initial transformations are: horizontal flip, left rotate, right rotate, right shift, upshift
class MNISTDataset(Dataset):
    def __init__(self,  transformer_names: List[str], train: bool = True):
        for transformer_name in transformer_names:
            assert transformer_name in transformers.keys(), f"{transformer_name} is not a valid transformer"
        
        self.transformer_names = transformer_names
        data = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
        self.X = data.data[:50_000] if train else data.data[50_000:]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].reshape((28, 28)) / 256
        x = (x - 0.1307)/0.3081
        t = self.get_random_transformer()
        return x.type(torch.FloatTensor), t(x).type(torch.FloatTensor)
    
    def get_random_transformer(self) ->  List[Callable[[torch.Tensor], torch.Tensor]]:
        # get two random uniqeue transformer names
        transformer_names = np.random.choice(self.transformer_names, 2, replace=False)
        return lambda x : transformers[transformer_names[1]](transformers[transformer_names[0]](x))