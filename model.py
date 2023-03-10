import torch
import torch.nn as nn

class Expert(nn.Module):

    def __init__(self, dataset: str, input_size: int) -> None:
        super(Expert, self).__init__()

        # Architecture
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if dataset == 'MNIST':
            # self.model = nn.Sequential(
            #     *block(input_size, 128, normalize=False),
            #     *block(128, 256),
            #     *block(256, 512),
            #     *block(512, 1024),
            #     nn.Linear(1024, int(input_size)),
            #     nn.Tanh()
            # )
            self.model = nn.Linear(input_size, input_size)
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out =  self.model(input.reshape(input.shape[0], -1))
        return out


class Discriminator(nn.Module):

    def __init__(self, input_size: int, dataset: str) -> None:
        super(Discriminator, self).__init__()

        # Architecture
        if dataset == 'MNIST':
            self.model = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
       return self.model(input.reshape(input.shape[0], -1))
