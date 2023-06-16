import matplotlib.pyplot as plt

from dataset import ChainedMNIST
from transformer import make_chains, all_transformers

if __name__ == "__main__":
    mnist = ChainedMNIST(
        seed=123,
        data_dir="./data",
        chains=make_chains(all_transformers, {1, 2}, True),
        train=True,
        max_samples=10,
    )
    canonical, transformed, chain = mnist[1]
    print(canonical.min(), canonical.max())
    print(chain)
    plt.imshow(transformed, vmin=-1, vmax=1)
    plt.show()
