import os

import lightning as L
import torch
import torchvision

# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10, MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class CIFARDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = torch.utils.data.random_split(cifar_full, [45000, 5000])
        if stage == "test" or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_test, batch_size=BATCH_SIZE)