import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.save_hyperparameters()
        # Стандартна нормалізація для CIFAR
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        # Завантажує дані (викликається лише на 1 GPU)
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(self.hparams.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [45000, 5000], generator=torch.Generator().manual_seed(13)
            )

        if stage == "test" or stage is None:
            self.cifar_test = datasets.CIFAR10(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,pin_memory=self.hparams.pin_memory)