# DataModules

This directory is dedicated exclusively to data handling logic. 

## Purpose
All classes here must inherit from `pytorch_lightning.LightningDataModule`. Their sole responsibility is to handle data downloading, reading from disk, transformations, and returning PyTorch `DataLoader` objects for the train, validation, and test splits. These classes should remain completely agnostic to the neural network architectures.

## Reference Example
- `cifar_datamodule_example.py`: A complete working example of a DataModule handling standard image transformations and data splitting.