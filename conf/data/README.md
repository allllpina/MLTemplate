# Data Configurations

This directory contains YAML files that define the parameters for data loading and preprocessing.

## Purpose
Each file here should correspond to a specific dataset or data pipeline (e.g., batch size, dataset paths, number of workers, pin memory settings). It must include the `_target_` key pointing to the corresponding DataModule in `src/data/`.

## Reference Example
- `cifar10_example.yaml`: Demonstrates how to configure the dataset parameters and target the `CIFAR10DataModule`.