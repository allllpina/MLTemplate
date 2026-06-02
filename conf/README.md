# Configuration Root (Hydra)

This directory serves as the control panel for the entire project. It relies on the Inversion of Control pattern using Hydra. There should be no Python code here, only YAML configuration files.

## Purpose
The files here define how different components (data, models, loggers, trainer) are assembled into a single pipeline. Hyperparameters and module targets are defined here and injected into the codebase via `hydra.utils.instantiate`.

## Reference Example
- `<name>.yaml`: The main entry point that aggregates defaults from the `data/` and `model/` subdirectories and configures the PyTorch Lightning Trainer. 
- See `example_config.yaml` for reference.