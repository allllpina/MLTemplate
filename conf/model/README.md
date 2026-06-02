# Model and Task Configurations

This directory holds the YAML files that define the neural network architectures and task-specific hyperparameters.

## Purpose
Configurations here specify the mathematical architecture (via the `_target_` key pointing to `src/models/architectures/`) and the hyperparameters required for initializing it (e.g., input channels, number of classes, learning rate). 

## Reference Example
- `convkan_example.yaml`: Shows how to configure the target architecture and its specific parameters like dropout rate and input dimensions.