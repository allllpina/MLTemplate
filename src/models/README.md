# Training Scenarios (Lightning Tasks)

This directory contains the training logic and orchestration.

## Purpose
Files here contain classes that inherit from `pytorch_lightning.LightningModule`. These "Tasks" do not define the layers of the neural network; instead, they define the *process* of training. This includes:
1. Calculating the Loss function.
2. Configuring optimizers and learning rate schedulers.
3. Calculating and logging metrics.

Name these files based on the objective (e.g., classification, segmentation, anomaly_detection) rather than the specific model architecture.

## Reference Example
- `classification_task_example.py`: Demonstrates a standard multiclass classification pipeline with accuracy and F1-score tracking.