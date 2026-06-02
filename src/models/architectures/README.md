# Neural Network Architectures

This directory contains the pure mathematics and structural definitions of the neural networks.

## Purpose
All classes here must be standard `torch.nn.Module` implementations. They define the layers (convolutions, splines, linear layers) and the `forward()` pass. These classes must NOT contain any training infrastructure, loss calculations, or logging logic. They solely accept an input tensor and return an output tensor.

## Reference Example
- `convkan_example.py`: Contains the implementation of the CNN encoder and the KAN classifier layers.