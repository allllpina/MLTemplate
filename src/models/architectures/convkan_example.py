import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleKANLayer(nn.Module): 
    # To be completed
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
    def forward(self, x):
        return F.silu(self.linear(x))
    
class HybridConvKAN(nn.Module):
    def __init__(self, input_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        self._to_linear = 32 * 8 * 8 
        
        self.kan_classifier = nn.Sequential(
            SimpleKANLayer(self._to_linear, 64),
            SimpleKANLayer(64, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.flatten(x)
        logits = self.kan_classifier(x)
        return logits