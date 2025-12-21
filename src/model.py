# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    """
    Minimal CNN for MNIST.
    Input:  (B, 1, 28, 28)
    Output: (B, 10) logits
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (B, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (B, 64, 14, 14) after pool
        self.pool = nn.MaxPool2d(2, 2)                            # halves H,W
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 14, 14)
        x = F.relu(self.conv2(x))             # (B, 64, 14, 14)
        x = x.flatten(1)                      # (B, 64*14*14)
        x = F.relu(self.fc1(x))               # (B, 128)
        return self.fc2(x)                    # (B, 10)

