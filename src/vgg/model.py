import torch
from torch import nn

class TinyVGGSample(nn.Module):
    def __init__(self, input: int, hidden: int, output: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.MaxPool2d(2)   
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.MaxPool2d(2)   
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 8 * 8, output)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
