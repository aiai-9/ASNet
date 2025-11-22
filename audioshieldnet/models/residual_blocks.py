# audioShieldNet/asnet_6/audioshieldnet/models/residual_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False) \
                     if (downsample or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + residual)

class EncoderLite(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 128, embed_dim: int = 128, gru_layers: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(64,  64, downsample=False),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128, downsample=False),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.body(y)
        y = y.transpose(1, 2)
        # Compact GRU params to avoid the runtime warning & overhead
        self.gru.flatten_parameters()
        y, _ = self.gru(y)
        y = y[:, -1, :]
        return self.head(y)
