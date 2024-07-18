import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y

class SeparableConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv1d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv1d_1x1 = nn.Conv1d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv1d_1x1(y)
        return y

import torch.optim as optim

class EEGNet(nn.Module):
    def __init__(self, nb_classes: int, Chans: int = 64, Samples: int = 128,
                 dropoutRate: float = 0.5, kernLength: int = 63,
                 F1:int = 8, D:int = 2):
        super().__init__()

        F2 = F1 * D

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv1d(Chans, F1, kernLength, padding=(kernLength // 2))
        self.bn1 = nn.BatchNorm1d(F1) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = nn.Conv1d(F1, F2, Chans, groups=F1)
        self.bn2 = nn.BatchNorm1d(F2) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool1d(4)
        self.dropout = nn.Dropout(dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.conv3 = SeparableConv1d(F2, F2, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(F2)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool1d(1)
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        # self.fc = nn.Linear(F2 * ((Samples - Chans + 1) // 32), nb_classes)
        self.fc = nn.Linear(32, nb_classes)

    def forward(self, x: torch.Tensor):
        # Block 1
        y1 = self.conv1(x)
        # print("conv1: ", y1.shape)
        y1 = self.bn1(y1)
        # print("bn1: ", y1.shape)
        y1 = self.conv2(y1)
        # print("conv2", y1.shape)
        y1 = F.relu(self.bn2(y1))
        # print("bn2", y1.shape)
        y1 = self.avg_pool(y1)
        # print("avg_pool", y1.shape)
        y1 = self.dropout(y1)
        # print("dropout", y1.shape)

        # Block 2
        y2 = self.conv3(y1)
        # print("conv3", y2.shape)
        y2 = F.relu(self.bn3(y2))
        # print("bn3", y2.shape)
        y2 = self.avg_pool2(y2)
        # print("avg_pool2", y2.shape)
        y2 = self.dropout(y2)
        # print("dropout", y2.shape)
        y2 = torch.flatten(y2, 1)
        # print("flatten", y2.shape)
        y2 = self.fc(y2)
        #print("fc", y2.shape)

        return y2
