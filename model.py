from torch import nn
from torch.nn import functional as F
from math import ceil


class VGG(nn.Module):
    def __init__(
        self,
        input_dim: tuple = (1, 64, 44),
        n_output: int = 12,
        stride: int = 1,
        n_channel: int = 16,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        n_input, n_ax1, n_ax2 = input_dim
        kernel_size = 3
        padding = 2
        pooling_size = 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_input,
                out_channels=n_channel,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channel,
                out_channels=2 * n_channel,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * n_channel,
                out_channels=4 * n_channel,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * n_channel,
                out_channels=8 * n_channel,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                device=device,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_size),
        )
        self.flatten = nn.Flatten()
        #  8 = 2 * 4
        #  16 = 2 ** 4
        linear_shape = 8 * n_channel * ceil((n_ax1 + 8) / 16) * ceil((n_ax2 + 8) / 16)
        self.linear = nn.Linear(
            in_features=linear_shape,
            out_features=n_output,
            device=device,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class M5(nn.Module):
    def __init__(self, n_input: int = 1, n_output: int = 35, stride: int = 16, n_channel: int = 32, device="cpu"):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride, device=device)
        self.bn1 = nn.BatchNorm1d(n_channel, device=device)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, device=device)
        self.bn2 = nn.BatchNorm1d(n_channel, device=device)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, device=device)
        self.bn3 = nn.BatchNorm1d(2 * n_channel, device=device)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, device=device)
        self.bn4 = nn.BatchNorm1d(2 * n_channel, device=device)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output, device=device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)
