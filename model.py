from torch import nn
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
