import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600


class simpleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            NUM_INPUT_CHANNELS, NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv2 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv3 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.conv4 = nn.Conv2d(NUM_HIDDEN_CHANNELS,
                               NUM_HIDDEN_CHANNELS, KERNEL_SIZE)
        self.linear = nn.Linear(NUM_HIDDEN_CHANNELS, num_classes)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv2.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv3.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = self.conv4.forward(x)
        x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)

        x = torch.mean(x, dim=[2, 3])

        x = self.linear.forward(x)

        return x

    def loss(self, images, labels):
        return F.cross_entropy(self.forward(images), labels)


class ResBlock(nn.Module):
    def __init__(self, dims=16, kernel_size=3):
        super(ResBlock, self).__init__()

        padding = int(kernel_size//2)*2
        self.model = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=kernel_size,
                      padding=padding, stride=1),
            nn.Conv2d(dims, dims, kernel_size=kernel_size,
                      padding=padding, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(ResNet, self).__init__()
        sequence = [
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=16,
                          kernel_size=5, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(in_channels=16, out_channels=32,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
            )
        ]
        for i in range(4):
            sequence.append(ResBlock(32))
            sequence.append(nn.ReLU())
            sequence.append(nn.BatchNorm2d(32))
        sequence.append(nn.MaxPool2d(5, 2))

        self.conv_sequence = nn.Sequential(
            *sequence
        )

        self.fc_seq = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        x = self.conv_sequence(x)
        x = x.mean(dim=[2, 3])
        # x = x.reshape(bs, -1)
        return self.fc_seq(x)

    def loss(self, imgs, labels):
        return F.cross_entropy(self.forward(imgs), labels)


if __name__ == "__main__":
    model = ResNet(3, 1)
    x = torch.randn(5, 3, 16, 16)
    y = model(x)
    print(y.shape)
    # print(y.shape)
