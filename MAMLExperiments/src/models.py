import torch
import torch.nn as nn
import torch.nn.functional as F

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
