import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) ## OUTPUT = 6 * 256 * 256
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2) ## 16 * 256 * 256
            self.conv3 = nn.Conv2d(16, 6, kernel_size=5, padding=2) ## 16 * 256 * 256
            self.conv4 = nn.Conv2d(6, 2, kernel_size=5, padding=2) ## 16 * 256 * 256

            self.fc1 = nn.Linear(16* 256 * 256, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)
            self.softmax = nn.Softmax(dim=1)

            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):
            # x = self.pool(F.relu(self.conv1(x)))
            # x = self.pool(F.relu(self.conv2(x)))
            # x = torch.flatten(x, 1) # flatten all dimensions except batch
            # x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            # x = self.fc3(x)

            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = self.softmax(x)

            # x = self.conv1(x)
            # x = self.maxpool(x)
            # x = self.conv2(x)
            # x = self.conv3(x)
            # x = self.maxpool(x)
            # x = self.conv4(x)

            return x