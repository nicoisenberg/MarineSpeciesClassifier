import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_layer = nn.Linear(4, 3)


    def forward(self, x):
        result = self.test_layer(x)

        return result

class MarineClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self. classifier = nn.Linear(50176, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x= self.classifier(x)

        return x