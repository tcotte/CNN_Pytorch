import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU()
        )
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.c1(x)
        out = self.c3(out)
        out = self.c5(out)
        out = self.c6(out)
        out = self.c7(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))  # 256*6*6 -> 4096
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)

        return out
