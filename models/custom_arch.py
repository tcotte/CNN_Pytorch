import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model_Max_Pool(nn.Module):
    def __init__(self):
        super(Model_Max_Pool, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),  # output dimension = (28-5+1)*24*6= 2187
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((24-2)%2+1)*12*6 =
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # output dimension = (12-5+1)*8*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((8-2)%2+1)*4*16 = 256
        )
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(256, 1000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.size())
        # use reshape() to match the input of the FC layer1
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # use F.log_softmax() to normalize the output
        sm = nn.functional.log_softmax(out, _stacklevel=3)
        return sm


class Model_Multiple_fc(nn.Module):
    def __init__(self):
        super(Model_Multiple_fc, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),  # output dimension = (28-5+1)*24*6= 2187
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((24-2)%2+1)*12*6 =
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # output dimension = (12-5+1)*8*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((8-2)%2+1)*4*16 = 256
        )
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(256, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc6 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.size())
        # use reshape() to match the input of the FC layer1
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        # use F.log_softmax() to normalize the output
        sm = nn.functional.log_softmax(out, _stacklevel=3)
        return sm


class Model_Big_fc(nn.Module):
    def __init__(self):
        super(Model_Big_fc, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),  # output dimension = (28-5+1)*24*6= 2187
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((24-2)%2+1)*12*6 =
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # output dimension = (12-5+1)*8*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((8-2)%2+1)*4*16 = 256
        )
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(256, 100000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc2 = nn.Linear(100000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print(out.size())
        # use reshape() to match the input of the FC layer1
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # use F.log_softmax() to normalize the output
        sm = nn.functional.log_softmax(out, _stacklevel=3)
        return sm


class Model_Deep_CNN(nn.Module):
    def __init__(self):
        super(Model_Deep_CNN, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3,3)),  # output dimension = (28-5+1)*24*6= 2187
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = 13*13*6
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3)),  # 11*11*12
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3)),  # 9*9*6
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4)),  # output dimension = 6*6*32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # output dimension = 3*3*32
        )
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(288, 1000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.size())
        # use reshape() to match the input of the FC layer1
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # use F.log_softmax() to normalize the output
        sm = nn.functional.log_softmax(out, _stacklevel=3)
        return sm