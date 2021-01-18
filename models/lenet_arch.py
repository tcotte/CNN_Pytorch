import torch.nn as nn

# Two CNN + two FC Layers NN
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2)),  # output dimension = (28-2+1)*27*3= 2187
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)  # output dimension = (27-2+1)*26*3 =
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(2, 2)),  # output dimension = (26-2+1)*25*9
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1)  # output dimension = (25-2+1)*24*9 = 5184
        )
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(5184, 1000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # use reshape() to match the input of the FC layer1
        out = out.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        # use F.log_softmax() to normalize the output
        sm = nn.functional.log_softmax(out, _stacklevel=3)
        return sm


# Another model inspired from LeNet
class Model_Type_LeNet(nn.Module):
    def __init__(self):
        super(Model_Type_LeNet, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),  # output dimension = (28-5+1)*24*6= 2187
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((24-2)%2+1)*12*6 =
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # output dimension = (12-5+1)*8*16
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # output dimension = ((8-2)%2+1)*4*16 = 256
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


model_lenet = Model_Type_LeNet()  # model type LeNet is not good for this applciation. Maybe it's because of the dimensions of the input picture or the MaxPool2d instead of the MeanPool2D
