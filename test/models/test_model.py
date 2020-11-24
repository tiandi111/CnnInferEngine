import torch
import torch.nn as nn
import onnx
import torchvision
import time
import torch.nn.functional as F
import caffe2.python.onnx.backend
from caffe2.python import core, workspace


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), 1, 1, bias=True)
        self.res = nn.Sequential(
            nn.BatchNorm2d(16, momentum=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), 1, 0, bias=True),
            nn.BatchNorm2d(16, momentum=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), 1, 1, bias=True),
            nn.BatchNorm2d(16, momentum=0),
            nn.ReLU(),
            nn.Conv2d(16, 64, (1, 1), 1, 0, bias=True)
        )
        self.conv2 = nn.Conv2d(16, 64, (1, 1), 1, 0, bias=True)
        self.fc = nn.Linear(64, 10, bias=True)
        # self.fc.weight.data.fill_(0)
        # self.fc.bias.data.fill_(0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.res(y) + self.conv2(y)
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = y.view(1, 64)
        y = self.fc(y)
        return y

if __name__ == '__main__':
    model = TestModel()
    dummy_input = torch.ones(1, 3, 32, 32)
    model.eval()
    print(model.forward(dummy_input))
    torch.onnx.export(model,
                      dummy_input,
                      "/Users/tiandi03/road-to-dl/d2l/server/test/models/{:d}.onnx".format(int(time.time())),
                      verbose=True)
