
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(100, 512, kernel_size=(4,4), stride=(1,1))
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(1,1), padding=(1,1))
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=(2,2), padding=(2,2))
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=(2,2), stride=(2,2), padding=(2,2))

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        # print(x.shape)
        x = self.batchnorm1(x)

        x = F.relu(self.deconv2(x))
        # print(x.shape)
        x = self.batchnorm2(x)

        x = F.relu(self.deconv3(x))
        # print(x.shape)
        x = self.batchnorm3(x)

        x = F.relu(self.deconv4(x))
        # print(x.shape)
        x = self.batchnorm4(x)

        x = torch.tanh(self.deconv5(x))
        # print(x.shape)

        return x

