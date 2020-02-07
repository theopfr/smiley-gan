
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # model structur:
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4,4),  stride=(2,2))
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4),  stride=(1,1))
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4,4),  stride=(1,1))
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4,4),  stride=(1,1))
        self.batchnorm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4,4),  stride=(1,1))

    def _gaussian_noise_layer(self, x, gaussian_noise_rate=0.0):
        x += gaussian_noise_rate * torch.randn(1, 28, 28).cuda()
        return x

    def forward(self, x, gaussian_noise_rate=0.0):
        # add gaussian noise
        x = self._gaussian_noise_layer(x, gaussian_noise_rate=gaussian_noise_rate)

        # convolutional layer
        x = F.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        # print(x.shape)

        x = F.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        # print(x.shape)

        x = F.leaky_relu(self.conv3(x))
        x = self.batchnorm3(x)
        # print(x.shape)

        x = F.leaky_relu(self.conv4(x))
        x = self.batchnorm4(x)
        # print(x.shape)

        x = torch.sigmoid(self.conv5(x))
        # print(x.shape)

        # flatten to dense output:
        x = x.view(-1, 1*1*1)
        return x

