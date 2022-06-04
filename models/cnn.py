import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs):
        return self.conv(obs)


class UnitCNN(nn.Module):

    def __init__(self):
        super(UnitCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

    def forward(self, obs):
        features = self.conv(obs)
        features = features / torch.sqrt(torch.square(features).sum(dim=1, keepdim=True))
        return features


# class DataEffCNN(nn.Module):
#
#     def __init__(self):
#         super(DataEffCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(4, 32, 5, stride=5, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 5, stride=5, padding=0),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#     def forward(self, obs):
#         return self.conv(obs)