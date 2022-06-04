import math
import torch
import torch.nn as nn
from torch.nn import functional as F
# from models.cnn import CNN, DataEffCNN
from models.components import *
from models.ensemble import Ensemble


class Q_Network(nn.Module):

    """ Nature 2015 implementation """

    def __init__(self, action_size):
        super(Q_Network, self).__init__()
        self.conv = CNN()
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_size)
        )

    def forward(self, obs):
        embed = self.conv(obs)
        return self.fc(embed)


class UnitQ_Network(nn.Module):

    def __init__(self, action_size):
        super(UnitQ_Network, self).__init__()
        self.conv = UnitCNN()
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=action_size)
        )

    def forward(self, obs):
        embed = self.conv(obs)
        return self.fc(embed)


class Dueling_Q_Network(nn.Module):
    def __init__(self, action_size):
        super(Dueling_Q_Network, self).__init__()
        self.conv = CNN()
        self.a = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_size)
        )
        self.v = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_size)
        )

    def forward(self, obs):
        embed = self.conv(obs)
        v = self.v(embed)
        a = self.a(embed)
        return v + a - a.mean(dim=1, keepdim=True)


class Noisy_Q_Network(nn.Module):

    """ use model.train() to enable noisy network """

    def __init__(self, action_size):
        super(Noisy_Q_Network, self).__init__()
        self.conv = CNN()
        self.noisylayers = [
            NoisyLinear(in_features=64 * 7 * 7, out_features=512, std_init=0.1),
            NoisyLinear(in_features=512, out_features=action_size, std_init=0.1)]
        self.fc = nn.Sequential(
            self.noisylayers[0],
            nn.ReLU(),
            self.noisylayers[1]
        )

    def reset_noise(self):
        self.train()
        for noisylayer in self.noisylayers:
            noisylayer.reset_noise()

    def forward(self, obs):
        embed = self.conv(obs)
        return self.fc(embed)


class Dropout_Q_Network(nn.Module):

    def __init__(self, action_size, dropout_prob=0.2):
        super(Dropout_Q_Network, self).__init__()
        self.conv = CNN()
        hidden_size = int((1 + dropout_prob) * 512)
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=hidden_size),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_size)
        )
    
    def reset_noise(self):
        self.train()

    def forward(self, obs):
        embed = self.conv(obs)
        return self.fc(embed)


class Distributional_Q_Network(nn.Module):

    def __init__(self, action_size, atoms):
        super(Distributional_Q_Network, self).__init__()
        self.action_size = action_size
        self.natoms = atoms
        self.cnn = CNN()
        self.v = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, atoms)
        )
        self.a = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, atoms * action_size)
        )

    def forward(self, obs, log=False):
        """
        B: batch size
        d: action size
        M: n atoms

        obs (B, 4, 84, 84) return (B, d, M)
        """
        embed = self.cnn(obs)
        v = self.v(embed).unsqueeze(-2)  # (B, 1)
        a = self.a(embed)  # (B, d * M)
        a = a.view(*a.shape[:-1], self.action_size, self.natoms)  # (B, d, M)
        logits = v + a - a.mean(-2, keepdim=True)

        if log:  # Use log softmax for numerical stability
            res = F.log_softmax(logits, dim=-1)
        else:
            res = F.softmax(logits, dim=-1)
        return res


class Distributional_Noisy_Q_Network(nn.Module):

    def __init__(self, action_size, atoms):
        super(Distributional_Noisy_Q_Network, self).__init__()
        self.action_size = action_size
        self.natoms = atoms
        self.cnn = CNN()
        self.noisylayers = [
            NoisyLinear(64 * 7 * 7, 256, std_init=0.1),
            NoisyLinear(256, atoms, std_init=0.1),
            NoisyLinear(64 * 7 * 7, 256, std_init=0.1),
            NoisyLinear(256, atoms * action_size, std_init=0.1)

        ]
        self.v = nn.Sequential(
            self.noisylayers[0],
            nn.ReLU(),
            self.noisylayers[1]
        )
        self.a = nn.Sequential(
            self.noisylayers[2],
            nn.ReLU(),
            self.noisylayers[3]
        )

    def forward(self, obs, log=False):
        """
        B: batch size
        d: action size
        M: n atoms

        obs (B, 4, 84, 84) return (B, d, M)
        """
        embed = self.cnn(obs)
        v = self.v(embed).unsqueeze(-2)  # (B, 1)
        a = self.a(embed)  # (B, d * M)
        a = a.view(*a.shape[:-1], self.action_size, self.natoms)  # (B, d, M)
        logits = v + a - a.mean(-2, keepdim=True)

        if log:  # Use log softmax for numerical stability
            res = F.log_softmax(logits, dim=-1)
        else:
            res = F.softmax(logits, dim=-1)
        return res

    def reset_noise(self):
        for noisylayer in self.noisylayers:
            noisylayer.reset_noise()



class RainbowMeanArch(nn.Module):

    def __init__(self, ensemble_size, action_size, atoms):
        super(RainbowMeanArch, self).__init__()
        self.model = Ensemble(Distributional_Noisy_Q_Network, ensemble_size=ensemble_size,
                 action_size=action_size, atoms=atoms)

    def forward(self, obs, log=False):
        return self.model(obs, log).mean(dim=1)

    def reset_noise(self):
        return self.model.reset_noise()