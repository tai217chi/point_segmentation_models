from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import FloatTensor

### 入力に処理をかける Transformation Network (T-Net) ###
class TNet3d(nn.Module) :

    def __init__(self, channel) -> None:

        super(TNet3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: FloatTensor) -> FloatTensor:

        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = Variable(
            torch.from_numpy(
                np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
        ).view(1, 9).repeat(batch_size, 1)

        if x.is_cuda:
            _device = x.device
            identity_matrix.to(_device)

        x = x + identity_matrix
        x = x.view(-1, 3, 3) # 回転行列の作成。ただし論文中でも述べられているように回転行列担っている保証はない

        return x

## 中間層におけるTransform Network. 入力の次元数は入力層のTNetと異なり高次元になる##
class TNetkd(nn.Module):

    def __init__(self, num_input_channel: int = 64) -> None:

        super(TNetkd, self).__init__()

        self.conv1 = nn.Conv1d(num_input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_input_channel*num_input_channel) # 高次元空間の回転行列。回転行列担っている保証はない。

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = num_input_channel

    def forward(self, x: FloatTensor) -> FloatTensor:

        batch_size = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity_matrix = Variable(
            torch.from_numpy(
                np.eye(self.k).flatten().astype(np.float32)
            )
        ).view(1, self.k*self.k).repeat(batch_size, 1)

        if x.is_cuda:
            _device = x.device
            identity_matrix.to(_device)

        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)

        return x

class PointNetEncoder(nn.Module):

    def __init__(
            self, 
            global_feat: bool=True, 
            feature_transform: bool=False, 
            input_channel=3
    )-> None:
        
        super(PointNetEncoder, self).__init__()

        self.tnet = TNet3d(input_channel)
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self._global_feat = global_feat
        self._feature_transform = feature_transform

        if self._feature_transform:
            self._feature_tnet = TNetkd(k=64)

    def forward(self, x:FloatTensor):
        
        B, D, N = x.size() # B: batch size, D: 点群の次元数, N: 点群数
        transform_mat = self.tnet(x)
        x = x.transpose(2, 1)
        