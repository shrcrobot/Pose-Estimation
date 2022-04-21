'''
import time
import math
import h5py
import numpy as np
import importlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data
import torch_geometric
'''

from torch import nn
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import XConv, fps, global_mean_pool


class Classifier(nn.Module):

    def __init__(self, NUM_CLASS):
        super(Classifier, self).__init__()
        self.pcnn1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        # in_c , out_c , nei_c , spread, n_rep
        self.pcnn2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)

        self.pcnn3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        # self.pcnn4 = XConv(
        #     192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)
        # self.pcnn5 = XConv(
        #     384, 768, dim=3, kernel_size=16, hidden_channels=512, dilation=2)

        self.lin1 = Lin(192, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, NUM_CLASS)

    def forward(self, pos, batch):
        x = F.relu(self.pcnn1(None, pos, batch))
        # print("pcnn1",x.shape)

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.pcnn2(x, pos, batch))
        # print("pcnn2",x.shape)

        idx = fps(pos, batch, ratio=0.333)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.pcnn3(x, pos, batch))
        # print("pcnn3",x.shape)

        # idx = fps(pos, batch, ratio=0.5)
        # x, pos, batch = x[idx], pos[idx], batch[idx]
        # x = F.relu(self.pcnn4(x, pos, batch))
        # # print("pcnn4",x.shape)

        # idx = fps(pos, batch, ratio=0.5)
        # x, pos, batch = x[idx], pos[idx], batch[idx]
        # x = F.relu(self.pcnn5(x, pos, batch))
        # print("pcnn5",x.shape)

        x = global_mean_pool(x, batch)
        # print("global_mean_pool",x.shape)
        self.feature = x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        # print("x.lin3",x.shape)
        return F.log_softmax(x, dim=-1)
