######
# test corner loss function
######
import argparse

import os
import sys
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import numpy as np
import random
import math
'''
import time
import math
import h5py
import importlib
import matplotlib.pyplot as plt
from torch.nn import Linear as Lin
from torch.autograd import Variable
from torch_geometric.data import Dataset, Data
import torch_geometric
from torch_geometric.nn import XConv, fps, global_mean_pool
'''

import torch
from torch import nn
import torch.nn.functional as F

import provider_ctrrot as provider
from rotation_cls_new import RotationEstimateNN
# from center_est_pt_rot import CenterEstimateNN

# import torchvision
# import torchvision.transforms as TV

from torch_geometric.data import DataLoader
# import torch_geometric.transforms as T


random.seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=20000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
EPOCHS_TO_WRITE = 5
       
# MAX_NUM_POINT = 1024

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


LEARNING_RATE_MIN = 0.00001
        
NUM_CLASS = 8
BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch


##################

class cubic_loss(nn.Module):
    def __init__(self):
        super(cubic_loss,self).__init__()
        return

    def paramToCubicVertex(self, tra, rot):
        cos = lambda ang: torch.cos(ang)
        sin = lambda ang: torch.sin(ang)
        tx, ty, tz = tra[:,0], tra[:,1], tra[:,2]   # batch_size *3
        roll, pitch, yaw = rot[:,0], rot[:,1], rot[:,2]
        batch_size = tra.size()[0]
        Rx = torch.zeros((batch_size, 4, 4))
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos(roll)
        Rx[:, 2, 2] = cos(roll)
        Rx[:, 2, 1] = sin(roll)
        Rx[:, 1, 2] = -sin(roll)
        Ry = torch.zeros((batch_size, 4, 4))
        Ry[:, 1, 1] = 1
        Ry[:, 0, 0] = cos(pitch)
        Ry[:, 2, 2] = cos(pitch)
        Ry[:, 2, 0] = -sin(pitch)
        Ry[:, 0, 2] = sin(pitch)
        Rz = torch.zeros((batch_size, 4, 4))
        Rz[:, 2, 2] = 1
        Rz[:, 0, 0] = cos(yaw)
        Rz[:, 1, 1] = cos(yaw)
        Rz[:, 0, 1] = -sin(yaw)
        Rz[:, 1, 0] = sin(yaw)
        yx = torch.matmul(Ry, Rx)
        zyx = torch.matmul(Rz, yx)
        zyx[:, 0, 3] = tx
        zyx[:, 1, 3] = ty
        zyx[:, 2, 3] = tz
        zyx[:, 3, 3] = 1
        cords_np = np.array([[-1,-1,1,1],[-1,1,1,1],[1,1,1,1],
                             [1,-1,1,1],[-1,-1,-1,1],[-1,1,-1,1],
                             [1,1,-1,1],[1,-1,-1,1]], dtype=np.float32)
        cords = torch.transpose(torch.from_numpy(cords_np),0,1)  # 4*8

        # print(zyx.size())  # 2*4*4
        # print(cords.size()) # 4*8
        # print(torch.matmul(zyx, cords).size()) # 2*4*8
        # print(torch.transpose(torch.matmul(zyx, cords),1,2).size())

        cords_tra = torch.transpose(torch.matmul(zyx, cords),1,2)[:,:,0:3]
        # print(cords_tra.size())
        return cords_tra

    def cal_loss(self, output, target):
        cb1 = self.paramToCubicVertex(output[:,0:3], output[:,3:6])
        cb2 = self.paramToCubicVertex(target[:,0:3], target[:,3:6])
        res = cb1 - cb2
        # print(output.size()[0])
        return torch.sum(torch.pow(res, 2))/output.size()[0]

    def forward(self, output, target):
        return self.cal_loss(output, target)

# criterion = cubic_loss().to(device)

criterion_cubic = cubic_loss()
##################




def to_residual(angle):
    return angle%30

def to_cls(angle):
    return (( (angle) // 30)%12).long()


train_losses = []
train_losses_cls = []





def train():
    global_step = 1
    train_loss = 0
    train_loss_cls = 0

    pred_ctr = torch.FloatTensor([0.1,0.1,0.1]).view((1,-1))
    pred_rot = torch.FloatTensor([0,30,60]).view((1,-1)) * math.pi /180
    pred = torch.cat((pred_ctr, pred_rot), 1)


    label_ctr = torch.FloatTensor([0.1,0.1,0.1]).view((1,-1))
    label_rot = torch.FloatTensor([0,30,50]).view((1,-1)) * math.pi /180
    label = torch.cat((label_ctr, label_rot), 1)

    loss = criterion_cubic(pred, label)

    print(loss.data)




if __name__ == '__main__':
    train()

