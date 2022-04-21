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
from rotation_cls import RotationEstimateNN
from center_est_pt_rot import CenterEstimateNN
from classifier import Classifier


# import torchvision
# import torchvision.transforms as TV

from torch_geometric.data import DataLoader
# import torch_geometric.transforms as T


random.seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
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
EPOCHS_TO_WRITE = 1
       
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(GPU_INDEX)
    print("GPU:"+str(GPU_INDEX))
else:
    dtype = torch.FloatTensor
# device = torch.device('cpu')
# dtype = torch.FloatTensor
print("------Building model and loading-------")

rotmodel = RotationEstimateNN(NUM_CLASS).to(device)
rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'rotation_est_value_ctr_8cls1024', '389_0.9404869031906128_1024'), map_location='cpu'))
cls_model = Classifier(NUM_CLASS).to(device)
cls_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'classifier_1024', '117_0.98_1024'), map_location='cpu'))


print("------Successfully Built model-------")

log_save_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_save_dir, exist_ok = True)

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/test_files.txt'))


# pre_transform, transform = T.NormalizeScale(), T.SamplePoints(NUM_POINT)
train_dataset= provider.PCDDataset(BASE_DIR, "train", None, pre_transform= False)
# print(train_dataset.ang_m, train_dataset.ang_range , train_dataset.ctr_m , train_dataset.ctr_range)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(rotmodel.parameters(), lr = 0.001)



criterion = torch.nn.SmoothL1Loss()


err = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=float)

cls_cnt = np.array([0,0,0,0,0,0,0,0])
def evaluate():
    rotmodel.eval()
    cls_model.eval()
    with torch.no_grad():
        eval_loss = 0
        for data in train_loader:
            data.to(device)
            angle =data.angle.view((data.num_graphs,3))/180*math.pi
            rotout = rotmodel(data.pos,data.center, data.batch).to(device)
            cls_out = cls_model(data.pos, data.batch)
            pred_cls = cls_out.max(1)[1].reshape(-1,1)
            err[pred_cls]=err[pred_cls]+torch.abs(rotout-angle).numpy()
            cls_cnt[pred_cls]= cls_cnt[pred_cls] +1
            reshaped_cls = np.reshape(cls_cnt,(-1,8))
            repeated_cls = np.repeat(reshaped_cls,3, axis=0).transpose()
            print(err)
            print(cls_cnt)
            print(np.divide(err,repeated_cls))
            print(pred_cls.view(data.num_graphs,1))

            print("err:", torch.abs(rotout-angle))

    return


if __name__ == '__main__':
    for epoch in range(1, NUM_EPOCHS+1):
        evaluate()
