import argparse

import os
import sys
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import provider
import random
'''
import time
import math
import h5py
import numpy as np
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
from center_est_pt import CenterEstimateNN



from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


random.seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 250]')
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(GPU_INDEX)
else:
    dtype = torch.FloatTensor
# device = torch.device('cpu')
# dtype = torch.FloatTensor
print("------Building model-------")
# model = Classifier(NUM_CLASS).cuda()
cemodel = CenterEstimateNN(NUM_CLASS).to(device)
# if torch.cuda.device_count() > 1:
#     print("Using "+ str(torch.cuda.device_count() )+" GPUS" )
#     model = nn.DataParallel(model, device_ids=[0])
print("------Successfully Built model-------")

model_save_dir = os.path.join(BASE_DIR, "models", "center_pt"+str(NUM_POINT)+ "MSE_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
log_save_dir = os.path.join(BASE_DIR, "logs", "center_pt", "MSE", str(datetime.datetime.timestamp(datetime.datetime.now())))
os.makedirs(model_save_dir, exist_ok = True)
os.makedirs(log_save_dir, exist_ok = True)

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/test_files.txt'))


# pre_transform, transform = T.NormalizeScale(), T.SamplePoints(NUM_POINT)
# train_dataset= provider.PCDDataset(BASE_DIR, "train", None, pre_transform= True)
train_dataset= provider.PCDDataset(BASE_DIR, "train", None, pre_transform= False)
# print(train_dataset.ang_m, train_dataset.ang_range , train_dataset.ctr_m , train_dataset.ctr_range)
test_dataset= provider.PCDDataset(BASE_DIR, "test", None, pre_transform= False)
# test_dataset= provider.PCDDataset(BASE_DIR, "test", None, pre_transform= False,
#                                   data_params = {'ang_m': train_dataset.ang_m,
#                                                  'ang_range': train_dataset.ang_range,
#                                                  'ctr_m': train_dataset.ctr_m,
#                                                  'ctr_std': train_dataset.ctr_std})
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(cemodel.parameters(), lr = 0.001)

print(train_dataset)
print(test_dataset)

crit = torch.nn.SmoothL1Loss()

train_losses = []

def train(epoch):
    cemodel.train()
    global_step = 1
    train_loss = 0
    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = cemodel(data.pos, data.batch).to(device)
        center = data.center.reshape(data.y.size()[0], -1)
        # loss = crit(out, center)
        loss = F.mse_loss(out, center)
        loss.backward()
        optimizer.step()
        print("epoch: "+str(epoch) + "   loss: "+str(loss.data))
        global_step += 1
        train_loss += loss.data.item() * data.y.size()[0]
    train_losses.append('{:.10f}'.format(train_loss/(len(train_dataset))))

eval_losses = []


def evaluate():
    cemodel.eval()
    with torch.no_grad():
        eval_loss = 0
        for data in test_loader:
            data.to(device)
            out = cemodel(data.pos, data.batch)
            center = data.center.reshape(data.y.size()[0],-1)
            # loss = crit(out, center)
            loss = F.mse_loss(out, center)
            print("out:", out[0])
            print("center", data.center.reshape(data.y.size()[0],-1)[0])
            # print(loss.data)
            eval_loss += loss.data.item() * data.center.size(0)
            # print(eval_loss)
        eval_losses.append('{:.10f}'.format(eval_loss / (len(test_dataset))))
        print('Test Loss: {:.10f}'.format(eval_loss / (len(test_dataset))))
    return eval_loss/(len(test_dataset))


def writeToFileOne(epoch):
    fl = open(os.path.join(log_save_dir, "loss_acc_"+str(NUM_POINT)+".txt"),'a+')
    if epoch == 0:
        fl.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n")
        fl.write("train_loss\t eval_loss\n")
        return
    # fl.write("epoch: " + str(epoch)+"\n")
    for i in range(len(train_losses)):
        fl.write(train_losses[i]+ "\t" + eval_losses[i] + "\n")
    train_losses.clear()
    eval_losses.clear()
    fl.close()

if __name__ == '__main__':
    # evaluate()
    writeToFileOne(0)
    for epoch in range(1, NUM_EPOCHS+1):
        if epoch % 1 == 0:
            eval_loss=evaluate()
        train(epoch)
        if epoch % EPOCHS_TO_WRITE == 0:
            writeToFileOne(epoch)
            torch.save(cemodel.state_dict(), os.path.join(model_save_dir, str(epoch)+"_"+str(eval_loss)+"_"+str(NUM_POINT)))
