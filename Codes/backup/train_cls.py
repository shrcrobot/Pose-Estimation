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
from classifier import Classifier



from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


random.seed(0)

# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
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
else:
    dtype = torch.FloatTensor
# device = torch.device('cpu')
# dtype = torch.FloatTensor
print("------Building model-------")
# model = Classifier(NUM_CLASS).cuda()
model = Classifier(NUM_CLASS).to(device)
# if torch.cuda.device_count() > 1:
#     print("Using "+ str(torch.cuda.device_count() )+" GPUS" )
#     model = nn.DataParallel(model, device_ids=[0])
print("------Successfully Built model-------")



model_save_dir = os.path.join(BASE_DIR, "models", "classifier_"+str(NUM_POINT))
log_save_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(model_save_dir, exist_ok = True)
os.makedirs(log_save_dir, exist_ok = True)

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/test_files.txt'))


# pre_transform, transform = T.NormalizeScale(), T.SamplePoints(NUM_POINT)
train_dataset= provider.PCDDataset(BASE_DIR, "train", None, None)
test_dataset= provider.PCDDataset(BASE_DIR, "test", None, None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

print(train_dataset)
print(test_dataset)

train_losses = []
train_accs = []

def train(epoch):
    model.train()
    global_step = 1
    train_loss = 0
    train_acc =0
    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print("epoch: "+str(epoch) + "   loss: "+str(loss.data))
        global_step += 1
        train_loss += loss.data.item() * data.y.size(0)
        train_acc += out.max(1)[1].eq(data.y).sum().item()
    train_losses.append('{:.6f}'.format(train_loss/(len(train_dataset))))
    train_accs.append('{:.6f}'.format(train_acc/(len(train_dataset))))


eval_losses = []
eval_accs = []

def evaluate():
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        eval_acc = 0
        for data in test_loader:
            data.to(device)
            out = model(data.pos, data.batch)
            loss = F.nll_loss(out, data.y)
            # print(out)
            pred = out.max(1)[1]
            # print(pred)
            eval_loss += loss.data.item()*data.y.size(0)
            eval_acc += pred.eq(data.y).sum().item()
        eval_losses.append('{:.6f}'.format(eval_loss / (len(test_dataset))))
        eval_accs.append('{:.6f}'.format(eval_acc / (len(test_dataset))))
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
                    eval_loss / (len(test_dataset)),
                    eval_acc / (len(test_dataset))
        ))
        return eval_acc/(len(test_dataset))

'''
def writeToFile(mode, epoch):
    assert mode in ['train', 'eval']
    fl = open('./'+mode +'_loss_acc.txt','a+')
    if epoch == 2:
        fl.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n")
    fl.write("epoch: " + str(epoch)+"\n")
    for i in range(len(eval(mode+"_losses"))):
        fl.write(str(eval(mode+"_losses")[i])+ "\t" + str(eval(mode+"_accs")[i])+ "\n")
    eval(mode + "_losses").clear()
    eval(mode + "_accs").clear()
    fl.close()
'''

def writeToFileOne(epoch):
    fl = open(os.path.join(log_save_dir, "loss_acc_"+str(NUM_POINT)+".txt"),'a+')
    if epoch == 0:
        fl.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n")
        fl.write("train_loss\t train_acc\t eval_loss\t eval_acc\n")
        return
    # fl.write("epoch: " + str(epoch)+"\n")
    for i in range(len(train_losses)):
        fl.write(train_losses[i]+ "\t" + train_accs[i]+ "\t" + eval_losses[i] +"\t"+ eval_accs[i] + "\n")
    train_losses.clear()
    train_accs.clear()
    eval_losses.clear()
    eval_accs.clear()
    fl.close()

if __name__ == '__main__':
    evaluate()
    writeToFileOne(0)
    for epoch in range(1, NUM_EPOCHS+1):
        train(epoch)
        acc=0
        if epoch % 1 == 0:
            acc=evaluate()
        if epoch % EPOCHS_TO_WRITE == 0:
            # writeToFile("train", epoch)
            # writeToFile("eval", epoch)
            writeToFileOne(epoch)
            torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)+"_"+str(acc)+"_"+str(NUM_POINT)))
