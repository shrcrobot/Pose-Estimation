######
# rotation + center 同时优化
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_save_dir = os.path.join(BASE_DIR, "logs")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(GPU_INDEX)
    print("GPU:"+str(GPU_INDEX))
else:
    dtype = torch.FloatTensor


print("------Building model and loading-------")

rotmodel = RotationEstimateNN(NUM_CLASS).to(device)
rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', '1_rotation_cls_entropy_box_residual_25_11:49:26', '130')))
# rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'rotation_cls_entropy_box_new_19_12:31:03', '1195')))
# rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', '1_huber_rotation_cls_entropy_box_residual_20_10:10:40', '160')))

print("------Successfully Built model-------")



TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/test_files.txt'))

train_dataset= provider.PCDDataset(BASE_DIR, "train", None, pre_transform= False)
test_dataset= provider.PCDDataset(BASE_DIR, "test", None, pre_transform= False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(train_dataset)
print(test_dataset)

def to_residual(angle):
    return angle%30

def to_cls(angle):
    return (( (angle) // 30)%12).long()

train_losses = []
train_losses_cls = []


eval_losses = []
eval_losses_cls = []
psi_acc = []
theta_acc = []
phi_acc = []
total_angle_acc = []


############

def evaluate():
    rotmodel.eval()
    cnt = 0
    err = torch.FloatTensor(1,3).zero_()
    all_cnt = 0
    all_err = torch.FloatTensor(1,3).zero_()
    with torch.no_grad():
        eval_loss = 0
        eval_loss_cls = 0
        eval_acc_psi = 0
        eval_acc_theta = 0
        eval_acc_phi = 0
        eval_acc_total = 0
        fl_rot = open(os.path.join(log_save_dir, "angle_err"), 'a+')
        fl_rot_cls = open(os.path.join(log_save_dir, "angle_err_cls"), 'a+')
        for data in test_loader:
            data.to(device)
            rotout = rotmodel(data.pos,data.center, data.batch).to(device)

            #############
            # angle residual
            res_psi = rotout[:, 36]
            res_theta = rotout[:, 37]
            res_phi = rotout[:, 38]

            relu_res_psi = torch.sigmoid(rotout[:, 36]) * math.pi / 6
            relu_res_theta = torch.sigmoid(rotout[:, 37]) * math.pi / 6
            relu_res_phi = torch.sigmoid(rotout[:, 38]) * math.pi / 6

            cls_psi = F.log_softmax(rotout[:, 0:12], dim=-1)
            cls_theta = F.log_softmax(rotout[:, 12:24], dim=-1)
            cls_phi = F.log_softmax(rotout[:, 24:36], dim=-1)
            angle = data.angle.view((data.num_graphs, 3))
            label = to_cls(angle)
            angle = (angle*math.pi/180)%(2*math.pi)

            # cls 优化
            pred_psi = ((cls_psi.max(1)[1].float() * 30) * math.pi / 180 + relu_res_psi).reshape(data.y.size()[0], -1)
            pred_theta = ((cls_theta.max(1)[1].float() * 30) * math.pi / 180 + relu_res_theta).reshape(data.y.size()[0], -1)
            pred_phi = ((cls_phi.max(1)[1].float() * 30) * math.pi / 180 + relu_res_phi).reshape(data.y.size()[0], -1)

            pred_angle = torch.cat((pred_psi, pred_theta, pred_phi), 1)

            #############
            # print((angle - pred_angle).abs().sum(dim=0).view(1,3))
            cur_err = (angle - pred_angle).abs().sum(dim=0).cpu()
            for i in range(cur_err.size()[0]):
                if cur_err[i]>math.pi:
                    # print("!!")
                    cur_err[i]=2*math.pi - cur_err[i]
            cur_err_d = cur_err*180/math.pi
            fl_rot.write(str(cur_err_d[0].item()) + "\t" + str(cur_err_d[1].item()) + "\t" + str(
                cur_err_d[2].item()) + "\n")
            all_err = all_err + cur_err
            all_cnt = all_cnt + data.num_graphs

            pred_psi_cls = cls_psi.max(1)[1]
            pred_theta_cls = cls_theta.max(1)[1]
            pred_phi_cls = cls_phi.max(1)[1]

            if (pred_psi_cls.eq(label[:,0]) & pred_theta_cls.eq(label[:,1]) & pred_phi_cls.eq(label[:,2])):
                cnt=cnt+1
                # print(angle)
                # print(pred_angle)
                cur_err = (angle - pred_angle).abs().cpu()
                cur_err_d = cur_err*180/math.pi
                fl_rot_cls.write(str(cur_err_d[0][0].item()) + "\t" + str(cur_err_d[0][1].item()) + "\t" + str(cur_err_d[0][2].item()) + "\n")
                err = err + cur_err

            ############
        print("err:")
        print((err/cnt))

        print("all_err:")
        print((all_err/all_cnt))
        fl_rot_cls.close()
        fl_rot.close()

    return eval_loss/(len(test_dataset))


############



if __name__ == '__main__':
    evaluate()