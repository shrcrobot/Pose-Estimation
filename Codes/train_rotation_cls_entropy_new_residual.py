######
# 每轴单独隐层
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
rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'rotation_cls_entropy_box_new_19_12:31:03', '1195')))
# rotmodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', '1_huber_rotation_cls_entropy_box_residual_20_10:10:40', '160')))

print("------Successfully Built model-------")

model_save_dir = os.path.join(BASE_DIR, "models", str(GPU_INDEX)+"_rotation_cls_entropy_box_residual_"+datetime.datetime.now().strftime('%d_%H:%M:%S'))
log_save_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(model_save_dir, exist_ok = True)
os.makedirs(log_save_dir, exist_ok = True)

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/test_files.txt'))


# pre_transform, transform = T.NormalizeScale(), T.SamplePoints(NUM_POINT)
train_dataset= provider.PCDDataset(BASE_DIR, "train", None, pre_transform= False)
# print(train_dataset.ang_m, train_dataset.ang_range , train_dataset.ctr_m , train_dataset.ctr_range)
test_dataset= provider.PCDDataset(BASE_DIR, "test", None, pre_transform= False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(rotmodel.parameters(), lr = 0.001)

print(train_dataset)
print(test_dataset)

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
        a = 1
        b = 2
        c = 3
        cords_np = np.array([[-1,-1,1,1],[-1,1,1,1],[1,1,1,1],
                             [1,-1,1,1],[-1,-1,-1,1],[-1,1,-1,1],
                             [1,1,-1,1],[1,-1,-1,1]], dtype=np.float32)
        # cords_np = np.array([[-b,a,c,1],[b,a,c,1],[-b,-a,c,1],
        #                      [b,-a,c,1],[-b,a,-c,1],[b,a,-c,1],
        #                      [-b,-a,-c,1],[b,-a,-c,1]], dtype=np.float32)
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

criterion = cubic_loss().to(device)

##################




def to_residual(angle):
    return angle%30

def to_cls(angle):
    return (( (angle) // 30)%12).long()


train_losses = []
train_losses_cls = []

def train(epoch):
    rotmodel.train()
    global_step = 1
    train_loss = 0
    train_loss_cls = 0

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        rotout = rotmodel(data.pos, data.center ,data.batch).to(device)
        angle_label = data.angle.view((data.num_graphs, 3))
        #############
        # angle residual
        res_psi = rotout[:,36]
        res_theta = rotout[:,37]
        res_phi = rotout[:,38]

        relu_res_psi = torch.sigmoid(rotout[:,36])* math.pi/6
        relu_res_theta = torch.sigmoid(rotout[:,37])* math.pi/6
        relu_res_phi = torch.sigmoid(rotout[:,38])* math.pi/6


        # res_angle = torch.cat((,,),1)
        # print(res_angle.size())

        cls_psi = F.log_softmax(rotout[:,0:12], dim=-1)
        cls_theta = F.log_softmax(rotout[:,12:24], dim=-1)
        cls_phi = F.log_softmax(rotout[:,24:36], dim=-1)


        label = to_cls(angle_label)

        ##### 只优化拟合
        #
        # pred_psi = ((label[:,0].float()*30) * math.pi /180 + relu_res_psi).reshape(data.y.size()[0], -1)
        # pred_theta = ((label[:,1].float()*30) * math.pi /180 + relu_res_theta).reshape(data.y.size()[0], -1)
        # pred_phi = ((label[:,2].float()*30) * math.pi /180 + relu_res_phi).reshape(data.y.size()[0], -1)
        #

        ##### cls 优化
        pred_psi = ((cls_psi.max(1)[1].float() * 30) * math.pi / 180 + res_psi).reshape(data.y.size()[0], -1)
        pred_theta = ((cls_theta.max(1)[1].float() * 30) * math.pi / 180 + res_theta).reshape(data.y.size()[0], -1)
        pred_phi = ((cls_phi.max(1)[1].float() * 30) * math.pi / 180 + res_phi).reshape(data.y.size()[0], -1)

        pred_angle = torch.cat((pred_psi,pred_theta,pred_phi),1)
        #############


        # huber_pred_psi = cls_psi.max(1)[1].float() * 30
        # huber_pred_theta = cls_theta.max(1)[1].float() * 30
        # huber_pred_phi = cls_phi.max(1)[1].float() * 30
        #
        # huber_pred_angle = torch.cat((huber_pred_psi,huber_pred_theta,huber_pred_phi),1)

        # print(huber_pred_angle.size())
        #############
        center = data.center.reshape(data.y.size()[0], -1)
        angle = data.angle.reshape(data.y.size()[0], -1)
        angle_rad = data.angle.reshape(data.y.size()[0], -1)*math.pi/180

        merged = torch.cat((center, angle_rad), 1)
        out = torch.cat((center, pred_angle), 1)
        loss = criterion(out, merged)
        #loss_cls = F.nll_loss(cls_psi, to_cls(angle[:, 0])) + F.nll_loss(cls_theta, to_cls(angle[:, 1])) + F.nll_loss(cls_phi, to_cls(angle[:, 2]))
        loss_cls = loss
        # huber_target =(angle_label-huber_pred_angle)* math.pi / 360
        # print(huber_target)
        # loss_huber =criterion_huber(res_angle, huber_target)
        #############


        #angle = data.angle.view((data.num_graphs,3))
        #loss = F.nll_loss(cls_psi, to_cls(angle[:,0]))+F.nll_loss(cls_theta, to_cls(angle[:,1]))+F.nll_loss(cls_phi, to_cls(angle[:,2]))
        loss.backward(retain_graph=True)
        # loss_cls.backward()
        # loss.backward()
        # loss_huber.backward(retain_graph=True)


        optimizer.step()

        print("epoch: "+str(epoch) + "   loss: "+str(loss.data) + "     loss_cls:"+ str(loss_cls.data))
        global_step += 1
        train_loss += loss.data.item() * data.num_graphs
        train_loss_cls += loss_cls.data.item() * data.num_graphs
    train_losses.append('{:.6f}'.format(train_loss/(len(train_dataset))))
    train_losses_cls.append('{:.6f}'.format(train_loss_cls/(len(train_dataset))))

eval_losses = []
eval_losses_cls = []
psi_acc = []
theta_acc = []
phi_acc = []
total_angle_acc = []

def evaluate():
    rotmodel.eval()
    with torch.no_grad():
        eval_loss = 0
        eval_loss_cls = 0
        eval_acc_psi = 0
        eval_acc_theta = 0
        eval_acc_phi = 0
        eval_acc_total = 0
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

            # cls 优化
            pred_psi = ((cls_psi.max(1)[1].float() * 30) * math.pi / 180 + relu_res_psi).reshape(data.y.size()[0], -1)
            pred_theta = ((cls_theta.max(1)[1].float() * 30) * math.pi / 180 + relu_res_theta).reshape(data.y.size()[0], -1)
            pred_phi = ((cls_phi.max(1)[1].float() * 30) * math.pi / 180 + relu_res_phi).reshape(data.y.size()[0], -1)
            label = to_cls(angle)

            # 拟合优化
            # pred_psi = ((label[:, 0].float() * 30) * math.pi / 180 + relu_res_psi).reshape(data.y.size()[0], -1)
            # pred_theta = ((label[:, 1].float() * 30) * math.pi / 180 + relu_res_theta).reshape(data.y.size()[0], -1)
            # pred_phi = ((label[:, 2].float() * 30) * math.pi / 180 + relu_res_phi).reshape(data.y.size()[0], -1)

            pred_angle = torch.cat((pred_psi, pred_theta, pred_phi), 1)
            #############
            center = data.center.reshape(data.y.size()[0], -1)
            angle_rad = data.angle.reshape(data.y.size()[0], -1) * math.pi / 180

            merged = torch.cat((center, angle_rad), 1)
            out = torch.cat((center, pred_angle), 1)

            loss = criterion(out, merged)
            # loss_cls = F.nll_loss(cls_psi, to_cls(angle[:, 0])) + F.nll_loss(cls_theta, to_cls(angle[:, 1])) + F.nll_loss(
            #     cls_phi, to_cls(angle[:, 2]))
            loss_cls = loss

            #############

            eval_loss += loss.data.item() * data.num_graphs
            eval_loss_cls += loss_cls.data.item() * data.num_graphs

            pred_psi = cls_psi.max(1)[1]
            pred_theta = cls_theta.max(1)[1]
            pred_phi = cls_phi.max(1)[1]
            label = to_cls(angle)
            eval_acc_psi += pred_psi.eq(label[:,0]).sum().item()
            eval_acc_theta += pred_theta.eq(label[:,1]).sum().item()
            eval_acc_phi += pred_phi.eq(label[:,2]).sum().item()

            total_acc = (pred_psi.eq(label[:,0]) & pred_theta.eq(label[:,1]) & pred_phi.eq(label[:,2])).sum().item()
            eval_acc_total += total_acc
            # print(eval_loss)
        eval_losses.append('{:.6f}'.format(eval_loss / (len(test_dataset))))
        eval_losses_cls.append('{:.6f}'.format(eval_loss_cls / (len(test_dataset))))
        psi_acc.append('{:.6f}'.format(eval_acc_psi / (len(test_dataset))))
        theta_acc.append('{:.6f}'.format(eval_acc_theta / (len(test_dataset))))
        phi_acc.append('{:.6f}'.format(eval_acc_phi / (len(test_dataset))))
        total_angle_acc.append('{:.6f}'.format(eval_acc_total / (len(test_dataset))))
        print('Test Loss: {:.6f}'.format(eval_loss / (len(test_dataset))))
        print('Test Loss cls: {:.6f}'.format(eval_loss_cls / (len(test_dataset))))
        print('Test psi acc: {:.6f}'.format(eval_acc_psi / (len(test_dataset))))
        print('Test theta acc: {:.6f}'.format(eval_acc_theta / (len(test_dataset))))
        print('Test phi acc: {:.6f}'.format(eval_acc_phi / (len(test_dataset))))
        print('Test total acc: {:.6f}'.format(eval_acc_total / (len(test_dataset))))
    return eval_loss/(len(test_dataset))

filename= str(GPU_INDEX)+ "_rotation_cls_entropy_box_residual_"+datetime.datetime.now().strftime('%d_%H:%M:%S')+".txt"

def writeToFileOne(epoch):
    fl = open(os.path.join(log_save_dir, filename),'a+')
    if epoch == 0:
        fl.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"\n")
        fl.write("train_loss\t train_loss_cls\t eval_loss\t eval_loss_cls\t psi\t theta \t phi \t total \n")
        return
    # fl.write("epoch: " + str(epoch)+"\n")
    for i in range(len(train_losses)):
        fl.write(train_losses[i]+ "\t" + train_losses_cls[i]+ "\t" + eval_losses[i]+ "\t" + eval_losses_cls[i]+ "\t" + psi_acc[i]+ "\t" + theta_acc[i]+ "\t" + phi_acc[i]+ "\t" + total_angle_acc[i] + "\n")
    train_losses.clear()
    train_losses_cls.clear()
    eval_losses.clear()
    eval_losses_cls.clear()
    psi_acc.clear()
    theta_acc.clear()
    phi_acc.clear()
    total_angle_acc.clear()
    fl.close()

if __name__ == '__main__':
    evaluate()
    writeToFileOne(0)
    for epoch in range(1, NUM_EPOCHS+1):
        train(epoch)
        if epoch % 1 == 0:
            eval_loss=evaluate()
        if epoch % EPOCHS_TO_WRITE == 0:
            writeToFileOne(epoch)
            torch.save(rotmodel.state_dict(), os.path.join(model_save_dir, str(epoch)))
