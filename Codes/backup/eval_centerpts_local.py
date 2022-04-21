import argparse

import os
import sys
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import provider
import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from center_est_pt import CenterEstimateNN

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


random.seed(0)


# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
# parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
# parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
# parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
# LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
# MOMENTUM = FLAGS.momentum
# EPOCHS_TO_WRITE = 1
       
# MAX_NUM_POINT = 1024

# DECAY_STEP = FLAGS.decay_step
# DECAY_RATE = FLAGS.decay_rate
# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)
# BN_DECAY_CLIP = 0.99


# LEARNING_RATE_MIN = 0.00001
        
NUM_CLASS = 8
BATCH_SIZE = FLAGS.batch_size #32
# NUM_EPOCHS = FLAGS.max_epoch

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(GPU_INDEX)
    dtype = torch.cuda.FloatTensor
else:
    device = torch.device('cpu')
    dtype = torch.FloatTensor

print("------Building model and loading params-------")

cemodel = CenterEstimateNN(NUM_CLASS).to(device)
# cemodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'center_pt1024_Huber', '500_0.0002767378449789248_1024'), map_location='cpu'))
cemodel.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'center_1024_old_2', '200_0.0008424578688573092_1024'), map_location='cpu'))
print("------Successfully Built model-------")


features_save_dir = os.path.join(BASE_DIR, "features")
os.makedirs(features_save_dir, exist_ok = True)

# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))

test_dataset= provider.PCDDataset(BASE_DIR, "test", None, None)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def getPredictCPts():
    cemodel.eval()
    err_sum = np.zeros(3, dtype=float);
    cnt = 0;
    with torch.no_grad():
        for data in test_loader:
            cnt=cnt+1;
            data.to(device)
            span = (torch.max(data.pos,0)[0]-torch.min(data.pos,0)[0]).cpu().numpy()
            # print(span)
            out = cemodel(data.pos, torch.zeros(1024, dtype = torch.int64))
            # print("label:",data.center)
            # print("predict cpt:",out)
            res = (torch.abs(data.center-out)).cpu().numpy()
            err = res/span
            err_sum=err_sum+err
            # square_res = np.square(res)[0]
            # dist_sum= dist_sum+np.sqrt(sum(square_res))
    # average_euclidean_dist=dist_sum/cnt
    average_euclidean_dist=0
    print(err_sum/cnt)
    return average_euclidean_dist

if __name__ == '__main__':
    getPredictCPts()
