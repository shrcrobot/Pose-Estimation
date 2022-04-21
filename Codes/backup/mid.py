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
from classifier import Classifier

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T


random.seed(0)


# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
# parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
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
model = Classifier(NUM_CLASS).to(device)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'classifier_1024', '117_0.98_1024')))
# if torch.cuda.device_count() > 1:
#     print("Using "+ str(torch.cuda.device_count() )+" GPUS" )
#     model = nn.DataParallel(model, device_ids=[0])
print("------Successfully Built model-------")

# model_save_dir = os.path.join(BASE_DIR, "models", "classifier_"+str(NUM_POINT))
# log_save_dir = os.path.join(BASE_DIR, "logs")
# os.makedirs(model_save_dir, exist_ok = True)
# os.makedirs(log_save_dir, exist_ok = True)

features_save_dir = os.path.join(BASE_DIR, "features")
os.makedirs(features_save_dir, exist_ok = True)

# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/threeclass/train_files.txt'))

train_dataset= provider.PCDDataset(BASE_DIR, "train", None, None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

def getMidFeatures():
    feature_file = open(os.path.join(features_save_dir, "features.txt"),'w')
    model.eval()
    features = torch.cuda.FloatTensor(0, 192)
    eval_acc = 0
    with torch.no_grad():
        for data in train_loader:
            data.to(device)
            out = model(data.pos, data.batch)
            features = torch.cat((features,model.feature),0)
            loss = F.nll_loss(out, data.y)
            pred = out.max(1)[1]
            # print(pred.eq(data.y).sum())
            eval_acc += pred.eq(data.y).sum().item()
            # print(loss.data)
            # print(features.size())
            # print(model.feature.dtype)
            # print(model.feature.shape)
    print('Acc: {:.6f}'.format(
        eval_acc / (len(train_dataset))
    ))
    np.savetxt(feature_file, features.cpu().numpy())
    feature_file.close()


if __name__ == '__main__':
    getMidFeatures()
