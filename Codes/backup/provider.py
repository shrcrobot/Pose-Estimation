import os
import sys
import numpy as np
import h5py

import torch
from torch_geometric.data import Dataset, Data, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    label = f['label'][:]
    angle = f['angle'][:]
    center = f['center'][:]
    return [data, label, angle, center]



class PCDDataset(Dataset):
    def __init__(self, root, dataset_name, transform= None, pre_transform= False, data_params= None):
        assert dataset_name in ['train', 'test']
        self.dataset_name = dataset_name
        self.root = root
        self.name = "PCDDataset-"+dataset_name
        self.pre_transform = pre_transform

        self.ang_m = None
        self.ctr_m = None
        self.ctr_range = None
        if dataset_name == 'test' and pre_transform == True and data_params != None:
            self.ang_m = data_params['ang_m']
            self.ctr_m = data_params['ctr_m']
            self.ctr_std = data_params['ctr_std']

        super(PCDDataset, self).__init__(root, transform, pre_transform)
        # super(PCDDataset, self).__init__(root, transform, pre_transform)
        # self.data = loadDataFile(self.raw_file_names)
        # self.train_filename=os.path.join(root,'data/threeclass/train_files.txt')
        # self.test_filename=os.path.join(root,'data/threeclass/train_files.txt')

    @property
    def raw_file_names(self):
        # print("raw_file_names")
        return getDataFiles(os.path.join(self.root, 'data/threeclass/'+ self.dataset_name + '_files.txt'))[0]
        # return [line.rstrip() for line in open(self.train_filename)]

    @property
    def processed_file_names(self):
        # print("processed_file_names")
        return self.raw_file_names
        # return ['data_1.pt', 'data_2.pt']

    def __len__(self):
        return len(self.data[0])

    def _download(self):
        pass

    def pre_trans_ang(self, input):
        input = input / 180 * np.pi
        self.ang_m = np.mean(input)
        return (input - self.ang_m) / (2 * np.pi)

    def pre_trans_center(self, input):
        # print(input[0])
        self.ctr_m = np.mean(input, axis=0)
        # print(self.ctr_m)
        mx = np.max(input, axis=0)
        mn = np.min(input, axis=0)
        # self.ctr_range = mx-mn
        self.ctr_std = np.std(input, axis=0)
        # print(self.ctr_range)
        # print(((input - self.ctr_m) /self.ctr_std)[0])
        return (input - self.ctr_m) /self.ctr_std

    def process(self):
        self.data = loadDataFile(self.raw_file_names)
        if self.pre_transform == True:
            if self.dataset_name == 'train':
                self.data[2] = self.pre_trans_ang(self.data[2]) # angle
                self.data[3] = self.pre_trans_center(self.data[3]) # center
            else:
                self.data[2] = ((self.data[2]/180 *np.pi) - self.ang_m) /self.ang_range
                self.data[3] = (self.data[3] - self.ctr_m) / self.ctr_std # center
        self.process_set()


    def process_set(self):
        # print("process_set")
        self.data_list = []
        for idx in range(self.__len__()):
            # label = torch.from_numpy(self.data[1][idx]).long()
            # angle = torch.from_numpy(self.data[2][idx]).float()
            # center = torch.from_numpy(self.data[3][idx]).float()
            # batnum * 3
            d = Data(x= None, y = torch.from_numpy(self.data[1][idx]).long(), pos = torch.from_numpy(self.data[0][idx]).float())
            d.angle = torch.from_numpy(self.data[2][idx]).float()
            d.center = torch.from_numpy(self.data[3][idx]).float()
            # d.pos=self.data[0][idx] #pos
            # d.y=self.data[1][idx] #label
            self.data_list.append(d)

        if self.pre_filter is not None:
            self.data_list = [d for d in self.data_list if self.pre_filter(d)]

        # if self.pre_transform is not None:
        #     self.data_list = [self.pre_transform(d) for d in self.data_list]
        return

    def get(self, idx):
        return self.data_list[idx]
    # def __repr__(self):
    #     return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

# print(BASE_DIR)
# dataset= PCDDataset(BASE_DIR)
# print(dataset[0])  # 1024*3
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# print(train_loader)
# for data in train_loader:
#     print(data.pos,data.batch)

