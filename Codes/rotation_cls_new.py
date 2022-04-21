import os
import torch
from torch import nn
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import XConv, fps, global_mean_pool
from classifier import Classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(1)
else:
    dtype = torch.FloatTensor

class RotationEstimateNN(nn.Module):

    def __init__(self, NUM_CLASS):
        super(RotationEstimateNN, self).__init__()
        self.NUM_CLASS = NUM_CLASS
        self.cls_model = Classifier(NUM_CLASS)
        self.cls_model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'models', 'classifier_1024', '117_0.98_1024')))
        self.cls_model.eval()
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

        self.linctr = Lin(3, 200)
        self.lin1 = Lin(400, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)
        self.lin_psi = Lin(128, 12+1)
        self.lin_theta = Lin(128, 12+1)
        self.lin_phi = Lin(128, 12+1)

    def forward(self, pos, ctr, batch):
        with torch.no_grad():
            cls_out = self.cls_model(pos, batch) #n*3
            pred = cls_out.max(1)[1].reshape(-1,1)
            cls = torch.zeros(pred.size()[0], self.NUM_CLASS).to(device)
            cls.scatter_(1, pred, 1)
        x = F.relu(self.pcnn1(None, pos, batch))
        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.pcnn2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.333)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.pcnn3(x, pos, batch))

        x = global_mean_pool(x, batch)

        center_pt=ctr.view((torch.max(batch)+1,3))
        c = F.relu(self.linctr(center_pt))
        x = torch.cat((x, cls), dim = 1)
        x = torch.cat((x, c), dim = 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin3(x))
        x1 = self.lin_psi(x)
        x2 = self.lin_theta(x)
        x3 = self.lin_phi(x)


        x=torch.cat((x1[:,0:12],x2[:,0:12],x3[:,0:12], x1[:,12:], x2[:,12:], x3[:,12:]),1)

        return x
