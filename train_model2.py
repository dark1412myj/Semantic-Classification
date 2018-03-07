import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import get_data

train_X_F, train_X_B, train_Y, test_X_F, test_X_B, test_Y = get_data.get_eosc_data()

class BioDataSet(data.Dataset):
    def __init__(self, data_lx,data_rx, data_Y):
        super(BioDataSet, self).__init__()
        self.data_lx = torch.Tensor(data_lx)
        self.data_rx = torch.Tensor(data_rx)
        self.data_y = torch.Tensor(data_Y)

    def __getitem__(self, index):
        return self.data_lx[index],self.data_rx[index], self.data_y[index]

    def __len__(self):
        return len(self.data_lx)

train_loader = torch.utils.data.DataLoader(
    BioDataSet(train_X_F,train_X_B,train_Y),
    batch_size = 10,
    shuffle = False,num_workers = 1)


class Object_Semantic(nn.Module):
    def __init__(self,wordsize,hidesize):
        super(Object_Semantic,self).__init__()
        self.lrnn = nn.LSTM(wordsize,hidesize,batch_first=True)
        self.rrnn = nn.LSTM(wordsize, hidesize, batch_first=True)

    def forward(self,lx,rx):
        _,lx = self.lrnn(lx)
        _,rx = self.rrnn(rx)
        torch.cat((lx,rx), 1 )


if __name__ == '__main__':
    for (x, y, z) in train_loader:
        print(x.size())
        print(y.size())
        print(z.size())
        exit()