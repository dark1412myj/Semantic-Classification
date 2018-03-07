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
        self.fc = nn.Linear(hidesize*2,4)

    def forward(self,lx,rx):
        #batch* seq_len
        lx,_ = self.lrnn(lx)
        rx,_ = self.rrnn(rx)
        y = torch.cat((lx[:,-1,:],rx[:,1,:]), 1 )
        y = self.fc(y)
        y = nn.functional.softmax(y,dim=1)
        return y


def train(model,loader,epoch=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = nn.CrossEntropyLoss()
    for _ in range(epoch):
        for (lx,rx, y_) in loader:
            y_ = Variable(y_)
            lx = Variable(lx)
            rx = Variable(rx)
            y = model(lx,rx)
            loss = loss_fun(y,y_.max(1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def test(model,lx,rx,y_):
    y = model(Variable(torch.Tensor(lx)),Variable(torch.Tensor(rx)) )
    y = y.max(1)[1]

if __name__ == '__main__':
    model = Object_Semantic(100,128)
    train(model,train_loader)