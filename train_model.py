import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import get_data


class BioDataSet(data.Dataset): 
    def __init__(self,data_X,data_Y):
        super(BioDataSet,self).__init__()
        self.data_x = torch.Tensor(data_X)
        self.data_y = torch.Tensor(data_Y)
    
    def __getitem__(self,index):
        return self.data_x[index],self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)


train_X,train_Y,test_X,test_Y  = get_data.return_eoa_data()

train_loader = torch.utils.data.DataLoader(
    BioDataSet(train_X,train_Y),
    batch_size = 10,
    shuffle = False,num_workers = 1)


# test_loader = torch.utils.data.DataLoader(
# 		BioDataSet(test_X,test_Y),
# 		batch_size = 10,
# 		shuffle = False,
# 		num_workers = 1)


class Bio(nn.Module):
    def __init__(self,wordsize,hidesize):
        super(Bio,self).__init__()
        self.rnn = nn.LSTM(wordsize,hidesize,batch_first=True)
        self.rnn2 = nn.LSTM(hidesize,hidesize,batch_first=True)
        self.classifier = nn.Linear(hidesize,4)

    def forward(self,x):
        x,_ = self.rnn(x)
        x = F.dropout(x,0.5)
        x,_ = self.rnn2(x)
        x = F.dropout(x,0.5)
        outs = []
        for i in range(x.size(1)):
            outs.append( self.classifier(x[:,i,:]) )
        return torch.stack(outs, dim=1)


def train(model,loader,epoch=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = nn.CrossEntropyLoss()
    for _ in range(epoch):
        for (x, y_) in loader:
            y_ = Variable(y_)
            x = Variable(x)
            y = model(x)
            loss = 0
            for i in range(y.size(1)):
                prd_ = y_[:, i, :].max(1)[1]
                loss += loss_fun(y[:, i, :], prd_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def test(model,x,y_):
    y = model(Variable(torch.Tensor(test_X)))
    y = y.max(2)[1]
    seq_len = get_data.get_input_len(test_X)
    total_correct = 0
    total = 0
    sentence_correct = 0
    sentence = y.size(0)
    for i in range(y.size(0)):
        flag = True
        for j in range(seq_len[i]):
            if y_[i][j].data[0] == y[i][j].data[0]:
                total_correct += 1
            else:
                flag = False
            total += 1
        if flag:
            sentence_correct += 1
    print(total_correct, '/', total)
    print(sentence_correct, '/', sentence)


if __name__ == "__main__":
    model = Bio(50,128)
    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)
    y_ = torch.Tensor(test_Y)
    y_ = Variable(y_.max(2)[1])#np.argmax(test_Y, 2)
    test(model, test_X, y_)
    for _ in range(6):
        model = train(model, train_loader)
        test(model, test_X, y_)
