import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import get_data


class BioDataSet(data.Dataset):
    def __init__(self, data_X, data_Y):
        super(BioDataSet, self).__init__()
        self.data_x = torch.Tensor(data_X)
        self.data_y = torch.Tensor(data_Y)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


train_X, train_Y, test_X, test_Y = get_data.return_eoa_data()

# print( type(test_X) )
train_loader = torch.utils.data.DataLoader(
    BioDataSet(train_X, train_Y),
    batch_size=10,
    shuffle=False,
    num_workers=1)

test_loader = torch.utils.data.DataLoader(
    BioDataSet(test_X, test_Y),
    batch_size=10,
    shuffle=False,
    num_workers=1)


class Bio(nn.Module):
    def __init__(self, wordsize, hidesize):
        super(Bio, self).__init__()
        self.rnn = nn.LSTM(wordsize, hidesize, batch_first=True)
        self.rnn2 = nn.LSTM(hidesize, hidesize, batch_first=True)
        self.classifier = nn.Linear(hidesize, 4)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.dropout(x, 0.5)
        x, _ = self.rnn2(x)
        x = F.dropout(x, 0.5)
        outs = []
        for i in range(x.size(1)):
            outs.append(self.classifier(x[:, i, :]))
        return torch.stack(outs, dim=1)


ry = np.argmax(test_Y, 2)

if __name__ == "__main__":
    model = Bio(50, 128)
    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fun = nn.CrossEntropyLoss()
    for ep in range(10):
        for (x, y_) in train_loader:
            x = Variable(x)
            # y_ = Variable(y_)
            y = model(x)
            loss = 0
            for i in range(y.size(1)):
                prd_ = np.argmax(y_[:, i, :].numpy(), 1)
                prd_ = torch.from_numpy(prd_)
                loss += loss_fun(y[:, i, :], Variable(prd_))
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y = model(Variable(torch.Tensor(test_X)))
        y = np.argmax(y.data.numpy(), 2)
        eq = (ry == y)
        print(eq.sum())
        print(eq.shape[0] * eq.shape[1])
