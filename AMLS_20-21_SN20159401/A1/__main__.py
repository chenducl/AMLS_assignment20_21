from sklearn.svm import SVC
from sklearn import metrics
import torch as pt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

size_batch = 40

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
  

class A1_CNN(nn.Module):
    def __init__(self):
        super(A1_CNN, self).__init__()
        self.dense = nn.Sequential(nn.Conv2d(3, 4, 3),
                                   nn.BatchNorm2d(4),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, 3),
                                   nn.MaxPool2d(kernel_size=(2, 2)),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8),
                                   nn.Conv2d(8, 8, 3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d(kernel_size=(2, 2)),
                                   nn.Conv2d(8, 4, 3),
                                   nn.ReLU(),
                                   Flatten(),
                                   nn.Linear(8364, 256), nn.PReLU(),
                                   nn.Dropout(0.6),
                                   nn.Linear(256, 2))

    def forward(self, x):
        return self.dense(x)

class A1(object):
    def __init__(self, use_CNN):
        super().__init__()
        self.use_CNN = use_CNN
        if use_CNN:
            self.model = A1_CNN().cuda()
        else:
            self.model = SVC(kernel='poly')

    def train(self, train, valid):
        if self.use_CNN:
            # init loss function and optimzer
            loss = nn.CrossEntropyLoss()
            opt = optim.Adam(self.model.parameters(), lr=0.001)

            best_acc = 0.0

            for epoch in range(20):
                running_loss = []
                for x,y in train:

                    # transfer data from memory to GPU
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()

                    # for some unknow reason, need to make the x float type
                    x = x.float()
                    
                    # set train mode
                    self.model.train()
                    yy = self.model(x)
                    # because the label is in {-1, 1} need to convert to {0, 1}
                    y = (y+1)//2

                    # calculate loss and backpropagation
                    l = loss(yy, y)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                    running_loss.append(float(l))

                running_acc = []
                for x, y in valid:

                    # transfer data from memory to GPU
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()

                    # set evaluation mode
                    self.model.eval()

                    # make the x float type
                    x = x.float()
                    yy = self.model(x)
                    yy = pt.argmax(yy, dim=1)
                    y = (y+1)//2
                    acc = float(pt.sum(yy == y)) / size_batch
                    running_acc.append(acc)

                print("#summary:% 4d %.4f %.2f%%" % (epoch, np.mean(running_loss), np.mean(running_acc)*100))
                if np.mean(running_acc) > best_acc:
                    best_acc = np.mean(running_acc)
                    pt.save(self.model.state_dict(), 'A1_CNN.pkl')
            return best_acc

        else:
            self.model.fit(train[0], train[1])
            pred = self.model.predict(valid[0])
            return metrics.accuracy_score(valid[1], pred)

    def test(self, test):
        if self.use_CNN:
            self.model.load_state_dict(pt.load('A1_CNN.pkl'))
            test_acc = []

            for x, y in test:
                # transfer data from memory to GPU
                x = Variable(x).cuda()
                y = Variable(y).cuda()

                # set evaluation mode
                self.model.eval()

                # make the x float type
                x = x.float()
                yy = self.model(x)
                yy = pt.argmax(yy, dim=1)
                y = (y+1)//2
                acc = float(pt.sum(yy == y)) / size_batch
                test_acc.append(acc)

            return np.mean(test_acc)

        else:
            pred = self.model.predict(test[0])
            return metrics.accuracy_score(test[1], pred)


