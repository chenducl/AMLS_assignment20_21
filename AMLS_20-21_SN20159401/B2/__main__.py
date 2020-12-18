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

class B2_CNN(nn.Module):
    def __init__(self):
        super(B2_CNN, self).__init__()
        self.dense = nn.Sequential(nn.Conv2d(4, 4, 1),
                                   nn.PReLU(),
                                   nn.BatchNorm2d(4),
                                   nn.Conv2d(4, 8, 3),
                                   nn.PReLU(),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d(kernel_size=(2, 2)),
                                   nn.Conv2d(8, 4, 3),
                                   nn.PReLU(),
                                   nn.Dropout(0.6),
                                   Flatten(),
                                   nn.Linear(244036, 5))

    def forward(self, x):
        return self.dense(x)
    

class B2(object):
    def __init__(self, use_CNN):
        super().__init__()
        self.use_CNN = use_CNN
        if use_CNN:
            self.model = B2_CNN().cuda()
        else:
            self.model = SVC(kernel='poly', decision_function_shape='ovo')

    def train(self, train, val):
        if self.use_CNN:
            # init loss function and optimzer
            loss = nn.CrossEntropyLoss()
            opt = optim.Adam(self.model.parameters(), lr=0.001)

            best_acc = 0.0

            for epoch in range(10):
                running_loss = []
                for x,y in train:
                    
                    # transfer data from memory to GPU
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()
                    
                    # set train mode
                    self.model.train()
                    yy = self.model(x)

                    # calculate loss and backpropagation
                    l = loss(yy, y)
                    opt.zero_grad()
                    l.backward()
                    opt.step()
                    running_loss.append(float(l))

                # set running acc array as empty 
                running_acc = []
                for x, y in val:
                    
                    # transfer data from memory to GPU
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()
                    
                    # set evaluation mode
                    self.model.eval()
                    
                    yy = self.model(x)
                    yy = pt.argmax(yy, dim=1)

                    # calculate accuracy
                    acc = float(pt.sum(yy == y)) / size_batch
                    running_acc.append(acc)

                # print("#summary:% 4d %.4f %.2f%%" % (epoch, np.mean(running_loss), np.mean(running_acc)*100))

                if np.mean(running_acc) > best_acc:
                    # if current accuracy is better, save the model
                    best_acc = np.mean(running_acc)
                    pt.save(self.model.state_dict(), 'B2_CNN.pkl')
            return best_acc
        else:
            self.model.fit(train[0], train[1])
            pred = self.model.predict(val[0])
            return metrics.accuracy_score(val[1], pred)

    def test(self, test):
        if self.use_CNN:
            # load the model 
            self.model.load_state_dict(pt.load('B2_CNN.pkl'))
            test_acc = []

            for x, y in test:
                x = Variable(x).cuda()
                y = Variable(y).cuda()
                self.model.eval()

                yy = self.model(x)
                yy = pt.argmax(yy, dim=1)

                acc = float(pt.sum(yy == y)) / size_batch
                test_acc.append(acc)

            return np.mean(test_acc)
        else:
            pred = self.model.predict(test[0])
            return metrics.accuracy_score(test[1], pred)


