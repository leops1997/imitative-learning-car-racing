import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2 ,2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8,8), stride=4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 640)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(640, 3)
        self.act4 = nn.Sigmoid()

        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        # x = torch.flatten(x, 1)
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x
    
    def train(self, trainset, testset, epochs, path):
        for i in range(epochs):
            for x, y in trainset:
                print(y)
                y_pred = self(x)
                target = torch.tensor(y)
                target_scaled = (target - torch.min(target))/(torch.max(target)-torch.min(target))
                print(target_scaled)
                loss = nn.BCELoss(y_pred, target_scaled)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            accuracy = 0
            count = 0
            for x, y in testset:
                y_pred = self(y)
                accuracy += (torch.argmax(y_pred, 1) == y).float().sum()
                count += len(y)
            accuracy /= count
            print("Epoch %d: model accuracy %.2f%%" % (i, accuracy*100))
 
        torch.save(self.state_dict(), path)