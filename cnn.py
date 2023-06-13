import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8,8), stride=4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 3)
        self.act4 = nn.Sigmoid()

        self.loss_function = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) #Linear needs 1 demensional vector
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x
    
    def train_model(self, trainset, testset, epochs, path):
        for i in range(epochs):
            for x, y in trainset:
                y_pred = self(x)
                target = torch.tensor(y).clone().detach().requires_grad_(True)
                target_normalized = torch.sigmoid(target)
                # print(target_normalized)
                # pred_normalized = torch.sigmoid(y_pred)
                pred_normalized = y_pred
                # print(torch.logit(pred_normalized))
                loss = self.loss_function(pred_normalized, target_normalized)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            accuracy = 0
            count = 0
            for x, y in testset:
                y_pred = self(x)
                
                accuracy += (torch.argmax(y_pred, 1) == y).float().sum()
                count += len(y)
            accuracy /= count
            print(loss)
            # print("Epoch %d: model accuracy %.2f%%" % (i, accuracy*100))
 
        torch.save(self.state_dict(), path+"trained_model.pth")