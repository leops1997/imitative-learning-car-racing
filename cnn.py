import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Build network        
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(8,8), stride=4)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 3)
        self.act4 = nn.Sigmoid()
        
        self.loss_function = nn.BCELoss() # Binary Cross Entropy
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001) # Optimization algorithm
        
        # Compute model on selected device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Current device: {self.device}")
        self.to(self.device) 

    def forward(self, x):
        x = x.to(self.device) # Send data to selected device
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) # Linear needs 1 demensional vector
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x
    
    def train_model(self, trainset, testset, epochs, path):
        """
        Train model with the trainset.
        Test model with the testset.
        Train the model by the amount of epochs.
        Save model on "path/trained_model.pth"
        :param trainset: training dataset
        :param testset: test dataset
        :param epochs: number os epochs
        :param path: path to save the model
        :return cost: List with the cost of each epoch
        :return accuracy: List with the accuracy of each epoch
        """
        epoch_accuracy = 0
        epoch_loss = 0
        accuracy = []
        cost = []
        for i in range(epochs):
            for x, y in trainset:

                # Send data to selected device
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))

                # Forward
                y_pred = self(x)
                target_normalized = torch.sigmoid(y) # Put target values between 0 and 1
                loss = self.loss_function(y_pred, target_normalized)
                self.optimizer.zero_grad() # Clear accumulated gradients

                epoch_loss += loss.item() # Sum loss

                # Backward
                loss.backward()
                self.optimizer.step() # Update parametes
    
            for x, y in testset:
                # Send data to selected device
                x = x.to(torch.device(self.device))
                y = y.to(torch.device(self.device))
                y_pred = self(x)

                epoch_accuracy += ((torch.sigmoid(y) - y_pred) ** 2).sum().item() # Mean squared error

            epoch_accuracy /= len(testset.dataset) # Mean squared error
            epoch_loss /= len(trainset.dataset) # Total loss

            accuracy.append(epoch_accuracy) # Records the accuracy for each epoch
            cost.append(epoch_loss) # Records the loss for each epoch
            
            print(f"Epoch {i}: loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        torch.save(self.state_dict(), path+"trained_model.pth")

        return cost, accuracy