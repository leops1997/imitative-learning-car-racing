import numpy as np
import matplotlib.pyplot as plt
import gym
import cnn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transform

def load_data():

    path = "data/"
    actions = np.load(path+"actions.npy")
    states = np.load(path+"states.npy")

    x = torch.Tensor(states)
    y = torch.Tensor(actions)
    x = x.permute(0,3,1,2)

    print(x.shape)

    dataset = TensorDataset(x,y)

    split = 0.75
    train_batch = int(split * len(dataset))
    test_batch = len(dataset) - train_batch

    train_dataset, test_dataset = random_split(dataset, [train_batch, test_batch])

    trainloader = DataLoader(train_dataset)
    testloader = DataLoader(test_dataset)


    return trainloader, testloader

def agent():

    env = gym.make("CarRacing-v2", render_mode="human")
    s = env.reset(seed=42) #choosing 42th map
    terminated = False
    while not terminated:
        env.render()
        action = [0, .5, 0]
        observation, reward, terminated, truncated, info = env.step(action)
        s = observation
    env.close()

if __name__=="__main__":
    train_dataset, test_dataset = load_data()
    net = cnn.Net()
    net.train(train_dataset, test_dataset, 10, "data/")
    # agent()