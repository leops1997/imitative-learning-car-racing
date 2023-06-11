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
    actions = torch.from_numpy(np.load(path+"actions.npy")).to(torch.float32)
    states = torch.from_numpy(np.load(path+"states.npy")).to(torch.float32)

    dataset = TensorDataset(states.transpose(1,3),actions)

    split = 0.75
    train_batch = int(split * len(dataset))
    test_batch = len(dataset) - train_batch

    train_dataset, test_dataset = random_split(dataset, [train_batch, test_batch])
    
    batch_size = 1
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Current device: {device}")
    train_dataset, test_dataset = load_data()
    net = cnn.Net()
    net.train(train_dataset, test_dataset, 10, "data/")
    # agent()