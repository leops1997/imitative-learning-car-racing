import numpy as np
import matplotlib.pyplot as plt
import gym
import cnn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transform
import imitate

def load_data():
    """
    Load data from data/actions.npy and data/states.npy.
    Transform data to tensor and create a DataLoader.
    Split data between train data and test data.
    """

    # load data from .npy and trasform to tensor.
    path = "data/"
    actions = torch.from_numpy(np.load(path+"actions.npy")).to(torch.float32)
    states = torch.from_numpy(np.load(path+"states.npy")).to(torch.float32)
    dataset = TensorDataset(states.transpose(1,3),actions)
    
    # split data between train data and test data.
    split = 0.75
    train_batch = int(split * len(dataset))
    test_batch = len(dataset) - train_batch
    train_dataset, test_dataset = random_split(dataset, [train_batch, test_batch])

    # create a DataLoader.
    batch_size = 1
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def agent(net, track):
    """
    Start simulation.
    :param net: network
    :param track: car track to play
    """

    env = gym.make("CarRacing-v2", render_mode="human") # choosing simulator
    env.reset(seed=track) # choosing car track
    terminated = False
    action = [0, 0, 0]
    while not terminated: 
        env.render()
        observation, reward, terminated, truncated, info = env.step(action)
        state = observation # get state (image)

        # preprocess data 
        data = torch.from_numpy(state.copy()).to(torch.float32) 
        data = data.unsqueeze(0)
        data = data.transpose(1,3)

        # get resultant action from network
        action = torch.logit(net(data)).squeeze().tolist()
        print(action)
    env.close()

if __name__=="__main__":
    train_dataset, test_dataset = load_data() 
    net = cnn.Net()
    net.train() # training mode
    net.train_model(train_dataset, test_dataset, 20, "data/")
    net.load_state_dict(torch.load("data/trained_model.pth")) # load trained model
    net.eval() # evaluation mode
    with torch.no_grad(): # disable gradient calculation for inference
        env = gym.make("CarRacing-v2", render_mode="human")
        model = net.load_state_dict(torch.load("data/trained_model.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track = 42
        agent = imitate.Imitative_Agent(net, env, model, device, track)
        agent.play_game()
        #agent(net, 42)