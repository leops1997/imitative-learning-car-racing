import numpy as np
import matplotlib.pyplot as plt
import gym
import cnn
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transform

def load_data():

    path = "data/"
    actions = np.load(path+"actions.npy", allow_pickle=True)
    states = np.load(path+"states.npy",  allow_pickle=True)
    print(states.shape[0])
    print(type(states[0][0][0][0]))

    # trans = transform.Compose([
    #     transform.ToPILImage(),
    #     transform.ToTensor()])
    
    # dataset = trans(states[100])
    # dataloader = DataLoader(dataset, 2, shuffle=True, num_workers=3, pin_memory=True)
    

    tensor_x = torch.Tensor(states) # transform to torch tensor
    tensor_y = torch.Tensor(actions)
    # print(tensor_x)

    dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    dataloader = DataLoader(dataset) # create your dataloader

    # fig, ax = plt.subplots(3,1)
    # for i in range(3):
    #     ax[i].plot(actions[:,i])
    # plt.show()

    # plt.imshow(dataset)
    # plt.show()
    return dataset

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
    data = load_data()
    net = cnn.Net()
    net.train(data, data, 10, "data/")
    # agent()