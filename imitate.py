import cnn
import gym
import numpy as np
import torch

class Imitative_Agent():
    def __init__(self, net, env, model, device, track):
        self.net = net
        self.env = env
        self.model = model
        self.device = device
        self.track = track

    def play_game(self):
        self.env.reset(seed=self.track)
        action = [0, 0, 0]
        self.main_game_loop(action)
        self.env.close()
    
    def main_game_loop(self, action):
        terminated = False
        while not terminated:
            self.env.render()
            observation, reward, terminated, truncated, info = self.env.step(action)
            state = observation
            data = torch.from_numpy(state.copy()).to(torch.float32) 
            data = data.unsqueeze(0)
            data = data.transpose(1,3)
            action = torch.logit(self.net(data)).squeeze().tolist()

'''
net = cnn.Net()
env = gym.make("CarRacing-v2", render_mode="human")
model = net.load_trained_model("data/trained_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
track = 42

agent = Imitative_Agent(net, env, model, device, track)

agent.play_game()
'''