import cnn
import imitate
import matplotlib.pyplot as plt
import sys
import torch
import gym

if len(sys.argv) >= 4:
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    track = int(sys.argv[3])
else:
    epochs = 20
    batch_size = 1
    track = 42

data_path = "data/"

def manage_neural_network(net):
    net.train() # Set the net to training mode
    net.train_model(epochs, "data/")

def setup_game():
    env = gym.make("CarRacing-v2", render_mode="human")
    net.load_state_dict(torch.load("data/trained_model.pth"))
    net.eval()
    return env

def plot_results():
    pass

if __name__=="__main__":
    net = cnn.Net(batch_size)
    manage_neural_network(net)
    plot_results()
    env = setup_game()
    agent = imitate.Imitative_Agent(net, env, track)
    agent.play_game() # And let the agent have fun =)
