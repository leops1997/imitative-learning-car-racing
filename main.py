import cnn
import gym
import imitate
import matplotlib.pyplot as plt
import sys
import torch

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
    net.load_state_dict(torch.load(data_path + "trained_model.pth")) # Then, we load the trained model
    net.eval() # And set the net to evaluation mode

def setup_game():
    env = gym.make("CarRacing-v2", render_mode="human")
    model = net.load_state_dict(torch.load(data_path + "trained_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return env, model, device

def plot_results():
    pass

if __name__=="__main__":
    net = cnn.Net(batch_size)
    manage_neural_network(net)
    plot_results()

    with torch.no_grad(): # Disable gradient calculation for inference
        env, model, device = setup_game()
        agent = imitate.Imitative_Agent(net, env, model, device, track)
        agent.play_game() # And let the agent have fun =)
