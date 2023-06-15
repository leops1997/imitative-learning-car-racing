import cnn
import imitate
import matplotlib.pyplot as plt
import sys
import torch
import gym

if len(sys.argv) >= 5:
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    track = int(sys.argv[3])
    should_train = bool(sys.argv[4])
else:
    epochs = 100
    batch_size = 1
    track = 42
    should_train = True

data_path = "data/"

def manage_neural_network(net):
    net.train() # Set the net to training mode
    cost, accuracy = net.train_model(epochs, "data/")
    return cost, accuracy

def setup_game():
    env = gym.make("CarRacing-v2", render_mode="human")
    net.load_state_dict(torch.load("data/trained_model.pth"))
    net.eval()
    return env

def plot_cost(cost):
    plt.plot(range(epochs), cost)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost Function Value over Epochs')
    plt.grid()
    plt.savefig("plot_cost_function_over_epochs.png")
    plt.show()
    
def plot_accuracy(accuracy):
    plt.plot(range(epochs), accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Value over Epochs')
    plt.grid()
    plt.savefig("plot_accuracy_over_epochs.png")
    plt.show()

def plot_results(cost, accuracy):
    plot_cost(cost)
    plt.clf()
    plot_accuracy(accuracy)

if __name__=="__main__":
    net = cnn.Net(batch_size)
    if should_train:
        cost, accuracy = manage_neural_network(net)
        plot_results(cost, accuracy)
    env = setup_game()
    agent = imitate.Imitative_Agent(net, env, track)
    agent.play_game() # And let the agent have fun =)
