import numpy as np
import matplotlib.pyplot as plt
import gym

def load_data():

    path = "data/"
    actions = np.load(path+"actions.npy", allow_pickle=True)
    states = np.load(path+"states.npy",  allow_pickle=True)

    fig, ax = plt.subplots(3,1)
    for i in range(3):
        ax[i].plot(actions[:,i])
    plt.show()

    plt.imshow(states[100,:,:,:])
    plt.show()

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
    # load_data()
    agent()