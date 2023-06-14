import torch

'''
The purpose of this class is to manage the simulation after training the model
'''
class Imitative_Agent():
    def __init__(self, net, env, track):
        self.net = net
        self.env = env
        self.track = track

    # This function sets the game up
    def play_game(self):
        self.env.reset(seed=self.track) # First, we set the track 
        action = [0, 0, 0] # Then, we initialize the action array as [0, 0, 0] (car is stalled)
        self.main_game_loop(action) # Then, we initialize the main loop of the game
        self.env.close() # After we're done with the simulation, we close the environment
    
    def main_game_loop(self, action):
        with torch.no_grad(): # Disable gradient calculation for inference
            terminated = False 
            while not terminated:
                self.env.render() # Here, we render the game

                observation, reward, terminated, truncated, info = self.env.step(action)
                state = observation # Getting the state of the car (as an image)

                # Preprocessing the data 
                data = torch.from_numpy(state.copy()).to(torch.float32)
                data = data.unsqueeze(0)
                data = data.transpose(1,3)

                # And, finallty, get the resultant action from the network
                action = torch.logit(self.net(data)).squeeze().tolist()
                print(action)
