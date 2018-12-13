import numpy as np
from task import Task

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

"""
Monte Carlo with Linear Function Approximation

Nearly the same as PolicySearch, but uses error gradient
to update the weights
"""
class MonteCarlo_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        
        # add for monte carlo
        self.episode = [] # array of (state,action,reward) triplets
        self.last_action = None # for saving into an episode step
        self.last_state = None # for saving into an episode step
        
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        
        # added for Monte Carlo
        self.episode = []
        self.last_state = None
        self.last_action = None
        
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        
        self.episode.append((self.last_state, self.last_action, reward))

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        out = sigmoid(np.dot(state, self.w))
        action = np.interp(out, [0,1], [self.action_low, self.action_high])

        # save state, action
        self.last_state = state
        self.last_action = action

        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        