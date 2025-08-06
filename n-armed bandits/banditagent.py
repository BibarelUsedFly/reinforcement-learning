import numpy as np
from auxiliary import epsilonize

class BanditAgent:
    def __init__(self, env, epsilon=0.1, alpha=None, startvalue=0.0):
        self.env = env
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate, if none, use average
        self.Q = {action: startvalue for action in env.available_actions()}
        self.N = {action: 0 for action in env.available_actions()}

    def choose_action(self):
        values = np.array(list(self.Q.values()))
        probs = epsilonize(values)
        return np.random.choice(probs)
    
    def learn(self, action, reward):
        self.N[action] += 1
        if self.alpha:
            self.Q[action] += self.alpha * (reward - self.Q[action])
        else: # Use average instead of a fixed learning rate
            self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

if __name__ == '__main__':
    a = BanditAgent()