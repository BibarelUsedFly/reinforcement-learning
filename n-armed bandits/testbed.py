import numpy as np

class Bandit10:
    
    def __init__(self, mean, stdev, arms):
        self.arms = np.random.normal(mean, stdev, arms)
        self.n_arms = arms
        self.arm_stdev = stdev

    def __repr__(self):
        return str(self.arms)

    def take_action(self, n):
        return np.random.normal(self.arms[n], self.arm_stdev)

    @property
    def best_action(self):
        return np.argmax(self.arms)

    def pretty_print(self):
        ret = "---Bandit---\n"
        for i in range(len(self.arms)):
            ret += "    Action {}: {}\n".format(i, self.arms[i])
        ret += "Best Action -> {}".format(self.best_action)
        print(ret)

def make_testbed(n, mean, stdev, arms): # Como la testbed de la pg.28 del libro
    return [Bandit10(mean, stdev, arms) for _ in range(n)]

## Aprender de un problema por n ciclos
def egreedy(bandit, n, epsilon):
    ## acción -> [x, y] donde x es el valor estimado de la acción e y es
    ## la cantidad de veces que ha sido seleccionada
    actions = np.array([[0.0, 0] for _ in range(bandit.n_arms)])
    
    ## Cosas para graficar
    action_history = []
    avg_reward = 0
    avg_reward_history = []
    best_action_selected = 0
    best_action_percent = []

    for t in range(1, n+1):
        values = np.take(actions, 0, axis=1)
        if np.random.random() < epsilon:
            action = np.random.choice(range(bandit.n_arms))
        else:    
            action = np.random.choice(np.flatnonzero(values == values.max()))
        reward = bandit.take_action(action)
        actions[action][1] = actions[action][1] + 1
        actions[action][0] = actions[action][0] + \
            (1/actions[action][1]) * (reward - actions[action][0])
        action_history.append(action)
        avg_reward += (1/t) * (reward - avg_reward)
        avg_reward_history.append(avg_reward)
        if action == bandit.best_action:
            best_action_selected += 1
        best_action_percent.append(best_action_selected/t)
    return action_history, avg_reward_history, best_action_percent, actions
