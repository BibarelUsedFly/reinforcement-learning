import numpy as np
import os
from datetime import datetime
from auxiliary import epsilonize
from DemonModel import RACTION_SET, FEATURES, INITIALSTATE, \
                       action_value, daction_value, \
                       get_actions, is_terminal, take_action
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from MakeHistory import print_weights

EPISODES = 100
ALPHA = 0.01
EPSILON = 0.05
GAMMA = 0.99
WEIGHTS = "Weights/weights_data_sarsa2023-06-12-1911.npy"
## (10 + 8) * 8 
## 10 features determinan el estado, 8 derivadas añaden información
## Y se cruzan con las 8 posibles acciones
if not WEIGHTS:
    weights = np.zeros(FEATURES * len(RACTION_SET))
else:
    weights = np.load(WEIGHTS)


def egreedy_policy(state, actions, weights, epsilon=0.1):
    '''Elige una acción de acuerdo a una política e-greedy'''
    action_scores = epsilonize(
        np.array([action_value(state, act, weights) for act in actions]),
        epsilon)
    try:
        return np.random.choice(actions, p=action_scores)
    except ValueError:
        # print(weights)
        # print(actions, [action_value(state, act, weights) for act in actions])
        # print(actions, action_scores)
        return np.random.choice(actions, p=action_scores)
    
def policy_probs(state, actions, action, weights, epsilon=0.1):
    '''Retorna la probabilidad de elegir la acción action'''
    action_scores = epsilonize(
        np.array([action_value(state, act, weights) for act in actions]),
        epsilon)
    action_index = actions.index(action)
    return action_scores[action_index]


def episodic_semigradient_sarsa(initial_state, weights,
                                alpha=0.10, epsilon=0.1, gamma=0.99,
                                n_episodes=10, verbose=False):
    for n in range(n_episodes):
        ## S_0, A_0
        S = initial_state
        A = egreedy_policy(S, get_actions(S), weights, epsilon)
        while not is_terminal(S): ## Mientras dure el episodio
            R, S1 = take_action(S, A)
            if is_terminal(S1):
                weights += alpha*(R - action_value(S, A, weights)) * \
                           daction_value(S, A, weights)
                break ## Next episode
            A1 = egreedy_policy(S1, get_actions(S1), weights, epsilon)
            weights += alpha * \
                (R + gamma*action_value(S1, A1, weights) - 
                 action_value(S, A, weights)) * daction_value(S, A, weights)
            S, A = S1, A1
        if verbose:
            print("Episode {}\nWeights: {}\n".format(n+1, weights))
        else:
            if ((n+1)%100) == 0:
                print("Episode {}".format(n+1))
    return weights


def n_step_semigradient_tree(initial_state, weights,
                                n=1, alpha=0.10, epsilon=0.1, gamma=0.99,
                                n_episodes=10, verbose=False):
    for _ in range(n_episodes):
        ## S_0, A_0
        States = [initial_state] * (n + 1)
        Actions = [egreedy_policy(initial_state, get_actions(initial_state),
                                   weights, epsilon)] * (n + 1)
        Rewards = [0] * (n + 1)
        time = 0        ## Current time
        T = np.inf      ## End time
        tau = time - n  ## Update time
        while tau != (T - 1):
            if time < T:
                Rewards[(time+1)%(n+1)], States[(time+1)%(n+1)] = \
                    take_action(States[(time+1)%(n+1)], Actions[(time+1)%(n+1)])
                if is_terminal(States[(time+1)%(n+1)]):
                    T = time + 1
                else:
                    Actions[(time+1)%(n+1)] = \
                        egreedy_policy(States[(time+1)%(n+1)],
                                       get_actions(States[(time+1)%(n+1)]),
                                       weights, epsilon)
            tau = time + 1 - n
            if tau >= 0:
                G = action_value(States[(tau)%(n+1)], Actions[(tau)%(n+1)], weights) + (
                    sum([
                        (
                            ## delta_k
                            (Rewards[(k+1)%(n+1)] + \
                            gamma*sum([policy_probs(States[(k+1)%(n+1)], get_actions(States[(k+1)%(n+1)]), ac, weights, 0.0)*action_value(States[(k+1)%(n+1)], ac, weights) for ac in get_actions(States[(k+1)%(n+1)])]) - \
                            action_value(States[(k)%(n+1)], Actions[(k)%(n+1)], weights))) \
                                * \
                            (np.prod([policy_probs(States[i%(n+1)], get_actions(States[i%(n+1)]), Actions[i%(n+1)], weights, 0.0) for i in range(tau+1, k+1)])
                        ) \
                        for k in range(tau, min(tau+n, T))
                    ])
                )
                weights += alpha * (
                    G - action_value(States[(tau)%(n+1)],
                                     Actions[(tau)%(n+1)],
                                     weights)) * \
                    daction_value(States[(tau)%(n+1)],
                                  Actions[(tau)%(n+1)],
                                  weights
                    )
            time += 1
        if verbose:
            print("Episode:", _+1, "Reward:", Rewards[T%(n+1)])
            print_weights(get_actions(initial_state), weights)
        else:
            if ((_+1)%1) == 0:
                print("Episode {}".format(_+1))
    return weights


if __name__ == '__main__':
    ## n = 4
    weights = n_step_semigradient_tree(INITIALSTATE, weights, 4,
                                          ALPHA, EPSILON, GAMMA, EPISODES,
                                          True)

    stamp = str(datetime.now()); stamp = stamp[:stamp.rindex(':')]
    stamp = stamp.replace(":", "").replace(" ", "-")
    np.save("Weights/weights_data_sarsa{}".format(stamp), weights)
    print(weights)