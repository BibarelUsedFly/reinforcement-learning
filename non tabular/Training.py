import numpy as np
np.set_printoptions(precision=2)
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
from DemonModel import INITIALSTATE, FEATURES, random_policy, \
    state_value, dstate_value, is_terminal, choose_action, take_action

EPISODES = 50
ALPHA = 0.10
GAMMA = 0.99

policy = random_policy
weights = np.zeros(FEATURES)
for _ in range(EPISODES):
    S = INITIALSTATE
    while not is_terminal(S):
        A = choose_action(S, policy)
        R, S2 = take_action(S, A)
        weights += ALPHA * \
            (R + GAMMA*state_value(S2, weights) - state_value(S, weights)) * \
            dstate_value(S, weights)
        S = S2
    if (_+1) % 100 == 0:
        print("{}/{}".format(_+1, EPISODES))

stamp = str(datetime.now()); stamp = stamp[:stamp.rindex(':')]
stamp = stamp.replace(":", "").replace(" ", "-")
np.save("Weights/weights_data{}".format(stamp), weights)

