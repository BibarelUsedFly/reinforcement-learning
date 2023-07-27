from random import random, choice
import numpy as np
# from itertools import product
from auxiliary import hardcode
from functions import simulate_episode

SIZE = 8
GOAL = (6, 2)
START = (1, 6)
ENEMYSTART = (5, 2)
ACTION_SET = ["↑", "↓", "→", "←"]
STEP_REWARD = 0.0
GOAL_REWARD = 1.0
VICTORY_REWARD = 1.0
DEFEAT_REWARD = -1.0
SLIPCHANCE = 0.0
WINCHANCE = 0.0

def asylum(state, action, goal, maxi, stepreward=-1, goalreward=1,
         slayreward=1, deathreward=-1, slip_chance=0.0, win_chance=0.0):
    start = state[0], state[1]
    end = cardinal_movement(state, action, maxi, slip_chance)
    reward = stepreward
    if state[4]: # Si aún vive el monstruo
        monster = state[2], state[3]
        ## El monstruo se mueve
        monster_act = np.random.choice(["↑", "↓", "→", "←", "0"],
                                       p=(0.1, 0.1, 0.1, 0.1, 0.6))
        monster = cardinal_movement(monster, monster_act, maxi)
        if (end == monster): ## Batalla!
            if random() < win_chance: ## Victoria!
                reward += slayreward
                state = (end[0], end[1], 0, 0, 0)
                # print(":)")
            else: ## Derrota
                reward += deathreward
                state = (-1, -1, monster[0], monster[1], 1)
                # print(":(")
        else:
            state = (end[0], end[1], monster[0], monster[1], 1)
    else:
        state = (end[0], end[1], 0, 0, 0)
    ## Si llegué a la meta
    if (state[0], state[1]) == goal:
        reward += goalreward
    return state, reward

## Maxi es el máximo en X e Y (↓→)
def cardinal_movement(state, action, maxi, slip_chance=0.0):
    if random() < slip_chance:
        action = choice(["↑", "↓", "→", "←"])
    if action == "↑":
        state = (state[0], max(state[1]-1, 0))
    elif action == "↓":
        state = (state[0], min(state[1]+1, maxi-1))
    elif action == "→":
        state = (min(state[0]+1, maxi-1), state[1])
    elif action == "←":
        state = (max(state[0]-1, 0), state[1])
    else:
        state = state[0], state[1]
    return state

states = [(x1, y1, x2, y2, e) \
            for x1 in range(SIZE) \
            for y1 in range(SIZE) \
            for x2 in range(SIZE) \
            for y2 in range(SIZE) \
            for e in range(2) if ((x2 == 0 and y2 == 0) or e)] ## S+
            ## 4160 estados
play_states = [(x1, y1, x2, y2, e) \
                for x1 in range(SIZE) \
                for y1 in range(SIZE) \
                for x2 in range(SIZE) \
                for y2 in range(SIZE) \
                for e in range(2) if (x1, y1) != GOAL \
                and ((x2 == 0 and y2 == 0) or e)] ## S

BEGINNING = (START[0], START[1], ENEMYSTART[0], ENEMYSTART[1], 1)
step_func = hardcode(asylum, None, None, GOAL, SIZE,
    STEP_REWARD, GOAL_REWARD, VICTORY_REWARD,
    DEFEAT_REWARD, SLIPCHANCE, WINCHANCE,
    n=2)
simu_func = hardcode(simulate_episode, None, BEGINNING, ACTION_SET, play_states,
                     step_func, 99999, n=1)
## (x1, y1) -> Posición del jugador
## (x2, y2) -> Posición del enemigo
## e -> 1 si el enemigo está vivo, 0 si no
Q = np.full((SIZE, SIZE, SIZE, SIZE, 2,
             len(ACTION_SET)), 0.0)



