from random import choice
import numpy as np

## States determined by
##  -- [x, y, Dx, Dy, Roll, C1x, C1y, C2x, C2y, HP] --
# x, y -> Player position
# Dx, Dy -> Demon position
# Roll -> Can player still roll? (1 or 0)
# C1x, C1y -> Hollow1 position
# C2x, C2y -> Hollow2 position
# HP -> Player HP (1 or 2)

## Weight vector adds:
# Demon distance X -> abs(x - Dx)     # DDX
# Demon distance Y -> abs(y - Dy)     # DDY
# Hollow1 distance X -> abs(x - C1x)  # HDX1
# Hollow1 distance Y -> abs(y - C1y)  # HDY1
# Hollow2 distance X -> abs(x - C2x)  # HDX2
# Hollow2 distance Y -> abs(y - C2y)  # HDY2
# Goal distance X -> abs(x - GOAL[0]) # GDX
# Goal distance Y -> abs(Y - GOAL[1]) # GDY

## Hollow contact deals one damage
## Demon contact deals two
## <= 0 HP means death, ends episode and grants DEATHREWARD
## Reaching goal alive ends episode and grants GOALREWARD
SIZE = 8
START = (1, 6)
DEMONSTART = (6, 3)
ROLLS = 1
HOLLOW1START = (5, 1)
HOLLOW2START = (6, 4)
HPSTART = 2

FEATURES = 8
INITIALSTATE = (START[0], START[1],
                DEMONSTART[0], DEMONSTART[1],
                ROLLS,
                HOLLOW1START[0], HOLLOW1START[1],
                HOLLOW2START[0], HOLLOW2START[1],
                HPSTART)

GOAL = (6, 2)

STEP_REWARD = 0.0
GOAL_REWARD = 1.0
DEFEAT_REWARD = -1.0
RACTION_SET = ["↑", "↓", "→", "←", "↑↑", "↓↓", "→→", "←←"] ## Con roll
ACTION_SET = ["↑", "↓", "→", "←"] ## Sin roll

def feature_vectorII(state):
    '''returns x(s) feature vector of state''' 
    ret = np.array([])
    ret = np.append(ret, (state[2] - state[0]))
    ret = np.append(ret, (state[3] - state[1]))
    ret = np.append(ret, (state[5] - state[0]))
    ret = np.append(ret, (state[6] - state[1]))
    ret = np.append(ret, (state[7] - state[0]))
    ret = np.append(ret, (state[8] - state[1]))
    ret = np.append(ret, (GOAL[0] - state[0]))
    ret = np.append(ret, (GOAL[1] - state[1]))
    return ret

def action_feature_vectorII(state, action):
    '''returns x(s, a) feature vector of state-action pair''' 
    final = np.array([])
    one_hot_action = [1 if act == action else 0 for act in RACTION_SET]
    for choice in one_hot_action:
        final = np.append(final, feature_vectorII(state)*choice)
    return final

def is_terminal(state):
    if state[9] <= 0:
        return True
    if (state[0], state[1]) == GOAL:
        return True
    else:
        return False

def feature_vector(state):
    '''returns x(s) feature vector of state''' 
    ret = np.array([x for x in state])
    ret = np.append(ret, abs(ret[0] - ret[2]))
    ret = np.append(ret, abs(ret[1] - ret[3]))
    ret = np.append(ret, abs(ret[0] - ret[5]))
    ret = np.append(ret, abs(ret[1] - ret[6]))
    ret = np.append(ret, abs(ret[0] - ret[7]))
    ret = np.append(ret, abs(ret[1] - ret[8]))
    ret = np.append(ret, abs(ret[0] - GOAL[0]))
    ret = np.append(ret, abs(ret[1] - GOAL[1]))
    return ret

def action_feature_vector(state, action):
    '''returns x(s, a) feature vector of state-action pair''' 
    final = np.array([])
    one_hot_action = [1 if act == action else 0 for act in RACTION_SET]
    for choice in one_hot_action:
        final = np.append(final, feature_vector(state)*choice)
    return final

def get_actions(state):
    if is_terminal(state):
        return [] ## Si estoy en estado terminal, no hay acciones disponibles
    if state[4]:  ## Si aún tengo el roll
        return RACTION_SET
    else:
        return ACTION_SET
    
def random_policy(actions, values=None):
    '''Elige una acción de las disponibles según la política'''
    return choice(actions)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - (np.tanh(x))**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def state_value(state, weights, clip=True):
    if clip:
        return sigmoid(np.dot(feature_vectorII(state), weights))
    else:  
        return np.dot(feature_vectorII(state), weights)

## ------------- Q values --------------------
def action_value(state, action, weights, clip=True):
    ## q̂(S, A, w) -> scalar
    if clip:
        return tanh(np.dot(action_feature_vectorII(state, action), weights))
    else:
        return np.dot(action_feature_vectorII(state, action), weights)

def daction_value(state, action, weights, clip=True):
    ## ∇ q̂(S, A, w) -> vector
    if clip:
        return dtanh(
            np.dot(action_feature_vectorII(state, action), weights)
            ) * action_feature_vectorII(state, action)
    else:
        return action_feature_vectorII(state, action)
## -------------------------------------------
    
def dstate_value(state, weights, clip=True):
    state = feature_vectorII(state)
    if clip:
        return dsigmoid(np.dot(state, weights)) * state
    else:
        return state


def choose_action(state, policy):
    '''Return the action that the policy would choose in that state'''
    return policy(get_actions(state))

## Maxi es el máximo en X e Y (↓→)
def cardinal_movement(state, action, maxi):
    d = len(action)
    if "↑" in action:
        state = (state[0], max(state[1]-d, 0))
    elif "↓" in action:
        state = (state[0], min(state[1]+d, maxi-1))
    elif "→" in action:
        state = (min(state[0]+d, maxi-1), state[1])
    elif "←" in action:
        state = (max(state[0]-d, 0), state[1])
    else:
        state = state[0], state[1]
    return state


def take_action(state, action):
    '''Returns next state and reward of taking action'''
    player_pos = state[0], state[1]
    demon_pos = state[2], state[3]
    hollow1_pos = state[5], state[6]
    hollow2_pos = state[7], state[8]
    hp = state[9]
    new_state = np.copy(state)
    player_coords = cardinal_movement(player_pos, action, SIZE)
    demon_cords = cardinal_movement(demon_pos, 
                    np.random.choice(["↑", "↓", "→", "←", "_"],
                                     p=[0.1, 0.1, 0.1, 0.1, 0.6]),
                                    SIZE)
    hol1_cords = cardinal_movement(hollow1_pos, 
                    np.random.choice(["↑", "↓", "→", "←", "_"],
                                     p=[0.2, 0.2, 0.2, 0.2, 0.2]),
                                    SIZE)
    hol2_cords = cardinal_movement(hollow2_pos, 
                    np.random.choice(["↑", "↓", "→", "←", "_"],
                                     p=[0.2, 0.2, 0.2, 0.2, 0.2]),
                                    SIZE)
    if (player_coords == demon_cords):
        hp -= 2
    elif (player_coords == hol1_cords) or \
         (player_coords == hol2_cords):
        hp -= 1
    if hp <= 0:
        reward = DEFEAT_REWARD
    elif player_coords == GOAL:
        reward = GOAL_REWARD
    else:
        reward = 0.0
    new_state[0] = player_coords[0]
    new_state[1] = player_coords[1]
    new_state[2] = demon_cords[0]
    new_state[3] = demon_cords[1]
    new_state[4] = 0 if (len(action) > 1) else state[4]
    new_state[5] = hol1_cords[0]
    new_state[6] = hol1_cords[1]
    new_state[7] = hol2_cords[0]
    new_state[8] = hol2_cords[1]
    new_state[9] = hp
    return reward, new_state

if __name__ == "__main__":
    print(feature_vectorII(INITIALSTATE))
    print(len(action_feature_vectorII(INITIALSTATE, '↑')))

