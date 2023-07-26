from parameters import SIZE ,PIZZA, STEP_REWARD, PIZZA_REWARD

def bind(state): ## Mínimo 0 y máximo SIZE
    return (min(max(state[0], 0), SIZE-1), min(max(state[1], 0), SIZE-1))

## pi(a | s)
def policy_pi(policy, state, action_number):
    ## A la matriz en python se accede con [y][x]
    return policy[state[1], state[0], state[2], action_number]

## V(s)
def value(state, state_values):
    return state_values[state[1], state[0], state[2]]

def step(state, action, walls): ## State son coordenadas
    init = state
    ispizza = state[2]
    if action == "↑":
        state = bind((state[0], state[1]-1))
    elif action == "↓":
        state = bind((state[0], state[1]+1))
    elif action == "→":
        state = bind((state[0]+1, state[1]))
    elif action == "←":
        state = bind((state[0]-1, state[1]))
    if state in walls:
        state = init
    reward = STEP_REWARD
    if ispizza and (state == PIZZA):
        reward += PIZZA_REWARD
        ispizza = 0
    return (state[0], state[1], ispizza), reward