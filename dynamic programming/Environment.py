from parameters import SIZE

def bind(state): ## Mínimo 0 y máximo SIZE
    return (min(max(state[0], 0), SIZE-1), min(max(state[1], 0), SIZE-1))

## pi(a | s)
def policy_pi(policy, state, action_number):
    ## A la matriz en python se accede con [y][x]
    return policy[state[1], state[0], action_number]

## V(s)
def value(state, state_values):
    return state_values[state[1], state[0]]

# print(policy_pi(policy, (3,4), 2))
def step(state, action): ## State son coordenadas
    if action == "↑":
        state = bind((state[0], state[1]-1))
    elif action == "↓":
        state = bind((state[0], state[1]+1))
    elif action == "→":
        state = bind((state[0]+1, state[1]))
    elif action == "←":
        state = bind((state[0]-1, state[1]))
    reward = -1
    return state, reward