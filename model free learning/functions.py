from random import random
import numpy as np

## Policy -> Matriz (n-dimensional) de estados en donde cada entrada 
##           es un vector de probabilidades con las probabilidades de las
##           acciones (Por eso es importante que las acciones sean numeradas)
## State -> Número o tupla identificador del estado en la matriz de acciones
## Actions -> Vector ordenado con las posibles acciones
def choose_action(policy, state, actions):
    "Retorna la acción elegida (por nombre)"
    probs = policy[state] ## Get actions probs vector for state under policy
    return np.random.choice(actions, p=probs) ## Choose action

## Retorna el resultado de simular un episodio 
## bajo una cierta política de acción
## play states es la lista de estados válidos no terminales
## step es la función que ejecuta la acción de acuerdo al entorno
def simulate_episode(behavioral_policy, starting_state, action_set,
                     play_states, step_function, max_timesteps=999, *params):
    state = starting_state ## inicializo
    ## (R_t, S_t, A_t) <- Formato de tuplas
    time = (0.0, state, choose_action(behavioral_policy, state, action_set))
    ## En t=0, Tengo R_t = 0.0, S_t = starting_state, A_T = policy_act(*params)
    history = [time] ## Creo la historia, que es la lista de tuplas de tiempo
    timesteps = 0 ## Límite por si todo demora demasiado
    while timesteps < max_timesteps:                 ## Take a step
        next_state, reward = step_function(state, time[2], *params)
        if next_state not in play_states: ## Si llegué a un estado terminal
            next_time = (reward, next_state, None)
            history.append(next_time)
            break
        else: ## Paso el tiempo
            state = next_state
            next_time = (reward, next_state,
                            choose_action(behavioral_policy, state, action_set))
            history.append(next_time)
            time = next_time 
        timesteps += 1
    return history