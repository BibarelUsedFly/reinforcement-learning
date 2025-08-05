from random import random
import numpy as np
from itertools import product
from collections import OrderedDict
from auxiliary import optimalize, epsilonize, manhattan_distance
from functions import choose_action

from Models import GOAL

## Tabla (Matriz) inicial de valores de las acciones Q(s, a)
## Conjunto (lista) de acciones ordenadas ej ["↓", "↑", "0"]
## Función STEP(S_t, A_t) -> (R_t+1, S_t+1)
## S_0 Estado inicial del agente
## Lista de estados VÁLIDOS y NO TERMINALES
## α <- Learning rate/Step size (Qué tanto vale la experiencia reciente)
## γ <- Discount factor (Pondera el valor de las recompensas futuras)
## ε <- Exploration factor (Qué tan frecuentemente la política explora)

## Off policy n-step sarsa with Q values
def n_sarsa(qstart_values, action_set, step_function, start_state, play_states,
                       n_steps=4, alpha=0.1, gamma=0.99, epsilon=0.0,
                       n_episodes=1000, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    target_policy = np.zeros(action_values.shape) ## π(S)
    soft_policy = np.zeros(action_values.shape)   ## b(S)
    ## Paso de los valores de acciones a las probabilidades de elegir
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], False) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False) ## Random
    ##### Data Spot -------
    if collect_data:
        final_data = []
        average_reward = 0.0
        rho_updates = OrderedDict()
        rho_updates_n = {}
    ##### -----------------
    count = 0
    while count < n_episodes:
        ##### Data Spot -------
        if collect_data:
            episode_reward = 0.0
        ##### -----------------
        S = start_state
        A = choose_action(soft_policy, S, action_set)
        n_history = [[0.0, S, A] for _ in range(n_steps+1)]   ## (R, S, A)
        T = np.inf ## Tiempo de término del episodio
        tau = 0    ## Tiempo cuyo valor estoy actualizando
        t = 0      ## Tiempo actual de la simulación
        while tau != T-1: ## Esto termina cuando el tiempo a actualizar (tau)
                          ## Es el último tiempo en que ejecuto acciones (T-1)
            if t < T: ## Si aún no termino la simulación
                S, R = step_function(S, A) ## S2 == S_{t+1}
                ##### Data Spot -------
                if collect_data:
                    episode_reward += R
                ##### -----------------
                n_history[(t+1) % (n_steps+1)][0] = R
                n_history[(t+1) % (n_steps+1)][1] = S
                if S not in play_states: ## Aquí acaba el episodio
                    A = ""
                    T = t + 1
                else:
                    A = choose_action(soft_policy, S, action_set)
                n_history[(t+1) % (n_steps+1)][2] = A
            tau = t - n_steps + 1 ## Tiempo que voy a actualizar
            if tau >= 0:
                rho = 1 ## ρ <- Sampling ratio
                for i in range(tau+1, min(t+1, T)):
                    _, state, action = n_history[i % (n_steps+1)]
                    action = action_set.index(action) ## N° en lugar de nombre
                    rho *= target_policy[state][action] /\
                           soft_policy[state][action]
                G = 0 ## Ganancia observada en n-pasos
                for j in range(tau+1, min(t+1, T)+1):
                    reward, _, _ = n_history[j % (n_steps+1)]
                    G += (gamma ** (j-tau-1)) * reward
                if (t+1) < T: ## Si aún quedan acciones por tomar
                    action = action_set.index(A) ## sumo Q(St+1, At+1)
                    G += (gamma**n_steps) * action_values[S][action]
                _, Stau, Atau = n_history[tau % (n_steps+1)]
                Atau = action_set.index(Atau)
                gain_difference = (G - action_values[Stau][Atau])
                ##### Data Spot -------
                if collect_data: ## Q change
                    isamp = round(rho, 2)
                    rho_updates.setdefault(isamp, 0.0)
                    rho_updates_n.setdefault(isamp, 0)
                    rho_updates_n[isamp] += 1
                    rho_updates[isamp] += (1/rho_updates_n[isamp]) * \
                        (abs(rho) - rho_updates[isamp])
                ##### -----------------
                action_values[Stau][Atau] += alpha * rho * gain_difference
            target_policy[S] = optimalize(action_values[S], True)
            soft_policy[S] = epsilonize(action_values[S], epsilon, False)
            t += 1 ## Subo el tiempo
        count += 1
        ##### Data Spot -------
        if collect_data:
            average_reward += (1/count) * (episode_reward - average_reward)
            final_data.append([episode_reward, average_reward])
        ##### -----------------
        if verbose:
            if count%100 == 0:
                print("Simulation {}/{}".format(count, n_episodes))
    if not collect_data:
        return target_policy
    else:
        return np.array(final_data), rho_updates

## Off policy n-step sarsa with Q values and control variates
def n_sarsav(qstart_values, action_set, step_function, start_state, play_states,
                       n_steps=4, alpha=0.1, gamma=0.99, epsilon=0.0,
                       n_episodes=1000, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    target_policy = np.zeros(action_values.shape) ## π(S)
    soft_policy = np.zeros(action_values.shape)   ## b(S)
    ## Paso de los valores de acciones a las probabilidades de elegir
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], False) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False) ## Random
    ##### Data Spot -------
    if collect_data:
        final_data = []
        average_reward = 0.0
        rho_updates = OrderedDict()
        rho_updates_n = {}
    ##### -----------------
    count = 0
    while count < n_episodes:
        ##### Data Spot -------
        if collect_data:
            episode_reward = 0.0
        ##### -----------------
        S = start_state
        A = choose_action(soft_policy, S, action_set)
        n_history = [[0.0, S, A] for _ in range(n_steps+1)]   ## (R, S, A)
        T = np.inf ## Tiempo de término del episodio
        tau = 0    ## Tiempo cuyo valor estoy actualizando
        t = 0      ## Tiempo actual de la simulación
        while tau != T-1: ## Esto termina cuando el tiempo a actualizar (tau)
                          ## Es el último tiempo en que ejecuto acciones (T-1)
            if t < T: ## Si aún no termino la simulación
                S, R = step_function(S, A) ## S2 == S_{t+1}
                ##### Data Spot -------
                if collect_data:
                    episode_reward += R
                ##### -----------------
                n_history[(t+1) % (n_steps+1)][0] = R
                n_history[(t+1) % (n_steps+1)][1] = S
                if S not in play_states: ## Aquí acaba el episodio
                    A = ""
                    T = t + 1
                else:
                    A = choose_action(soft_policy, S, action_set)
                n_history[(t+1) % (n_steps+1)][2] = A
            tau = t - n_steps + 1 ## Tiempo que voy a actualizar
            if tau >= 0:
                rho = 1 ## ρ <- Sampling ratio
                for i in range(tau+1, min(t+1, T)):
                    _, state, action = n_history[i % (n_steps+1)]
                    action = action_set.index(action) ## N° en lugar de nombre
                    rho *= target_policy[state][action] /\
                           soft_policy[state][action]
                G = 0 ## Ganancia observada en n-pasos
                for j in range(tau+1, min(t+1, T)+1):
                    reward, _, _ = n_history[j % (n_steps+1)]
                    G += (gamma ** (j-tau-1)) * reward
                if (t+1) < T: ## Si aún quedan acciones por tomar
                    action = action_set.index(A) ## sumo Q(St+1, At+1)
                    G += (gamma**n_steps) * action_values[S][action]
                _, Stau, Atau = n_history[tau % (n_steps+1)]
                Atau = action_set.index(Atau)
                gain_difference = (G - action_values[Stau][Atau])
                ##### Data Spot -------
                if collect_data: ## Q change
                    isamp = round(rho, 2)
                    rho_updates.setdefault(isamp, 0.0)
                    rho_updates_n.setdefault(isamp, 0)
                    rho_updates_n[isamp] += 1
                    rho_updates[isamp] += (1/rho_updates_n[isamp]) * \
                        (abs(rho) - rho_updates[isamp])
                ##### -----------------
                action_values[Stau][Atau] += alpha * rho * gain_difference
            target_policy[S] = optimalize(action_values[S], True)
            soft_policy[S] = epsilonize(action_values[S], epsilon, False)
            t += 1 ## Subo el tiempo
        count += 1
        ##### Data Spot -------
        if collect_data:
            average_reward += (1/count) * (episode_reward - average_reward)
            final_data.append([episode_reward, average_reward])
        ##### -----------------
        if verbose:
            if count%100 == 0:
                print("Simulation {}/{}".format(count, n_episodes))
    if not collect_data:
        return target_policy
    else:
        return np.array(final_data), rho_updates

def nQsigma(qstart_values, action_set, step_function, start_state, play_states,
                       n_steps=4, alpha=0.1, gamma=0.99, epsilon=0.0, sigma=0.0,
                       n_episodes=1000, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    target_policy = np.zeros(action_values.shape) ## π(S)
    soft_policy = np.zeros(action_values.shape)   ## b(S)
    ## Paso de los valores de acciones a las probabilidades de elegir
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], False) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False) ## Random
    ##### Data Spot -------
    if collect_data:
        final_data = []
        average_reward = 0.0
        rho_updates = OrderedDict()
        rho_updates_n = {}
    ##### -----------------
    sigmas = {}
    rhos = {}
    count = 0
    while count < n_episodes:
        ##### Data Spot -------
        if collect_data:
            episode_reward = 0.0
        ##### -----------------
        S = start_state
        A = choose_action(soft_policy, S, action_set)
        n_history = [[0.0, S, A] for _ in range(n_steps+1)]   ## (R, S, A)
        T = np.inf ## Tiempo de término del episodio
        tau = 0    ## Tiempo cuyo valor estoy actualizando
        t = 0      ## Tiempo actual de la simulación
        while tau != T-1: ## Esto termina cuando el tiempo a actualizar (tau)
                          ## Es el último tiempo en que ejecuto acciones (T-1)
            if t < T: ## Si aún no termino la simulación
                S, R = step_function(S, A) ## S2 == S_{t+1}
                ##### Data Spot -------
                if collect_data:
                    episode_reward += R
                ##### -----------------
                n_history[(t+1) % (n_steps+1)][0] = R
                n_history[(t+1) % (n_steps+1)][1] = S
                if S not in play_states: ## Aquí acaba el episodio
                    A = ""
                    T = t + 1
                else:
                    A = choose_action(soft_policy, S, action_set)

                    sigmas[t+1] = sigma ## For later use SIGMA
                    action = action_set.index(A)
                    rhos[t+1] = target_policy[S][action] / soft_policy[S][action]

                n_history[(t+1) % (n_steps+1)][2] = A

            tau = t - n_steps + 1 ## Tiempo que voy a actualizar
            if tau >= 0:
                G = 0 ## Ganancia observada en n-pasos
                for k in range(min(t+1, T), tau, -1): ## Loop down
                    if k == T:
                        G = n_history[T % (n_steps+1)][0] ## R_T
                    else:
                        Rk, Sk, Ak = n_history[k % (n_steps+1)]
                        Ak = action_set.index(Ak)
                        V = sum(target_policy[Sk] * action_values[Sk])
                        G = Rk + gamma * (sigmas[k]*rhos[k] + 
                            (1-sigmas[k])*target_policy[Sk][Ak]) * \
                            (G - action_values[Sk][Ak]) + gamma * V
                        
                ##### Data Spot -------
                # if collect_data: ## Q change
                #     isamp = round(rho, 2)
                #     rho_updates.setdefault(isamp, 0.0)
                #     rho_updates_n.setdefault(isamp, 0)
                #     rho_updates_n[isamp] += 1
                #     rho_updates[isamp] += (1/rho_updates_n[isamp]) * \
                #         (abs(rho) - rho_updates[isamp])
                ##### -----------------
          
                _, Stau, Atau = n_history[tau % (n_steps+1)]
                Atau = action_set.index(Atau)
                action_values[Stau][Atau] += alpha * (G - action_values[Stau][Atau])
            target_policy[S] = optimalize(action_values[S], True)
            soft_policy[S] = epsilonize(action_values[S], epsilon, False)
            t += 1 ## Subo el tiemp
        count += 1
        ##### Data Spot -------
        if collect_data:
            average_reward += (1/count) * (episode_reward - average_reward)
            final_data.append([episode_reward, average_reward])
        ##### -----------------
        if verbose:
            if count%100 == 0:
                print("Simulation {}/{}".format(count, n_episodes))
    if not collect_data:
        return target_policy
    else:
        return np.array(final_data), rho_updates

def q_learning(qstart_values, action_set, step_function,
                       start_state, play_states,
                       alpha=0.1, gamma=0.99, epsilon=0.0,
                       n_episodes=999, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    target_policy = np.zeros(action_values.shape)
    soft_policy = np.zeros(action_values.shape)
    ## Paso de los valores de acciones a las probabilidades de elegir
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], True) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False) ## Random
    ##### Data Spot -------
    if collect_data:
        final_data = []
        average_reward = 0.0
    ##### -----------------
    count = 0
    while count < n_episodes:
        ##### Data Spot -------
        if collect_data:
            episode_reward = 0.0
        ##### -----------------
        S = start_state
        while S in play_states: ## Un episodio
            A = choose_action(soft_policy, S, action_set)
            S2, R = step_function(S, A)
            ##### Data Spot -------
            if collect_data:
                episode_reward += R
            ##### -----------------
            A = action_set.index(A)
            action_values[S][A] += alpha * \
                (R + gamma * max(action_values[S2]) - action_values[S][A])
            target_policy[S] = optimalize(action_values[S], True)
            soft_policy[S] = epsilonize(action_values[S], epsilon, False)
            S = S2
        count += 1
        ##### Data Spot -------
        if collect_data:
            average_reward += (1/count) * (episode_reward - average_reward)
            final_data.append([episode_reward, average_reward])
        ##### -----------------
        if verbose:
            if count%100 == 0:
                print("Simulation {}/{}".format(count, n_episodes))
    if not collect_data:
        return target_policy
    else:
        return np.array(final_data)

## Off policy, weighted importance sampling estimator
def montecarlo_control(qstart_values, action_set, simulation_function,
                       gamma=0.99, epsilon=0.0, n_episodes=999, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    weight_sum = np.zeros(qstart_values.shape) ## C(s, a)
    target_policy = np.zeros(action_values.shape)
    soft_policy = np.zeros(action_values.shape)
    ## Recorrer todas las posibilidades de tupla
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], True) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False)  ## Random
    count = 0
    ##### Data Spot -------
    if collect_data:
        final_data = []
        average_reward = 0.0
        breaks = 0
        goons = 0
        states_mh = OrderedDict()
    ##### -----------------
    while count < n_episodes:
        history = simulation_function(soft_policy)
        ##### Data Spot -------
        if collect_data:
            episode_reward = history[-1][0]
            # episode_reward = sum(list(zip(*history))[0])
            # print("EPISODE {} - REWARD {}".format(count+1, episode_reward))
            average_reward += (1/(count+1)) * (episode_reward - average_reward)
            final_data.append([episode_reward, average_reward])
        ##### -----------------
        G = 0
        W = 1 ## Rho
        test = 1
        for t in range(len(history)-2, -1, -1): ## Parto de t = T-1
            R = history[t+1][0]                 ## R_{t+1}
            # print("Reward:", R, test)
            S = history[t][1]                   ## S_{t}
            A = action_set.index(history[t][2]) ## A_{t}
            G = gamma*G + R
            weight_sum[S][A] += W
            action_values[S][A] += (W/weight_sum[S][A]) * \
                                   (G - action_values[S][A])
            ##### Data Spot -------
            if collect_data: ## Q change
                dist = manhattan_distance(S, GOAL)
                states_mh.setdefault(dist, 0)
                states_mh[dist] += 1
            ##### -----------------
            target_policy[S] = optimalize(action_values[S], True)
            if choose_action(target_policy, S, action_set) != history[t][2]:
                breaks += 1
                # if test == 1:
                    # print("Break after {} actions".format(test))
                    # print("Simulation chose {}".format(history[t][2]))
                    # print("PI  would choose {}".format(
                    #     choose_action(target_policy, S, action_set)))
                soft_policy[S] = epsilonize(action_values[i], epsilon, False)
                break ## if pi(A|S) = 0
            goons += 1; test += 1
            W = W * (1/soft_policy[S][A]) ## Rho *= pi(A|S) / b(A|S)
            soft_policy[S] = epsilonize(action_values[i], epsilon, False)
        count += 1
        if verbose:
            if count%100 == 0:
                print("Simulation {}/{}".format(count, n_episodes))
    if not collect_data:
        return target_policy
    else:
        print("BREAKS:", breaks)
        print("Go Ons:", goons)
        return np.array(final_data), states_mh
    
    
## Puro random para probar
def non_control(qstart_values, action_set, simulation_function,
                       n_episodes=999, verbose=False,
                       collect_data=False, n_data_points=0):
    action_values = qstart_values ## Q(s, a)
    weight_sum = np.zeros(qstart_values.shape) ## C(s, a)
    target_policy = np.zeros(action_values.shape)
    soft_policy = np.zeros(action_values.shape)
    ## Recorrer todas las posibilidades de tupla
    for i in product(*(range(e) for e in target_policy.shape[:-1])):
        target_policy[i] = optimalize(action_values[i], True) ## Deterministic
        soft_policy[i] = epsilonize(action_values[i], epsilon, False)  ## Random
    count = 0