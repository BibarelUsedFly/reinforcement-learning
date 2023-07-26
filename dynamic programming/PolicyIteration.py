import pygame
import sys
from entity import Entity
import numpy as np
from copy import deepcopy
from time import time
from parameters import *
from Environment import *


## Cosas gráficas
screen = pygame.display.set_mode((XMAX,YMAX)) # Dimensiones pantalla
pygame.display.set_caption('Policy Iteration') # Título

img_rob = pygame.image.load('../Assets/mRoball.png') # Cargar al cute_robot
img_wall = pygame.image.load('../Assets/WallN.png')  # Cargar la cute_wall

## Inicio con política equiprobable
start_policy = np.full((SIZE, SIZE, len(ACTION_SET)), 1/len(ACTION_SET))
policy_history = []
policy_history.append(start_policy)

states = [(x, y) for x in range(SIZE) for y in range(SIZE)] ## S+
play_states = [(x, y) for x in range(SIZE) for y in range(SIZE)] ## S
play_states.remove(GOAL)

## ----- Policy Iteration ----- ##
## p(r, s' | s, a) = 1 para r = -1, s' = step(s, a)
## y 0 en otro caso
def policy_evaluation(policy, state_values):
    history_of_state_values = []
    delta = np.inf
    while delta >= THETA:
        delta = 0
        state_values_f = deepcopy(state_values)
        history_of_state_values.append(deepcopy(state_values))
        for state in play_states: ## for s in S (no S+)
            v = value(state, state_values)
            policy_average = 0
            ## Este for es para la sumatoria ↓
            for action_number in range(len(ACTION_SET)): # for a in A
                probability = policy_pi(policy, state, action_number) # pi(a|s)
                next_state, reward = step(
                    state, ACTION_SET[action_number]) # s' y r
                policy_average += probability * \
                    (reward + GAMMA * value(next_state, state_values))
            state_values_f[state[1], state[0]] = policy_average ## Y↓,X→
            delta = max(delta, abs(v - value(state, state_values_f)))
        state_values = state_values_f
    history_of_state_values.append(deepcopy(state_values))
    return state_values, history_of_state_values


def policy_improvement(policy, state_values):
    new_policy = deepcopy(policy)
    for state in play_states:
        action_values = []
        for action in ACTION_SET:
            new_state, reward = step(state, action)
            action_value = reward + GAMMA * value(new_state, state_values)
            action_values.append(action_value)
        new_probs = np.zeros(len(ACTION_SET))
        new_probs[np.argmax(action_values)] = 1.0
        new_policy[state[1], state[0]] = new_probs
    return new_policy

## ------------------------------------------------------------------------
start_time = time()
history_of_histories = []
state_values, history = policy_evaluation(
    policy_history[-1], np.zeros((SIZE, SIZE)))
history_of_histories.append(history)
while (len(policy_history) < 2) or \
      (policy_history[-1] != policy_history[-2]).all():

    new_policy = policy_improvement(start_policy, state_values)
    policy_history.append(new_policy)

    state_values, history = policy_evaluation(
        policy_history[-1], deepcopy(state_values))
    history_of_histories.append(history)
end_time = time() - start_time
print("TIME: {}".format(end_time))

## PyGame ------------------------------------------------------------------
def drawGrid():
    rect = pygame.Rect(GOAL[0] * RESOLUTION, GOAL[1] * RESOLUTION,
                       RESOLUTION, RESOLUTION)
    pygame.draw.rect(screen, GREENBLU, rect)
    for x in range(0, XMAX, RESOLUTION):
        for y in range(0, YMAX, RESOLUTION):
            rect = pygame.Rect(x, y, RESOLUTION, RESOLUTION)
            pygame.draw.rect(screen, GRAY, rect, 1)


def drawArrow(color, size, position, angle):
    baseline = np.array([(0, -30), (0, 30), (200, 30),
                (200, 80), (300, 0), (200, -80), (200, -30)])
    rotation = np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])
    new_baseline = []
    for spot in baseline:
        new_baseline.append(rotation.dot(spot))
    new_baseline = np.array(new_baseline)
    traslation = np.full((7, 2), position)
    pygame.draw.polygon(screen, color, new_baseline*size + traslation)

def drawPolicy(policy):
    for x in range(SIZE):
        for y in range(SIZE):
            if (x, y) != GOAL:
                n = 0
                for action in policy[y][x]:
                    label = ACTION_SET[n]
                    if label == "↑":
                        angle = np.pi/2
                    elif label == "↓":
                        angle = -np.pi/2
                    elif label == "→":
                        angle = 0
                    elif label == "←":
                        angle = np.pi
                    arrow = drawArrow(BLACK, ARROWSIZE*action, (
                                (x+DRAW_DISPLACEMENT_FACTOR_XA) * RESOLUTION,
                                (y+DRAW_DISPLACEMENT_FACTOR_YA) * RESOLUTION
                                ), angle)
                    n += 1
    

pygame.font.init() # Iniciar texto en pygame
font = pygame.font.Font(None, 20)
agent = Entity(img_rob, 1, 1, screen, RESOLUTION)

t = 0
policy_number = 0
draw_values = True
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # ↑↓→←
            if event.key == pygame.K_LEFT:
                t = max(t-1, 0)
            if event.key == pygame.K_RIGHT:
                t = min(t+1, len(history_of_histories[policy_number])-1)
            if event.key == pygame.K_UP:
                policy_number = max(policy_number-1, 0) 
                t = 0
            if event.key == pygame.K_DOWN:
                policy_number = min(policy_number+1, len(policy_history)-1)
                t = 0
            # WASD
            if event.key == pygame.K_a:
                t = 0
            if event.key == pygame.K_d:
                t = len(history_of_histories[policy_number])-1
            if event.key == pygame.K_w:
                policy_number = 0
                t = 0
            if event.key == pygame.K_s:
                policy_number = len(policy_history)-1
                t = 0

            if event.key == pygame.K_p:
                draw_values = not draw_values

    screen.fill(LIGHTBLU)
    drawGrid()

    if draw_values:
        time_text = \
            font.render("TIME: {} — POLICY: {}".format(t, policy_number),
            True, pygame.Color("black"))
        screen.blit(time_text, (2,1))
        for x in range(SIZE):
            for y in range(SIZE):
                text = font.render(
                    str(round(history_of_histories[policy_number][t][y][x], 2)),
                              True, pygame.Color("black"))
                screen.blit(text, ((x+DRAW_DISPLACEMENT_FACTOR_X) * RESOLUTION,
                                (y+DRAW_DISPLACEMENT_FACTOR_Y) * RESOLUTION))
    else:
        time_text = \
            font.render("POLICY: {}".format(policy_number),
            True, pygame.Color("black"))
        screen.blit(time_text, (2,1))
        drawPolicy(policy_history[policy_number])

    pygame.display.update()
    