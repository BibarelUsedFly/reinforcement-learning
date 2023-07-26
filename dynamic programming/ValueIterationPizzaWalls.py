import pygame
import sys
from entity import Entity
import numpy as np
from copy import deepcopy
from time import time
from Reinforce import optimalize
from parameters import *
from PizzaEnvironment import *

## Cosas gráficas
screen = pygame.display.set_mode((XMAX,YMAX)) # Dimensiones pantalla
pygame.display.set_caption('Value Iteration - Pizza Version') # Título

img_rob = pygame.image.load('../Assets/mRoball.png') # Cargar al cute_robot
img_wall = pygame.image.load('../Assets/WallN.png') # Cargar la cute_wall
img_piz = pygame.image.load('../Assets/Pizza.png')
img_piz = pygame.transform.scale(img_piz,(.9*RESOLUTION, .9*RESOLUTION))
img_wall = pygame.transform.scale(img_wall,(.99*RESOLUTION, .99*RESOLUTION))



## Inicio con política equiprobable ## MATRIZ
start_policy = np.full((SIZE, SIZE, 2, len(ACTION_SET)), 1/len(ACTION_SET))
policy_history = []
policy_history.append(start_policy)

states = [(x, y, z) for x in range(SIZE) \
          for y in range(SIZE) for z in range(2)] ## S+
play_states = [(x, y, z) for x in range(SIZE) \
               for y in range(SIZE) for z in range(2)] ## S
## NonStates
states.remove((PIZZA[0], PIZZA[1], 1))
play_states.remove((PIZZA[0], PIZZA[1], 1))
for wall in WALLS:
    states.remove((wall[0], wall[1], 1))
    play_states.remove((wall[0], wall[1], 1))
    states.remove((wall[0], wall[1], 0))
    play_states.remove((wall[0], wall[1], 0))
## End states
play_states.remove((GOAL[0], GOAL[1], 1))
play_states.remove((GOAL[0], GOAL[1], 0))


## ----- Policy evaluation ----- ##
## p(r, s' | s, a) = 1 para r = -1, s' = step(s, a)
## y 0 en otro caso
def policy_evaluation2(policy, state_values):
    history_of_state_values = []
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
                state, ACTION_SET[action_number], WALLS) # s' y r
            policy_average += probability * \
                (reward + GAMMA * value(next_state, state_values))
        state_values_f[state[1], state[0], state[2]] = policy_average ## Y↓,X→
        delta = max(delta, abs(v - value(state, state_values_f)))
    state_values = state_values_f
    history_of_state_values.append(deepcopy(state_values))
    return state_values, history_of_state_values, delta


def policy_improvement(policy, state_values):
    new_policy = deepcopy(policy)
    for state in play_states:
        action_values = []
        for action in ACTION_SET:
            new_state, reward = step(state, action, WALLS)
            action_value = reward + GAMMA * value(new_state, state_values)
            action_values.append(action_value)
        if DETERMINISTIC:
            new_probs = np.zeros(len(ACTION_SET))
            new_probs[np.argmax(action_values)] = 1.0
        else:
            new_probs = optimalize(np.array(action_values))
        new_policy[state[1], state[0], state[2]] = new_probs
    return new_policy

print("INICIO")
## ------------------------------------------------------------------------
start_time = time()
## POLICY ITERATION
history_of_histories = []
state_values, history, delta = policy_evaluation2(
    policy_history[-1], np.zeros((SIZE, SIZE, 2)))
history_of_histories.append(history)
while (delta >= THETA):
    new_policy = policy_improvement(start_policy, state_values)
    policy_history.append(new_policy)

    state_values, history, delta = policy_evaluation2(
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

## 0 -> Full 1 -> left, 2 -> Right
def drawArrow(color, size, position, angle, half=False):
    if half == 0:
        baseline = np.array([(0, -30), (0, 30), (200, 30),
                    (200, 80), (300, 0), (200, -80), (200, -30)])
    elif half == 1:
        baseline = np.array([(0, -30), (0, 0), (200, 0),
                    (200, 0), (300, 0), (200, -80), (200, -30)])
    elif half == 2:
        baseline = np.array([(0, 0), (0, 30), (200, 30),
                    (200, 80), (300, 0), (300, 0), (200, 0)])
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
            if (x, y, 0) in play_states:
                n = 0
                for action in policy[y][x][0]:
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
                                ), angle, half=1)
                    n += 1
            if (x, y, 1) in play_states:
                n = 0
                for action in policy[y][x][1]:
                    label = ACTION_SET[n]
                    if label == "↑":
                        angle = np.pi/2
                    elif label == "↓":
                        angle = -np.pi/2
                    elif label == "→":
                        angle = 0
                    elif label == "←":
                        angle = np.pi
                    arrow = drawArrow(REDBLU, ARROWSIZE*action, (
                                (x+DRAW_DISPLACEMENT_FACTOR_XA) * RESOLUTION,
                                (y+DRAW_DISPLACEMENT_FACTOR_YA) * RESOLUTION
                                ), angle, half=2)
                    n += 1
    

pygame.font.init() # Iniciar texto en pygame
font = pygame.font.Font(None, 20)
colors = [pygame.Color("black"), REDBLU]
agent = Entity(img_rob, 1, 1, screen, RESOLUTION)
pizza = Entity(img_piz, PIZZA[0], PIZZA[1], screen, RESOLUTION)
wall_entities = [Entity(img_wall, w[0], w[1], screen, RESOLUTION) for w in WALLS]
## Qué dibujo?
draw_values = True
t = 0
policy_number = 0
n_photo = 0 ## For photo saving
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        ## Efectos de teclas
        elif event.type == pygame.KEYDOWN:
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

            if event.key == pygame.K_f: #Photo
                pygame.image.save(screen,
                                   "Out/PizzaImage{}.png".format(n_photo))
                n_photo += 1
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
    pizza.draw()
    for wallentity in wall_entities:
        wallentity.draw()

    if draw_values:
        time_text = \
            font.render("TIME: {} — POLICY: {}".format(t, policy_number),
            True, pygame.Color("black"))
        screen.blit(time_text, (2,1))
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(2):
                    if (x, y, z) in states:
                        text = font.render(
                            str(round(history_of_histories[policy_number][t][y][x][z], 2)),
                                    True, colors[z])
                        screen.blit(text, ((x+DRAW_DISPLACEMENT_FACTOR_X) * RESOLUTION,
                                        (y + DRAW_DISPLACEMENT_FACTOR_Y + 0.15*z) * RESOLUTION))
    else:
        time_text = \
            font.render("POLICY: {}".format(policy_number),
            True, pygame.Color("black"))
        screen.blit(time_text, (2,1))
        drawPolicy(policy_history[policy_number])

    pygame.display.update()
    