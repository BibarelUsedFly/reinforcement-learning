import pygame
import sys
from entity import Entity
import numpy as np
from visualgoal import drawGrid
from functions import simulate_episode
from Models import asylum
from Agents import n_sarsa, q_learning
from auxiliary import hardcode
from colors import *
from time import time
from datetime import datetime

XMAX = 720
YMAX = XMAX
SIZE = 8
RESOLUTION = int(XMAX/SIZE)
ARROWSIZE = 0.001625*RESOLUTION
ALPHA = 0.10
GAMMA = 0.99
EPSILON = 0.10
SAVE = True
LOAD = "model free learning/Checkpoint3000.npy"
CONTROLLER = "sarsa"

## Cosas gráficas --------------------------------------------------------------
screen = pygame.display.set_mode((XMAX,YMAX)) # Dimensiones pantalla
pygame.display.set_caption('Episode Simulation') # Título
# pygame.key.set_repeat(1, 50) # Leer input de las teclas cada 50ms

img_rob = pygame.image.load('../Assets/mRoball.png')
img_dev = pygame.image.load('../Assets/AsylumDemon150.png')

## Game Parameters -------------------------------------------------------------
GOAL = (6, 2)
START = (1, 6)
ENEMYSTART = (5, 2)
ACTION_SET = ["↑", "↓", "→", "←"]
STEP_REWARD = 0.0
GOAL_REWARD = 1.0
VICTORY_REWARD = 10.0
DEFEAT_REWARD = -25.0
MAX_TIMESTEPS = 9999 ## Por simulación

## (x1, y1) -> Posición del jugador
## (x2, y2) -> Posición del enemigo
## e -> 1 si el enemigo está vivo, 0 si no
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

behavioral_policy = np.full((SIZE, SIZE, SIZE, SIZE, 2, len(ACTION_SET)),
                            1/len(ACTION_SET)) ## b
# target_policy = np.full((SIZE, SIZE, SIZE, SIZE, 2, len(ACTION_SET)),
#                             1/len(ACTION_SET)) ## pi

## Simulation ------------------------------------------------------------------
## (R_t, S_t, A_t)
BEGINNING = (START[0], START[1], ENEMYSTART[0], ENEMYSTART[1], 1)
start_time = time()
simulation = hardcode(simulate_episode, behavioral_policy,
    BEGINNING,
    ACTION_SET, play_states, asylum,
    MAX_TIMESTEPS, GOAL, SIZE,
    STEP_REWARD, GOAL_REWARD, VICTORY_REWARD, DEFEAT_REWARD, 0.0,
    n=1)

target_policy = np.load(LOAD)
history = simulation(target_policy)

## Game start ------------------------------------------------------------------
pygame.font.init() # Iniciar texto en pygame
font = pygame.font.Font(None, 20)
colors = [pygame.Color("black"), REDBLU]
agent = Entity(img_rob, START[0], START[1], screen, RESOLUTION)
demon = Entity(img_dev, ENEMYSTART[0], ENEMYSTART[1], screen, RESOLUTION)
actual_time = 0
## Event Loop
n_photo = 0 ## For photo saving
while True:
    ## Eventos -----------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                actual_time = max(actual_time-1, 0)

            if event.key == pygame.K_RIGHT:
                actual_time = min(actual_time+1, len(history)-1)
            
            if event.key == pygame.K_UP:
                actual_time = 0

            if event.key == pygame.K_DOWN:
                actual_time = len(history)-1

            if event.key == pygame.K_f: #Photo
                pygame.image.save(screen, "Out/Demon{}.png".format(
                    str(n_photo).zfill(3)))
                n_photo += 1

    ## Dibujar -----------------------------
    screen.fill(LIGHTBLU)
    drawGrid(screen, XMAX, YMAX, RESOLUTION, GRAY, GOAL)

    time_text = font.render("TIME: {}".format(actual_time),
                            True, pygame.Color("black"))
    reward_text = font.render("REWARD: {}".format(history[actual_time][0]),
                            True, pygame.Color("black"))
    gen = sum(history[n][0] for n in range(actual_time+1))
    cum_reward_text = font.render("G: {}".format(gen),
                            True, pygame.Color("black"))
    
    screen.blit(time_text, (0*RESOLUTION+2, 1))
    screen.blit(reward_text, (1*RESOLUTION+2, 1))
    screen.blit(cum_reward_text, (2*RESOLUTION+2, 1))

    current_state = history[actual_time][1]
    agent.x = current_state[0]; agent.y = current_state[1]
    demon.x = current_state[2]; demon.y = current_state[3]
    agent.draw()
    if current_state[4]:
        demon.draw()

    pygame.display.update()