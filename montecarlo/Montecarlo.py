import os
import numpy as np
import pygame
from drawfuncs import drawGrid
from grid3 import GridWorld


ALPHA = 1.0
GAMMA = 1.0
LOAD = False


env = GridWorld()
size = env.size
env.screen = pygame.display.set_mode((1280, 640)) # Override


if LOAD and os.path.exists('q_table.npy'):
    q_table = np.load('q_table.npy')
else:
    q_table = np.zeros((size * size, len(env.actions)))


def draw_q_table(screen, q_table, size, cell_size):
    pygame.draw.rect(
        screen, (240, 240, 240),
        (screen.get_width()/2, 0, screen.get_width()/2, screen.get_height()))
    font = pygame.font.Font(None, 22)
    offset_x = screen.get_width() // 2  # Shift everything right
    for i in range(size):
        for j in range(size):
            x, y = j * cell_size + offset_x, i * cell_size
            pygame.draw.rect(screen, (200, 200, 200), (x, y, cell_size, cell_size), 1)
            state_index = i * size + j
            q_values = q_table[state_index]
            
            # Draw action values
            action_positions = [(x + cell_size//2 + 5, y + 15),  # ↑
                                (x + cell_size//2 + 5, y + cell_size - 15), # ↓
                                (x + 25, y + cell_size//2),  # ←
                                (x + cell_size - 25, y + cell_size//2)]  # →
            
            for k, (qx, qy) in enumerate(action_positions):
                text_surface = font.render(f"{q_values[k]:.2f}", True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(qx, qy))
                screen.blit(text_surface, text_rect)


## Begin
state = env.reset()
trajectory = []
running = 1
while running == 1:
    for event in pygame.event.get():
        action = None
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = "↑"
            elif event.key == pygame.K_DOWN:
                action = "↓"
            elif event.key == pygame.K_LEFT:
                action = "←"
            elif event.key == pygame.K_RIGHT:
                action = "→"
            if action:
                old_state = env.state_to_index(state)
                state, reward, done = env.step(action)
                trajectory.append((old_state, action, reward))
                print(f"State: {env.state_to_index(state)}, Reward: {reward}, Done: {done}")
            if done:
                running = 2
    env.render()
    draw_q_table(env.screen, q_table, size, env.resolution)
    pygame.display.flip()
    env.clock.tick(30)

G = 0  # Initialize return
t = len(trajectory)
while running == 2:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            t -= 1
            if t < 0:
                running = 3
                break
            state, action, reward = trajectory[t]
            G = reward + GAMMA * G  # Compute return
            action_map = {'↑': 0, '↓': 1, '←': 2, '→': 3}
            state_index = state - 1
            action_index = action_map[action]
            q_table[state_index][action_index] += \
                  ALPHA * (G - q_table[state_index][action_index])

            
    env.render()
    draw_q_table(env.screen, q_table, size, env.resolution)
    pygame.display.flip()
    env.clock.tick(30)

while running == 3:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

np.save('q_table.npy', q_table)
pygame.quit()