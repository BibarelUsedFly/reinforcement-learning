import numpy as np
import pygame
from drawfuncs import drawGrid
from entity import Entity

# Gridworld Environment
class GridWorld:
    def __init__(self, size=3, xmax=640, ymax=640):
        self.size = size
        self.start_state = (2, 0)
        self.goal_state = (0, 2)
        self.bad_state = (0, 0)
        self.actions = ["↑", "↓", "←", "→"]
        self.state = self.start_state

        # Initialize Q-table
        self.q_table = np.zeros((size, size, len(self.actions)))

        # Initialize PyGame
        pygame.init()
        self.xmax = xmax
        self.ymax = ymax
        self.resolution = int(xmax/size)
        self.screen = pygame.display.set_mode((xmax, ymax))
        pygame.display.set_caption("Montecarlo")
        self.clock = pygame.time.Clock()
        self.img_main = pygame.image.load('Assets/minibidoof.png')
        self.img_skull = pygame.image.load('Assets/skull.png')
        self.img_main = pygame.transform.scale(
            self.img_main,(.8*self.resolution, .7*self.resolution))
        self.img_skull = pygame.transform.scale(
            self.img_skull,(.7*self.resolution, .8*self.resolution))
        
        self.agent = Entity(self.img_main, *self.start_state,
                            self.screen, self.resolution)
        self.skull = Entity(self.img_skull, *self.bad_state,
                            self.screen, self.resolution)

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        """Return: state, reward, terminated"""
        x, y = self.state
        
        # Action transitions
        if action == "↑":
            x = max(0, x - 1)
        elif action == "↓":
            x = min(self.size - 1, x + 1)
        elif action == "←":
            y = max(0, y - 1)
        elif action == "→":
            y = min(self.size - 1, y + 1)
        else:
            raise Exception("Illegal Action")

        self.state = (x, y)
        
        # Reward system
        if self.state == self.goal_state:
            return self.state, -1, True  # Goal reached
        elif self.state == self.bad_state:
            return self.state, -10, False  # Bad state
        else:
            return self.state, -1, False  # Step penalty
        
    def state_to_index(self, state):
        """Maps (row, col) state to a single integer index."""
        x, y = state
        return x * self.size + y + 1
    
    def render(self):
        drawGrid(self.screen, self.goal_state, self.resolution,
                 self.xmax, self.ymax)
        self.agent.x = self.state[1]
        self.agent.y = self.state[0]
        self.skull.draw()
        self.agent.draw()

if __name__ == "__main__":
    env = GridWorld()
    env.reset()