import pygame

## Draw a grid on pygame surface of dimensions xmax, ymax
## Optionally draw a rectangle at position goal
def drawGrid(surface, xmax, ymax, resolution, linecolor=(50, 50, 50), \
             goal=False, goalcolor=(100, 200, 200)):
    if goal:
        rect = pygame.Rect(goal[0] * resolution, goal[1] * resolution,
                        resolution, resolution)
        pygame.draw.rect(surface, goalcolor, rect)
    for x in range(0, xmax, resolution):
        for y in range(0, ymax, resolution):
            rect = pygame.Rect(x, y, resolution, resolution)
            pygame.draw.rect(surface, linecolor, rect, 1)