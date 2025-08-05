import pygame

LIGHTBLU = (150, 150, 255)
GREENBLU = (100, 200, 200)
REDBLU = (180, 0, 180)
WHITE = (240, 240, 240)
BLACK = (5, 5, 5)
GRAY = (50, 50, 50)

def drawGrid(screen, goal, resolution, xmax, ymax):
    screen.fill(LIGHTBLU)
    rect = pygame.Rect(goal[1] * resolution, goal[0] * resolution,
                       resolution, resolution)
    pygame.draw.rect(screen, GREENBLU, rect)
    font = pygame.font.Font(None, 36)
    n = 1
    for x in range(0, xmax, resolution):
        n -= 1
        for y in range(0, ymax, resolution):
            n += 1
            rect = pygame.Rect(x, y, resolution, resolution)
            pygame.draw.rect(screen, GRAY, rect, 1)
            number = font.render(str(n), True, pygame.Color("black"))
            screen.blit(number, (x + resolution*0.01, y + resolution*0.01))
