
## Reinforcement stuff
SIZE = 8 # board size (NxN)
GOAL = (6, 2)
STEP_REWARD = -1
PIZZA_REWARD = 3
ACTION_SET = ["↑", "↓", "→", "←"]
THETA = 0.1
GAMMA = 1.0 ## Sin decaimiento porque el problema es finito

## Size in pixels
XMAX = 640
YMAX = XMAX
RESOLUTION = int(XMAX/SIZE)

## Colors
BLACK = (0, 0, 0)
LIGHTBLU = (150, 150, 250)
GREENBLU = (100, 200, 200)
BLACK = (5, 5, 5)
GRAY = (50, 50, 50)

## Graphic stuff
ARROWSIZE = 0.001625*RESOLUTION
DRAW_DISPLACEMENT_FACTOR_X = 0.3
DRAW_DISPLACEMENT_FACTOR_Y = 0.4
DRAW_DISPLACEMENT_FACTOR_XA = 0.5
DRAW_DISPLACEMENT_FACTOR_YA = 0.5