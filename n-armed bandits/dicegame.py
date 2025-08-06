import random
from banditagent import BanditAgent
import pygame

class DiceGameEnv:
    def __init__(self, T):
        self.T = T
        self.turn = 0
        self.score = 0.
        self.done = False
        self.actions = {
            0: 4,   # d4
            1: 6,   # d6
            2: 8,   # d8
            3: 10,  # d10
            4: 12,  # d12
            5: 20   # d20
        }

    def reset(self):
        self.turn = 0
        self.score = 0.
        self.done = False
        return self.get_state()

    def step(self, action):
        die_faces = self.actions[action]
        roll = random.randint(1, die_faces)
        self.turn += 1
        self.score += (1/self.turn) * (roll - self.score)
        if self.turn >= self.T:
            self.done = True
        # State, reward, done
        return self.get_state(), roll, self.done

    def get_state(self):
        return {
            "turn": self.turn,
            "game_length": self.T,
            "current_score": self.score
        }

    def available_actions(self):
        return list(self.actions.keys())

if __name__ == '__main__':

    ## PyGame Setup
    from parameters import EPSILON, ALPHA, VALUE, GAMELENGTH, \
        XMAX, YMAX, LIGHTBLU
    pygame.init()
    screen = pygame.display.set_mode((XMAX, YMAX))
    pygame.display.set_caption("Dice Bandit Game")

    font = pygame.font.Font(None, 36)
    big_font = pygame.font.Font(None, 48)

    GRAY = (200, 200, 200)
    BLACK = (0, 0, 0)

    # Define dice buttons
    dice_sprites = [
        pygame.image.load('Assets/d4.png'),
        pygame.image.load('Assets/d6.png'),
        pygame.image.load('Assets/d8.png'),
        pygame.image.load('Assets/d10.png'),
        pygame.image.load('Assets/d12.png'),
        pygame.image.load('Assets/d20.png')
    ]
    dice_actions = [0, 1, 2, 3, 4, 5]
    buttons = []
    rects = []

    spacing = 120
    x_pos = 400
    y_pos = 50

    for i, sprite in enumerate(dice_sprites):
        coords = (x_pos + i%3 * spacing, y_pos + i//3 * 1.1 * spacing)
        buttons.append((sprite, coords))
        rects.append(sprite.get_rect(topleft=coords))

    ## Game
    env = DiceGameEnv(T=GAMELENGTH)
    state = env.reset()
    agent = BanditAgent(env, epsilon=EPSILON, alpha=ALPHA, startvalue=VALUE)

    def draw_text(text, x, y, font, color=BLACK):
        rendered = font.render(text, True, color)
        screen.blit(rendered, (x, y))

    def draw_q_table(agent, env, x, y, font):
        for action, value in agent.Q.items():
            die_faces = env.actions[action]
            die_label = f"d{die_faces}"
            q_text = f"Q[{die_label}]:" + \
                ("  " if die_faces < 10 else " ") + f"{value:.2f}"
            draw_text(q_text, x, y, font)
            y += 30


    running = True
    message = ""
    last_roll = ""
    while running:
        screen.fill(LIGHTBLU)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not env.done and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for action, rect in enumerate(rects):
                    if rect.collidepoint(mouse_pos):
                        state, reward, done = env.step(action)
                        agent.learn(action, reward)
                        die_faces = env.actions[action]
                        last_roll = \
                            f"Rolled d{env.actions[action]}: {reward}"
                        message = f"Avg score: {state['current_score']:.2f}"

        # Draw buttons
        for sprite, position in buttons:
            screen.blit(sprite, position)

        # Draw game status
        draw_text(f"Turn {state['turn']}/{env.T}", 30, 30, big_font)
        draw_text(last_roll, 30, 80, font)
        draw_text(message, 30, 120, font)
        draw_q_table(agent, env, 200, 400, font)

        if env.done:
            draw_text("Game Over", 30, 180, big_font)
            draw_text(
                f"Final Score: {state['current_score']:.2f}", 30, 230, font)

        pygame.display.flip()

    pygame.quit()
