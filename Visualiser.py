import pygame
import sys
import numpy as np

ROW = 0
COL = 1


class Visualiser:
    """
    This class uses the pygame library to create a visualisation given a maze,
    start coordinate, end coordinate and a route.

    - The maze being a numpy array (0: empty cell, 1: wall, 2: start, 3: end)
    - The route being a list of directions (0: up, 1: down, 2: left, 3: right)
    """
    def __init__(self, grid: np.ndarray, pos: tuple[int, int], goal: tuple[int, int], route: list[tuple[int, int]]) -> None:
        pygame.init()

        self.grid = grid
        self.pos = pos
        self.route = route
        self.goal = goal

        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.grid_color = (200, 200, 200)

        self.pause_text = pygame.font.SysFont('Consolas', 52)
        self.pause_text = self.pause_text.render('Paused, press "p"',
                                                 True, self.black,
                                                 self.grid_color)
        self.solved_text = pygame.font.SysFont('Consolas', 52)
        self.solved_text = self.solved_text.render('Maze solved!',
                                                     True,
                                                     self.black,
                                                     self.grid_color)
        
        self.solved = False


    def draw_maze(self):
        """
        This method draws the maze and is able to receive some keyboard inputs:
        'p':    Plays and pauses the visualisation
        'Esc':  Stops and exits the visualisation
        'up arrow':     Increases fps with 1
        'down arrow':   Decreases fps with 1
        'r':    Resets and replays the visualisation
        """
        cell_size = min(self.width // np.shape(self.grid)[COL],
                        self.height // np.shape(self.grid)[ROW])
        pygame.display.set_caption('Maze')
        clock = pygame.time.Clock()

        fps = 2
        route_index = 0

        run = True
        RUNNING, PAUSE = 0, 1
        state = PAUSE

        while run:
            # Gets the keyboard inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_UP:
                        fps += 1
                    elif event.key == pygame.K_DOWN and fps > 0:
                        fps -= 1
                    elif event.key == pygame.K_p:
                        if state == RUNNING:
                            state = PAUSE
                        else:
                            state = RUNNING
                    elif event.key == pygame.K_r:
                        self.pos = [start[ROW], start[COL]]
                        route_index = 0
                        self.solved = False
                        state = PAUSE

            # When the player reaches the goal, a text will be displayed and the
            # visualisation will be closed after the predetermined delay.
            if self.pos[ROW] == self.goal[ROW] and self.pos[COL] == self.goal[COL] and not self.solved:
                self.solved = True
                self.screen.blit(self.solved_text,
                                 (self.width // 2 - self.solved_text.get_width() // 2, self.height // 2 - self.solved_text.get_height() // 2))
                pygame.display.flip()
            if self.solved:
                continue


            # Moves the player to the next position.
            if state == RUNNING:
                if route_index < len(self.route):
                    next_move = self.route[route_index]
                    route_index += 1
                    if next_move == 0:
                        self.pos[ROW] -= 1
                    elif next_move == 1:
                        self.pos[ROW] += 1
                    elif next_move == 2:
                        self.pos[COL] -= 1
                    elif next_move == 3:
                        self.pos[COL] += 1

            # Draws the screen and the grid with the corresponding color code.
            self.screen.fill(self.white)
            for row in range(len(self.grid)):
                for col in range(len(self.grid[row])):
                    if self.grid[row][col] == 3:
                        color = self.green
                    elif row == self.pos[ROW] and col == self.pos[COL]:
                        color = self.blue
                    elif self.grid[row][col] == 2:
                        color = self.red
                    elif self.grid[row][col] == 1:
                        color = self.black
                    else:
                        color = self.white

                    pygame.draw.rect(self.screen,
                                     color,
                                     (col * cell_size, row * cell_size, cell_size, cell_size))

            # Draws the gridlines.
            for x in range(0, self.width, cell_size):
                pygame.draw.line(self.screen,
                                 self.grid_color,
                                 (x, 0),
                                 (x, self.height),
                                 1)
            for y in range(0, self.height, cell_size):
                pygame.draw.line(self.screen,
                                 self.grid_color,
                                 (0, y),
                                 (self.width, y),
                                 1)

            pygame.draw.rect(self.screen,
                             self.black,
                             (0, 0, self.width,
                              self.height), width=3)

            # Draws the pause textbox.
            if state == PAUSE:
                self.screen.blit(self.pause_text,
                                 (self.width // 2 - self.pause_text.get_width() // 2, self.height // 2 - self.pause_text.get_height() // 2))

            pygame.display.flip()
            clock.tick(fps)


if __name__ == '__main__':
    grid = np.array([[2, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                    [0, 1, 1, 0, 1, 1, 1, 0, 1, 3],
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])

    start = (0, 0)
    goal = (8, 9)
    route = [3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1,
             1, 3, 3, 3, 0, 0, 3, 3, 3, 1, 1, 3, 1, 1, 3, 3, 0]
    Visualiser(grid, [start[ROW], start[COL]], goal, route).draw_maze()
