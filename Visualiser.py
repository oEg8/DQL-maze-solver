import pygame
import sys
import numpy as np

ROW = 0
COL = 1


class Visualiser:
    """
    This class uses the pygame library to create a visualisation of a maze,
    start coordinate, end coordinate, and a route.

    - The maze is represented as a numpy array:
        0: empty cell (path)
        1: wall
        2: start
        3: end
    """
    def __init__(self, fps: int = 1) -> None:
        """
        Initializes the Visualiser object.

        Parameters:
            fps (int): Frames per second for the visualisation. Default is 1.
        """
        self.fps = fps
        pygame.init()

        # self.grid = grid
        # self._position = position
        # self.position = self._position

        self.width = 800
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.grid_color = (200, 200, 200)

        self.font = pygame.font.SysFont('Arial', 24)
        self.pause_text = self.font.render('Paused, press "p"',
                                                 True, self.black,
                                                 self.grid_color)
        self.solved_text = self.font.render('Maze solved!',
                                                     True,
                                                     self.black,
                                                     self.grid_color)
        
        self.solved = False

    def draw_maze(self, grid: np.ndarray, start: tuple[int, int], position: list[int], epoch_cost: float, step: int, win_count: int) -> None:
        """
        Visualizes the maze.

        The following keyboard inputs are allowed:
        - 'p': Plays and pauses the visualisation.
        - 'Esc': Stops and exits the visualisation.
        - 'up arrow': Increases fps by 1.
        - 'down arrow': Decreases fps by 1.
        - 'r': Resets and replays the visualisation.

        Parameters:
            grid (np.ndarray): The maze grid.
            start (Tuple[int, int]): Starting coordinates of the maze.
            position (List[int]): Current position of the agent.
            epoch_cost (float): Cost for the current epoch.
            step (int): Current step number.
            win_count (int): Number of wins.
        """
        cell_size = min(self.width // np.shape(grid)[COL],
                        self.height // np.shape(grid)[ROW])
        pygame.display.set_caption('Maze')
        clock = pygame.time.Clock()

        RUNNING, PAUSE = 0, 1
        state = RUNNING

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
                    self.fps += 1
                elif event.key == pygame.K_DOWN and self.fps > 0:
                    self.fps -= 1
                elif event.key == pygame.K_p:
                    if state == RUNNING:
                        state = PAUSE
                    else:
                        state = RUNNING
                elif event.key == pygame.K_r:
                    position = [start[ROW], start[COL]]
                    self.solved = False
                    state = PAUSE

        # When the player reaches the goal, a text will be displayed and the
        # visualisation will be closed after the predetermined delay.
        if grid[position[0]][position[1]] == 3 and not self.solved:
            self.solved = True
            self.screen.blit(self.solved_text,
                                (self.width // 2 - self.solved_text.get_width() // 2, self.height // 2 - self.solved_text.get_height() // 2))
            pygame.display.flip()

        # Draws the screen and the grid with the corresponding color code.
        self.screen.fill(self.white)
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if row == position[ROW] and col == position[COL]:
                    color = self.blue
                elif grid[row][col] == 3:
                    color = self.green
                elif grid[row][col] == 2:
                    color = self.red
                elif grid[row][col] == 1:
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
            
        epoch_cost_text = self.font.render(f'Epoch Cost: {epoch_cost}', True, self.black, self.grid_color)
        self.screen.blit(epoch_cost_text, (10, 10))

        step_text = self.font.render(f'Step: {step}', True, self.black, self.grid_color)
        self.screen.blit(step_text, (10, 40))

        win_count_text = self.font.render(f'Win count: {win_count}', True, self.black, self.grid_color)
        self.screen.blit(win_count_text, (10, 70))

        pygame.display.flip()
        clock.tick(self.fps)
