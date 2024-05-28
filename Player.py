import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ROW = 0
COL = 1

class Player:
    def __init__(self, row, col) -> None:
        self.row = row
        self.col = col
        self.position = [self.row, self.col]


    def move(self, action: int) -> list[int, int]:
        """
        This method moves the player 1 step to the desired position.
        """
        if action == UP:  
                self.position[ROW] -= 1
        elif action == DOWN:
                self.position[ROW] += 1
        elif action == LEFT:
                self.position[COL] -= 1
        elif action == RIGHT:
                self.position[COL] += 1

        return self.position 


    def possible_actions(self, grid: np.ndarray) -> list:
        """
        This method returns a list of possible moves the player can take given
        a grid and a position.
        """
        actions = []
        if self.position[ROW] - 1 >= 0 and grid[self.position[ROW] - 1][self.position[COL]] != 1:
            actions.append(UP)
        if self.position[ROW] + 1 < len(grid) and grid[self.position[ROW] + 1][self.position[COL]] != 1:
            actions.append(DOWN)
        if self.position[COL] - 1 >= 0 and grid[self.position[ROW]][self.position[COL] - 1] != 1:
            actions.append(LEFT)
        if self.position[COL] + 1 < len(grid[ROW]) and grid[self.position[ROW]][self.position[COL] + 1] != 1:
            actions.append(RIGHT)

        return actions
    

    def distance_to_goal(self, goal: tuple[int, int]) -> int:
        return abs(self.position[ROW] - goal[ROW]) + abs(self.position[COL] - goal[COL])   