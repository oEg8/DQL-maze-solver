from MazeMaker import MazeMaker
import numpy as np

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class MazeEnv:
    def __init__(self, rows: int, columns: int, seed: int = None) -> None:
        """
        Initializes the maze environment.

        Parameters:
            rows (int): Number of rows in the maze.
            columns (int): Number of columns in the maze.
            seed (int): Seed for random number generator.
        """
        self.rows = rows
        self.columns = columns
        self.seed = seed

        maze = MazeMaker(rows, rows, 0.4, (rows+columns)/2)
        self.grid = maze.return_maze()
        self.start = maze.return_start_coor()
        self.goal = maze.return_goal_coor()
        self.position = [self.start[0], self.start[1]]


    def move(self, action: int) -> list[int]:
        """
        Moves the agent in the maze based on the given action.

        Parameters:
            action (int): The action to move the agent. Can be one of UP, DOWN, LEFT or RIGHT.

        Returns:
            List[int]: The new position of the agent after performing the action.
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


    def possible_actions(self) -> list[int]:
        """
        Checks the possible actions the agent can take from its current position.

        Returns:
            List[int]: A list of possible actions (one or more of UP, DOWN, LEFT, RIGHT).
        """
        actions = []
        if self.position[ROW] - 1 >= 0 and self.grid[self.position[ROW] - 1][self.position[COL]] != 1:
            actions.append(UP)
        if self.position[ROW] + 1 < len(self.grid) and self.grid[self.position[ROW] + 1][self.position[COL]] != 1:
            actions.append(DOWN)
        if self.position[COL] - 1 >= 0 and self.grid[self.position[ROW]][self.position[COL] - 1] != 1:
            actions.append(LEFT)
        if self.position[COL] + 1 < len(self.grid[ROW]) and self.grid[self.position[ROW]][self.position[COL] + 1] != 1:
            actions.append(RIGHT)

        return actions
    

    def get_state_size(self) -> int:
        """
        Returns the size of the state space, which in this case, includes the flattened grid size and the position coordinates.

        Returns:
            int: The size of the state space.
        """
        return self.grid.size + 2


    def get_grid(self) -> np.ndarray:
        """
        Returns the current grid (maze).

        Returns:
            List[List[int]]: 2D list representing the maze grid.
        """
        return self.grid


    def get_position(self) -> list[int]:
        """
        Returns the current position of the agent.

        Returns:
            List[int]: Current position of the agent as [row, column].
        """
        return self.position


    def get_start(self) -> tuple[int]:
        """
        Returns the starting position of the agent.

        Returns:
            List[int]: Starting position of the agent as [row, column].
        """
        return self.start


    def get_goal(self) -> tuple[int]:
        """
        Returns the goal position of the maze.

        Returns:
            List[int]: Goal position of the maze as [row, column].
        """
        return self.goal