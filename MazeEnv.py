from MazeMaker import MazeMaker
import numpy as np

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

MOVE_PLAYER = 0
MOVE_GOAL = 1


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

        maze = MazeMaker(self.rows, self.rows, 0.4, (self.rows+self.columns)/2, seed)
        self.grid = maze.get_maze()
        self.start = maze.get_start_coor()
        self.goal = [maze.get_goal_coor()[0], maze.get_goal_coor()[1]]
        self.position = [self.start[0], self.start[1]]


    def reset(self) -> tuple[np.ndarray, tuple[int, int], list[int]]:
        """
        Resets the environment and generates a new maze.

        Returns:
            tuple: grid, start coordinates, position coordinates
        """
        # generates a new maze
        maze = MazeMaker(self.rows, self.rows, 0.4, (self.rows+self.columns)/2)
        self.grid = maze.get_maze()
        self.start = maze.get_start_coor()
        self.goal = [maze.get_goal_coor()[0], maze.get_goal_coor()[1]]
        self.position = [self.start[0], self.start[1]]
        self.total_reward = 0
        self.step = 0

        return self.grid, self.position, self.total_reward, self.step


    def move(self, action: int, action_type: int) -> list[int]:
        """
        Moves the agent in the maze based on the given action.

        Parameters:
            action (int): The action to move the agent. Can be one of UP, DOWN, LEFT or RIGHT.
            action (type): The action type that needs to be taken.

        Returns:
            List[int]: The new position of the agent after performing the action.
        """
        if action_type == MOVE_PLAYER:
            self.position = self.move_position(self.position, action)
        elif action_type == MOVE_GOAL:
            action = action - 4
            self.grid[self.goal[0], self.goal[1]] = 0
            self.goal = self.move_position(self.goal, action)
            self.grid[self.goal[0], self.goal[1]] = 3


    def move_position(self, position, action):
        if action == UP:  
            position[ROW] -= 1
        elif action == DOWN:
            position[ROW] += 1
        elif action == LEFT:
            position[COL] -= 1
        elif action == RIGHT:
            position[COL] += 1

        return position


    def get_state_size(self) -> int:
        """
        Returns the size of the state space, which in this case, includes the flattened grid size, 
        the position coordinates and the action type.

        Returns:
            int: The size of the state space.
        """
        return self.rows * self.columns + 3

    def possible_actions(self, action_type: int) -> list[int]:
        """
        Checks the possible actions the agent can take from its current position.

        Parameters:
            action_type (int): The action type the possible actions need be checked for.

        Returns:
            List[int]: A list of possible actions (one or more of UP, DOWN, LEFT, RIGHT).
        """
        actions = []
        if action_type == 0:
            position = self.position
        elif action_type == 1:
            position = self.goal


        if position[ROW] - 1 >= 0 and self.grid[position[ROW] - 1][position[COL]] != 1:
            actions.append(UP)
        if position[ROW] + 1 < len(self.grid) and self.grid[position[ROW] + 1][position[COL]] != 1:
            actions.append(DOWN)
        if position[COL] - 1 >= 0 and self.grid[position[ROW]][position[COL] - 1] != 1:
            actions.append(LEFT)
        if position[COL] + 1 < len(self.grid[ROW]) and self.grid[position[ROW]][position[COL] + 1] != 1:
            actions.append(RIGHT)

        if action_type == 0:
            return actions
        if action_type == 1:
            return [a + 4 for a in actions]
        
    

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
    

if __name__ == '__main__':
    m = MazeEnv(5, 5, 2)
    print(m.get_grid())
    print(m.possible_actions(1))
    m.move(6, 1)
    print(m.get_grid())
