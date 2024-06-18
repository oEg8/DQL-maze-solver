from MazeMaker import MazeMaker
from Visualiser import Visualiser
import numpy as np
import os

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

MOVE_PLAYER = 0
MOVE_GOAL = 1


class MazeEnv:
    """
    This class defines a maze environment where a reinforcement learning agent can navigate trough.

    
    Attributes
    __________
    reset               : tuple[np.ndarray, tuple[int, int], list[int]]
                        Resets the environment variables and generates a new maze.
    test_for_completion : bool
                        Checks if the agent has reached the goal.
    calculate_state     : np.ndarray
                        Calculates the state, which is the flattened grid, the current position of the agent and 
                        the action type.
    get_action_space    : list[int]
                        Returns the action space.
    get_action_size     : int
                        Returns the size of the action space.
    get_action_type     : int
                        Selects the action type.
    move                : list[int]
                        Moves the agent in the maze based on the given action and action_type.
    move_position       : tuple[int, int]
                        This method changes the position of the agent.
    get_state_size      : int
                        Returns the size of the state space, which in this case, includes the flattened grid size, 
                        the position coordinates and the action type.
    possible_actions    : list[int]
                        Checks the possible actions the agent can take from its current position.
    get_grid            : np.ndarray
                        Returns the current grid (maze).
    get_position        : list[int]
                        Returns the current position of the agent.
    get_start           : tuple[int, int]
                        Returns the starting position of the agent.
    get_goal            : tuple[int, int]
                        Returns the goal position of the maze.
    save_grid           : None
                        Saves the grids in a folder.
    """
    def __init__(self, rows: int, columns: int, visualise: bool = False, seed: int = None, save_grids: bool = False) -> None:
        """
        Initializes the maze environment.

        Parameters:
            rows (int): Number of rows in the maze.
            columns (int): Number of columns in the maze.
            visualise (bool): Whether to visualize the maze. Default is False.
            seed (int): Seed for random number generator.
        """
        self.rows = rows
        self.columns = columns
        self.visualise = visualise
        self.seed = seed
        self.save_grids = save_grids

        maze = MazeMaker(self.rows, self.rows, 0.4, (self.rows+self.columns)/2, seed)
        self.grid = maze.get_maze()
        self.start = maze.get_start()
        self.goal = [maze.get_goal()[0], maze.get_goal()[1]]
        self.position = [self.start[0], self.start[1]]
        self.episode = 0

        self.type_switch = 5

        if self.visualise:
            self.visualiser = Visualiser()


    def reset(self) -> tuple[np.ndarray, tuple[int, int], list[int]]:
        """
        Resets the environment variables and generates a new maze.

        Returns:
            tuple: grid, position coordinates, total reward, step
        """
        # generates a new maze
        self.episode =+ 1
        maze = MazeMaker(self.rows, self.rows, 0.4, (self.rows+self.columns)/2)
        self.grid = maze.get_maze()
        if self.save_grids:
            self.save_grid(self.grid)
        self.start = maze.get_start()
        self.goal = [maze.get_goal()[0], maze.get_goal()[1]]
        self.position = [self.start[0], self.start[1]]
        self.total_reward = 0
        self.step = 0

        return self.grid, self.position, self.total_reward, self.step
    

    def test_for_completion(self) -> bool:
        """
        Checks if the agent has reached the goal.

        Returns:
            bool: True if the agent has reached the goal, False otherwise.
        """
        return self.grid[self.position[ROW]][self.position[COL]] == 3
    

    def calculate_state(self) -> np.ndarray:
        """
        Calculates the state, which is the flattened grid, the current position of the agent and the action type.

        Returns:
            np.ndarray: The current state.
        """
        if self.step % self.type_switch == 0:
            return np.append(self.grid.flatten(), [self.position[0], self.position[1], 1])  # move the goal
        else:
            return np.append(self.grid.flatten(), [self.position[0], self.position[1], 0])  # move the player


    def get_action_space(self, action_type: int) -> list[int]:
        """
        Returns the action space.

        Parameters:
            action_type (int): The action type to be taken.

        Returns:
            list: List of possible actions.
        """
        if action_type == 0:
            return [0, 1, 2, 3]  # action_type 1
        elif action_type == 1:
            return [4, 5, 6, 7]  # action_type 2
    

    def get_action_size(self) -> int:
        """
        Returns the size of the action space.

        Returns:
            int: Size of the action space.
        """
        return 8
    

    def get_action_type(self) -> int:
        """
        Selects the action type.
        
        Returns:
            int: Action type.
        """
        return 1 if self.step % self.type_switch == 0 else 0


    def move(self, action: int, action_type: int) -> list[int]:
        """
        Moves the agent in the maze based on the given action and action_type.

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


    def move_position(self, position: tuple[int, int], action: int) -> tuple[int, int]:
        """
        This method changes the position of the agent.
        
        Parameters: 
            position (tuple): The position coordinates.
            action (int): What action to take
            
        Returns:
            tuple: The position coordinates.
        """
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
    

    def save_grid(self) -> None:
        """
        Saves the grids in a folder.
        """
        if os.path.exists('grids'):
            np.save(f'grid{self.episode}.npy', self.grid)
        else:
            os.mkdir('grids')
            np.save(f'grids/grid{self.episode}.npy', self.grid)
    

if __name__ == '__main__':
    m = MazeEnv(5, 5, 2)
    print(m.get_grid())
    print(m.possible_actions(1))
    m.move(6, 1)
    print(m.get_grid())