import numpy as np
import time

ROW = 0
COL = 1

UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])


class MazeMaker:
    """
    This MazeMaker class generates a randomized maze containing 4 values.
    0: empty cell (the path)
    1: wall
    2: start
    3: end
    """
    def __init__(self, rows: int, columns: int, obstacle_ratio: float, minimum_route_length: int, seed: int = None) -> None:
        """
        Initialize the MazeMaker object.

        Parameters:
            rows (int): The number of rows in the maze.
            columns (int): The number of columns in the maze.
            obstacle_ratio (float): The ratio of obstacles to empty cells.
            minimum_route_length (int): The minimum required length of the route.
        """
        self.rows = rows
        self.columns = columns
        self.directions = [UP, DOWN, LEFT, RIGHT]
        np.random.seed(seed)

        self.obstacle_ratio = obstacle_ratio
        self.minimum_route_length = minimum_route_length
        self.start, self.goal = self.random_start_finish()
        self.final_grid = self.build_grid(self.obstacle_ratio)


    def random_start_finish(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        This method generates a random coordinate for the player to start.
        Based on that coordinate the opposite coordinate is calculated and will
        be set as the end coordinate.

        Returns:
            tuple: Tuple containing start coordinates and goal coordinates.
        """
        side = np.random.randint(0, 4)

        if side == 0:    # top
            start = (0, np.random.randint(0, self.columns))
        elif side == 1:  # bottom
            start = (self.rows - 1, np.random.randint(0, self.columns))
        elif side == 2:  # left
            start = (np.random.randint(0, self.rows), 0)
        elif side == 3:  # right
            start = (np.random.randint(0, self.rows), self.columns - 1)

        return start, (self.rows - start[ROW] - 1, self.columns - start[COL] - 1)


    def get_neighbours(self, location: tuple[int, int], grid: np.ndarray) -> list[tuple[int, int]]:
        """
        This method returns a list of all valid neighbours for a given location.

        Parameters:
            square (tuple): The coordinates of the square.
            search_grid (np.ndarray): The grid to search for neighbours.

        Returns:
            list: A list of valid neighbours.
        """
        neighbours = []

        if location[ROW] > 0 and grid[a := tuple(location + UP)] < -1:
            neighbours.append(a)

        if location[ROW] < self.rows-1 and grid[a := tuple(location + DOWN)] < -1:
            neighbours.append(a)

        if location[COL] > 0 and grid[a := tuple(location + LEFT)] < -1:
            neighbours.append(a)

        if location[COL] < self.columns-1 and grid[a := tuple(location + RIGHT)] < -1:
            neighbours.append(a)

        return neighbours


    def get_neighbour_by_value(self, grid: np.ndarray, location: tuple[int, int],  value: int) -> tuple[int, int]:
        """
       This method checks and returns the neighbouring locations for a given
        value (0: path or 1: wall).

        Parameters:
            square (tuple): The coordinates of the square.
            search_grid (np.ndarray): The grid to search for neighbours.
            value (int): The value to search for.

        Returns:
            tuple: The coordinates of the neighbouring square with the specified value.
        """
        if location[ROW] > 0 and grid[a := tuple(location + UP)] == value:
            return a
        elif location[ROW] < self.rows-1 and grid[a := tuple(location + DOWN)] == value:
            return a
        elif location[COL] > 0 and grid[a := tuple(location + LEFT)] == value:
            return a
        elif location[COL] < self.columns-1 and grid[a := tuple(location + RIGHT)] == value:
            return a


    def explore_path(self, new_expand: list[tuple[int, int]], grid: np.ndarray) -> bool:
        length = 0
        while len(to_expand := new_expand) != 0:  # as long as there is something to expand
            new_expand = []
            length += 1
            for position in to_expand:

                for neighbour in self.get_neighbours(position, grid):
                    if grid[neighbour] == -3:  # a legal stepping stone
                        grid[neighbour] = length
                        new_expand.append(neighbour)  # save for next expand
                    elif grid[neighbour] == -2:  # exit is found
                        grid[neighbour] = length
                        return True
        return False
    

    def trace_route(self, grid: np.ndarray, finish: tuple[int, int]) -> list[tuple[int, int]]:
        # trace back the route

        route = [finish]
        while (length:= grid[route[0]]) != 0:
            route[:0] = [self.get_neighbour_by_value(grid, route[0], length := length - 1)]

        return route
    

    def find_route(self, start: tuple[int, int], finish: tuple[int, int], grid_with_obstacles: np.ndarray) -> list[tuple[int, int]]:
        """
        This method uses breadth first search in order to determine a route
        between the start and end coordinates, given the obstacles.
        """

        search_grid = np.full((self.rows, self.columns), -3, dtype=np.int32)
        search_grid[start], search_grid[finish] = (0, -2)

        search_grid = np.vectorize(lambda a, b: -1 if b == 1 else a)(search_grid, grid_with_obstacles)

        if self.explore_path([start], search_grid):
            # return self.get_route([finish], search_grid)
            return self.trace_route(search_grid, finish)

        return []  # no path in this maze


    def path_exists(self, min_route_length: int, grid) -> bool:
        """
        This method checks if a route is possible and meets the minimal
        length requirement.

        Parameters:
            min_route_length (int): The minimum length of the route.

        Returns:
            bool: True if a valid route exists, False otherwise.
        """
        found_route = self.find_route(self.start,
                                      self.goal,
                                      grid)
        if len(found_route) >= min_route_length:
            self.optimal_route = found_route
            return True
        else:
            self.optimal_route = None
            return False


    def build_grid(self, obstacle_ratio: float) -> np.ndarray:
        """
        Creates obstacles for the maze based on the obstacle ratio and sets the start
        and goal coordinates.

        Returns:
            np.ndarray: Grid.
        """
        while True:
            grid = np.random.choice(
                            [0, 1], 
                            (self.rows, self.columns), 
                            p=[1-obstacle_ratio, obstacle_ratio])
            
            grid[self.start], grid[self.goal] = (2, 3)

            if self.path_exists(self.minimum_route_length, grid):
                break

        return grid


    def get_maze(self) -> np.ndarray:
        """
        Returns the generated maze with obstacles.

        Returns:
            np.ndarray: The maze grid.
        """
        return self.final_grid


    def get_optimal_route(self) -> list[tuple[int, int]]:
        """
        Returns the optimal route found in the maze.

        Returns:
            list: List of tuples representing the optimal route.
        """
        return self.optimal_route


    def get_directions(self, steps: list[tuple[int, int]]) -> list[int]:
        """
        This method returns a list with directions taken given a path.
        0: up, 1: down, 2: left, 3: right.

        Parameters:
            step_list (list): The list of steps representing the path.

        Returns:
            list: A list of directions.
        """
        actions = {UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3}

        directions = [actions[(end[ROW] - start[ROW], end[COL] - start[COL])]
                      for start, end in zip(steps, steps[1:])]
        
        return directions


    def get_start_coor(self) -> tuple[int]:
        """
        Returns the start coordinates.

        Returns:
            tuple (int): Start coordinates.
        """
        return self.start


    def get_goal_coor(self) -> tuple[int]:
        """
        Returns the goal coordinates.

        Returns:
            tuple (int): Goal coordinates.
        """
        return self.goal


if __name__ == '__main__':
    maze = MazeMaker(10, 10, 0.5, 10)
    print(maze.get_maze())
