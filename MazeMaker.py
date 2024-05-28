import numpy as np

ROW = 0
COL = 1


class MazeMaker:
    """
    This MazeMaker class generates a randomized maze containing 4 values.
    0: empty cell (the path)
    1: wall
    2: start
    3: end
    """
    def __init__(self, rows: int, columns: int, obstacle_ratio: float, minimum_route_length: int) -> None:

        self.rows = rows
        self.columns = columns

        # comment for random mazes
        # np.random.seed(354)

        self.grid = np.zeros((rows, columns), dtype=np.int32)
        self.obstacle_ratio = obstacle_ratio
        self.minimum_route_length = minimum_route_length
        self.grid_with_obstacles = self.grid
        self.grid_with_obstacles = self.create_obstacles(self.obstacle_ratio)


    def random_start_goal(self) -> tuple:
        """
        This method generates a random coordinate for the player to start.
        Based on that coordinate the opposite coordinate is calculated and will
        be set as the end coordinate.
        """
        rng_side = np.random.randint(0, 4)

        if rng_side == 0:  # top
            rng_start = (0, np.random.randint(0, self.columns))
            rng_goal = (self.rows-1, self.columns-rng_start[COL]-1)
        elif rng_side == 1:  # bottom
            rng_start = (self.rows-1, np.random.randint(0, self.columns))
            rng_goal = (0, self.columns-rng_start[COL]-1)
        elif rng_side == 2:  # left
            rng_start = (np.random.randint(0, self.rows), 0)
            rng_goal = (self.rows-rng_start[ROW]-1, self.columns-1)
        elif rng_side == 3:  # right
            rng_start = (np.random.randint(0, self.rows), self.columns-1)
            rng_goal = (self.rows-rng_start[ROW]-1, 0)

        return rng_start, rng_goal


    def get_neighbours(self, square: tuple, search_grid: np.ndarray) -> list:
        """
        This method returns a list of all valid neighbours for a given square.
        """
        neighbours = []

        # neighbours = {}
        # if square[ROW] < (self.rows -1) :
        #     neighbours |= {(square[ROW]+1, square[COL]):(square[ROW]+1, square[COL])}
        if square[ROW] < (self.rows -1) and search_grid[square[ROW]+1][square[COL]] < -1:
            neighbours.append((square[ROW]+1, square[COL]))

        if square[ROW] > 0 and search_grid[square[ROW]-1][square[COL]] < -1:
            neighbours.append((square[ROW]-1, square[COL]))

        if square[COL] > 0 and search_grid[square[ROW]][square[COL]-1] < -1:
            neighbours.append((square[ROW], square[COL]-1))
            
        if square[COL] < (self.columns -1) and search_grid[square[ROW]][square[COL]+1] < -1:
            neighbours.append((square[ROW], square[COL]+1))

        return neighbours


    def get_neighbours_with_values(self, square: tuple, search_grid: np.ndarray, value: int) -> tuple:
        """
        This method checks and returns the neighbouring squares for a given
        value (0: path or 1: wall).
        """
        if square[ROW]+1 < self.rows and search_grid[square[ROW]+1][square[COL]] == value:
            return (square[ROW]+1, square[COL])
        elif square[ROW]-1 >= 0 and search_grid[square[ROW]-1][square[COL]] == value:
            return (square[ROW]-1, square[COL])
        elif square[COL]-1 >= 0 and search_grid[square[ROW]][square[COL]-1] == value:
            return (square[ROW], square[COL]-1)
        elif square[COL]+1 < self.columns and search_grid[square[ROW]][square[COL]+1] == value:
            return (square[ROW], square[COL]+1)


    def find_route(self, start_coor: tuple, end_coor: tuple, grid_with_obstacles: np.ndarray) -> list:
        """
        This method uses breadth first search in order to determine a route
        between the start and end coordinates, given the obstacles.
        """
        route = []
        search_grid = np.full((self.rows, self.columns), -3, dtype=np.int32)
        search_grid[(start_coor[ROW], start_coor[COL])] = 0
        search_grid[(end_coor[ROW], end_coor[COL])] = -2
        for i in range(self.rows):
            for j in range(self.columns):
                if grid_with_obstacles[i][j] == 1:
                    search_grid[i][j] = -1

        length = 0
        new_expand = [(start_coor[ROW], start_coor[COL])]
        searching = True
        while searching:
            to_expand = []
            to_expand.extend(new_expand)
            new_expand = []
            for square in to_expand:
                neighbours = self.get_neighbours(square, search_grid)
                for n in neighbours:
                    if search_grid[n[ROW]][n[COL]] == -2: # exit found
                        searching = False
                        search_grid[n[ROW]][n[COL]] = length+1 # increase length of path
                    elif search_grid[n[ROW]][n[COL]] == -3: # path
                        search_grid[n[ROW]][n[COL]] = length+1 # increase length of path
                        new_expand.append((n[ROW], n[COL]))
            length += 1
            if len(new_expand) == 0:
                searching = False

        if search_grid[end_coor[ROW]][end_coor[COL]] > 0:
            max_length = search_grid[end_coor[ROW]][end_coor[COL]]
            route.append((end_coor[ROW], end_coor[COL]))
            square = (end_coor[ROW], end_coor[COL])
            while max_length > 0:
                next_square = self.get_neighbours_with_values(square,
                                                              search_grid,
                                                              max_length-1)
                route.append(next_square)
                square = next_square
                max_length -= 1
            route.reverse()
        return route


    def path_check(self, min_route_length: int) -> bool:
        """
        This method checks if a route is possible and meets the minimal
        length requirement.
        """
        found_route = self.find_route(self.start_coor,
                                      self.goal_coor,
                                      self.grid_with_obstacles)
        if len(found_route) >= min_route_length:
            self.optimal_route = found_route
            return True
        else:
            self.optimal_route = None
            return False


    def create_obstacles(self, obstacle_ratio: float) -> np.ndarray:
        """
        This method creates the obstacles for the maze.
        """
        while True:
            self.grid_with_obstacles = np.random.choice([0, 1],
                                                        (self.rows, self.columns),
                                                        p=[1-obstacle_ratio, obstacle_ratio])  # randomly creates obstacles
            self.start_coor, self.goal_coor = self.random_start_goal()
            self.grid_with_obstacles[self.start_coor] = 2
            self.grid_with_obstacles[self.goal_coor] = 3
            path_exists = self.path_check(self.minimum_route_length)
            if path_exists:
                break

        return self.grid_with_obstacles


    def return_maze(self) -> np.ndarray:
        return self.grid_with_obstacles


    def return_optimal_route(self) -> list[tuple]:
        return self.optimal_route


    def return_directions(self, step_list: list) -> list:
        """
        This method returns a list with directions taken given a path.
        0: up, 1: down, 2: left, 3: right.
        """
        directions = []
        for i in range(len(step_list) - 1):
            current_step = step_list[i]
            next_step = step_list[i + 1]
            direction = (next_step[ROW] - current_step[ROW], 
                         next_step[COL] - current_step[COL])

            # Maps direction to a numeric value
            if direction == (-1, 0):
                directions.append(0)  # up
            elif direction == (1, 0):
                directions.append(1)  # down
            elif direction == (0, -1):
                directions.append(2)  # left
            elif direction == (0, 1):
                directions.append(3)  # right

        return directions


    def return_start_coor(self):
        return self.start_coor


    def return_goal_coor(self):
        return self.goal_coor


if __name__ == '__main__':
    maze = MazeMaker(10, 10, 0.5, 10)
    print(maze.return_maze())
    # print(maze.return_optimal_route())
    # a = maze.return_optimal_route()
    # print(maze.return_directions(a))
    # print(maze.return_end_coor())
    # print(maze.return_start_coor())
