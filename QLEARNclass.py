import numpy as np
from MazeMaker import MazeMaker
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from Visualiser import Visualiser
import pickle
import os
import time

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3 

ACTION_SIZE = 4
WIN_THRESHOLD = 25


class Qmaze:
    """
    This class solves randomly generated mazes using the Deep Q-Learning algorithm.

    """
    def __init__(self, grid) -> None:
        self.grid = grid
        # self.goal = goal
        # self.start = start
        # self.position = [self.start[ROW], self.start[COL]]

        self.step_cost = -0.01
        self.illegal_cost = -0.05
        self.completion_reward = 1
        self.termination_cost = -1
        self.total_reward = 0

        self.maze_completed = False
        self.terminated = False
        self.step = 0
        self.max_steps = self.grid.size * 3

        self.exploration_rate = 0.9
        self.exploration_rate_decay = 0.995
        self.exploration_min = 0.01

        self.visualiser = Visualiser()


    def reset(self):
        self.position = [self.start[ROW], self.start[COL]]
        self.state = self.calculate_state()
        self.total_reward = 0
        self.step = 0
        self.maze_completed = False
        self.terminated = False


    def move(self, action: int) -> list:
        if action == 0:  
            self.position[ROW] -= 1
        elif action == 1:
            self.position[ROW] += 1
        elif action == 2:
            self.position[COL] -= 1
        elif action == 3:
            self.position[COL] += 1

        return self.position 


    def possible_actions(self) -> list:
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


    def calculate_state(self):
        state = self.grid.flatten().tolist()
        state.extend(self.position)
        return np.array(state)


    def act(self, action):
        possible_actions = self.possible_actions()
        if action not in possible_actions:
            reward = self.illegal_cost
            self.step += 1
            if self.step >= self.max_steps:
                reward = self.termination_cost
        else:
            self.move(action)
            self.step += 1
            if self.test_for_completion():
                reward = self.completion_reward
            elif self.test_for_termination():
                reward = self.termination_cost
            else:
                reward = self.step_cost

        state = self.calculate_state()

        return state, reward


    def get_state_size(self):
        return len(self.calculate_state())


    def get_action_size(self):
        return ACTION_SIZE


    def test_for_completion(self):
        return self.grid[self.position[ROW]][self.position[COL]]==3
    

    def test_for_termination(self):
        return self.step >= self.max_steps
    
    
    def save_model(self, model, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
            f.close()


    def load_model(self, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f) 


    def qtrain(self, model, **opt):
        n_epoch = opt.get('n_epoch', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 50)
        start_time = time.time()

        model_file = 'best_dql_solver.h5'

        if os.path.exists(model_file):
            model.load_weights(model_file)

        experience = Experience(model, max_memory=max_memory)

        win_history = list()

        for epoch in range(n_epoch):
            maze = MazeMaker(3, 3, 0.4, 4)
            self.grid = maze.return_maze()
            self.start = maze.return_start_coor()
            self.position = [start[0], start[1]]

            start_time_epoch = time.time()
            loss = 0.0
            self.reset()
            game_over = False
            epoch_cost = 0

            state = self.calculate_state()

            n_episodes = 0
            while not game_over:
                self.visualiser.draw_maze(self.grid, self.start, self.position, round(epoch_cost, 3), self.step, sum(win_history))
                possible_actions = self.possible_actions()
                prev_state = state
                if np.random.rand() < self.exploration_rate:
                    action = np.random.choice(possible_actions)
                else:
                    q_values = experience.predict(prev_state)
                    action = np.argmax(q_values[0])

                state, reward = self.act(action)
                epoch_cost += reward

                if self.test_for_completion():
                    win_history.append(1)
                    game_over = True
                elif self.test_for_termination():
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                episode = [prev_state, action, reward, state, game_over]
                experience.remember(episode)
                n_episodes += 1

                inputs, targets = experience.get_data(data_size=data_size)

                model.fit(inputs, targets, epochs=8, batch_size=16, verbose=0)

                loss = model.evaluate(inputs, targets, verbose=0)

                end_time_epoch = time.time()
                epoch_time = end_time_epoch - start_time_epoch

            # uncomment voor een lager exploration_rate per epoch.
            # self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_rate_decay)

            win_rate = sum(win_history) / len(win_history)
            end_time = time.time()
            total_time = end_time - start_time
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:.3f} | Win rate: {:.3f} | time (s): {:.3f} | total time (s): {:.3f}"
            print(template.format(epoch+1, n_epoch, loss, n_episodes, sum(win_history), win_rate, epoch_time, total_time))

            if win_rate > 0.9:
                self.exploration_rate = 0.5

            if sum(win_history[-WIN_THRESHOLD:]) == WIN_THRESHOLD or loss == 0.0:
                print(f"Reached 100% win rate at epoch: {epoch+1}")
                break
        
        model.save_weights('best_dql_solver.h5', True, 'h5')


    def build_model(self):
        model = Sequential()

        model.add(Dense(units=64, input_dim=self.grid.size+2, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.get_action_size(), activation='linear'))

        model.compile(optimizer=Adam(), loss='mse')


        # model.summary()
        return model
    

    def run(self):
        model = self.build_model()
        print(self.qtrain(model=model))


class Experience:
    def __init__(self, model, max_memory=1000, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = ACTION_SIZE


    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]


    def predict(self, state):
        state = np.expand_dims(state, axis=0)

        return self.model.predict(state, verbose=0)


    def get_data(self, data_size=10):
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        env_size = self.memory[0][0].shape[0]
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))

        for i, idx in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[idx]
            inputs[i] = envstate
            targets[i] = self.predict(envstate)
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets
  

if __name__ == '__main__':
    # grid = np.array([[2, 0, 0], 
    #                  [0, 0, 0], 
    #                  [0, 0, 3]])
    
    # start = (0, 0)
    # goal = (2, 2)

    # Qmaze(grid, start, goal).run()


    maze = MazeMaker(3, 3, 0.4, 4)
    grid = maze.return_maze()
    start = maze.return_start_coor()
    goal = maze.return_goal_coor()

    Qmaze(grid).run()


