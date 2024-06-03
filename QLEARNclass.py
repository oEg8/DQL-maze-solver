import numpy as np
from MazeMaker import MazeMaker
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from Visualiser import Visualiser
import tensorflow as tf
import pickle
import os
import time

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3 

"""
TODO:
- QLearn generiek maken. Maze env losmaken
- Documenteren
- Opschonen
"""


class MazeEnv:
    def __init__(self, rows, columns, seed=None) -> None:
        self.rows = rows
        self.columns = columns

        maze = MazeMaker(rows, rows, 0.4, (rows+columns)/2)
        self.grid = maze.return_maze()
        self.start = maze.return_start_coor()
        self.goal = maze.return_goal_coor()
        self.position = [self.start[0], self.start[1]]

    @classmethod
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

    @classmethod
    def possible_actions(self, grid, position) -> list:
        actions = []
        if position[ROW] - 1 >= 0 and grid[position[ROW] - 1][position[COL]] != 1:
            actions.append(UP)
        if position[ROW] + 1 < len(grid) and grid[position[ROW] + 1][position[COL]] != 1:
            actions.append(DOWN)
        if position[COL] - 1 >= 0 and grid[position[ROW]][position[COL] - 1] != 1:
            actions.append(LEFT)
        if position[COL] + 1 < len(grid[ROW]) and grid[position[ROW]][position[COL] + 1] != 1:
            actions.append(RIGHT)

        return actions


    def get_grid(self):
        return self.grid
    

    def get_position(self):
        return self.position
    

    def get_start(self):
        return self.start
    

    def get_goal(self):
        return self.goal
    

    def run(self):
        grid = self.get_grid()
        Q = QLearn(grid=grid, visualise=False)
        model = Q.build_model()
        print(Q.qtrain(model=model))
    

class QLearn:
    """
    TODO
    """
    def __init__(self, grid, visualise=False) -> None: 
        self.grid = grid
        self.visualise = visualise

        self.step_cost = -1
        self.illegal_cost = -3
        self.completion_reward = 100
        self.termination_cost = -100
        self.total_reward = 0

        self.env_completed = False
        self.env_terminated = False
        self.step = 0
        self.max_steps = 50

        self.exploration_rate = 0.1
        self.exploration_rate_decay = 0.995
        self.exploration_min = 0.01
        self.win_threshhold = 20

        if self.visualise:
            self.visualiser = Visualiser()


    def reset(self):
        self.state = self.calculate_state()
        self.total_reward = 0
        self.step = 0
        self.env_completed = False
        self.env_terminated = False


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
        return np.append(self.grid.flatten(), self.position)


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


    @classmethod
    def get_state_size(self):
        return len(self.calculate_state())


    @classmethod
    def get_action_space(self):
        return [0, 1, 2, 3]
    

    @classmethod
    def get_action_size(self):
        return len(self.get_action_space())


    def test_for_completion(self):
        return self.grid[self.position[ROW]][self.position[COL]] == 3
    

    def test_for_termination(self):
        return self.step >= self.max_steps
    

    def learn(self, model, prev_state, action, reward, state, game_over):
        prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        game_over = tf.convert_to_tensor(game_over, dtype=tf.bool)

        discount = 0.95
        target_qv = model(state)
        max_target_pv = tf.reduce_max(target_qv, axis=1)

        target = reward + (discount * max_target_pv) * tf.cast(~game_over, dtype=tf.float32)

        loss = self.run_gradient(model, prev_state, action, target)

        return loss


    def run_gradient(self, model, prev_states, actions, target):
        with tf.GradientTape(persistent=True) as tape:
            prediction = model(prev_states)
            
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            prediction_add_action = tf.gather(prediction, actions, axis=1, batch_dims=0)

            loss = tf.square((prediction_add_action - target))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss


    def qtrain(self, model, **opt):
        n_episodes = opt.get('n_episodes', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 32)
        start_time = time.time()

        model_file = 'best_dql_solver.h5'

        if os.path.exists(model_file):
            model.load_weights(model_file)

        memory = Experience_Replay(model, max_memory=max_memory)

        win_history = list()

        for episodes in range(n_episodes):
            maze = MazeEnv(3, 3)
            self.grid = maze.get_grid()
            self.start = maze.get_position()
            self.position = [self.start[0], self.start[1]]

            start_time_episode = time.time()
            loss = 0.0
            self.reset()
            game_over = False
            episode_cost = 0

            state = self.calculate_state().reshape(1, -1)

            steps = 0
            while not game_over:
                losses = []
                if self.visualise:
                    self.visualiser.draw_maze(self.grid, self.start, self.position, round(episode_cost, 3), self.step, sum(win_history))
                possible_actions = self.possible_actions()
                prev_state = state.reshape(1, -1)
                if np.random.rand() < self.exploration_rate: 
                    action = np.random.choice(self.get_action_space())  # willekeurige acties mogen wel illegaal zijn
                    # action = np.random.choice(possible_actions)       # willekeurige acties mogen niet illegaal zijn
                else:
                    q_values = model(prev_state.reshape(1, -1))
                    action = np.argmax(q_values[0])

                state, reward = self.act(action)
                episode_cost += reward
                if self.test_for_completion():
                    win_history.append(1)
                    game_over = True
                elif self.test_for_termination():
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                experience = [prev_state, action, reward, state, game_over]
                memory.remember(experience)
                steps += 1

                if steps % 4 == 0:
                    prev_states, actions, rewards, states, game_overs = memory.get_data(data_size=data_size)
                    loss = self.learn(model, prev_states[-1], actions[-1], rewards[-1], states[-1], game_overs[-1])
                    losses.append(loss)
                    mean_loss = tf.reduce_mean(losses)

                end_time_episode = time.time()
                epoch_time = end_time_episode - start_time_episode

            # uncomment voor een lager exploration_rate per epoch.
            # self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_rate_decay)

            win_rate = sum(win_history) / len(win_history)
            end_time = time.time()
            total_time = end_time - start_time
            template = "Epoch: {:05d}/{:d} | Mean loss: {:06.3f} | Steps: {:02d} | Win count: {:.2f} | Win rate: {:.2f} | time (s): {:.1f} | total time (s): {:.1f}"
            print(template.format(episodes+1, n_episodes, mean_loss, steps, sum(win_history), win_rate, epoch_time, total_time))

            if win_rate > 0.9:
                self.exploration_rate = 0.5

            if sum(win_history[-self.win_threshhold:]) == self.win_threshhold:
                print(f"Reached sufficient win rate at epoch: {episodes+1}")
                break
        
        model.save_weights('best_dql_solver.h5', True, 'h5')


    def build_model(self):
        model = Sequential()

        model.add(Dense(units=64, input_dim=11, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.get_action_size(), activation='linear'))

        model.compile(optimizer=Adam())

        # model.summary()
        return model
    

    def run(self):
        model = self.build_model()
        print(self.qtrain(model=model))


class Experience_Replay:
    def __init__(self, model, max_memory=1000, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = QLearn.get_action_size()


    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)


    def get_data(self, data_size=10):
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)

        # Determine the shapes of states
        state_shape = self.memory[0][0].shape

        # Initialize numpy arrays to store batches of data
        prev_states = np.zeros((data_size,) + state_shape)
        actions = np.zeros((data_size,), dtype=int)
        rewards = np.zeros((data_size,))
        states = np.zeros((data_size,) + state_shape)
        game_overs = np.zeros((data_size,), dtype=bool)

        indices = np.random.choice(range(mem_size), data_size, replace=False)
        for i, idx in enumerate(indices):
            prev_state, action, reward, state, game_over = self.memory[idx]
            prev_states[i] = prev_state
            actions[i] = action
            rewards[i] = reward
            states[i] = state
            game_overs[i] = game_over

        return prev_states, actions, rewards, states, game_overs
  

if __name__ == '__main__':
    maze = MazeMaker(3, 3, 0.4, 4)
    grid = maze.return_maze()
    start = maze.return_start_coor()
    goal = maze.return_goal_coor()

    QLearn(grid=grid, visualise=False).run()

    print('testtt')

# if __name__ == '__main__':
#     m = MazeEnv(3, 3)
#     print(m.run())


