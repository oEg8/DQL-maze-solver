import numpy as np
from Experience_Replay import ExperienceReplay
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from Visualiser import Visualiser
import tensorflow as tf
from MazeEnv import MazeEnv
import os
import time

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3 

MOVE_PLAYER = 0
MOVE_GOAL = 1
    

class QLearn:
    """
    This class implements Q-learning with experience replay for a maze environment.
    """

    def __init__(self, env: MazeEnv, visualise: bool = False) -> None: 
        """
        Initializes the QLearn object.

        Parameters:
            env (MazeEnv): The maze environment.
            visualise (bool): Whether to visualize the maze. Default is False.
        """
        self.env = env
        self.visualise = visualise
        self.model = self.build_model(env)

        self.step_cost = -1
        self.illegal_cost = -5
        self.novelty_weight = 1.0

        self.max_steps = self.env.get_state_size()*2
        self.exploration_rate = 0.1
        self.win_threshhold = 20
        self.discount_factor = 0.95
        
        self.action_type_switch = 5

        if self.visualise:
            self.visualiser = Visualiser(fps=2)


    def calculate_state(self) -> np.ndarray:
        """
        Calculates the state, which is the flattened grid and the current position of the agent.

        Returns:
            np.ndarray: The current state.
        """
        if self.step % self.action_type_switch == 0:
            return np.append(self.grid.flatten(), [self.position[0], self.position[1], 1])  # move the goal
        else:
            return np.append(self.grid.flatten(), [self.position[0], self.position[1], 0])  # move the player



    def act(self, action: int, action_type: int) -> tuple[np.ndarray, int]:
        """
        Executes the given action in the environment.

        Parameters:
            action (int): The action to be taken.

        Returns:
            tuple: The new state and the reward.
        """
        possible_actions = self.env.possible_actions(action_type)
        if action not in possible_actions:
            self.step += 1
            reward = self.illegal_cost
        else:
            self.step += 1
            reward = self.step_cost
            self.env.move(action, action_type)

        state = self.calculate_state()

        # novelty score calculation
        state_tuple = tuple(state)
        self.state_visit_count[state_tuple] = self.state_visit_count.get(state_tuple, 0) + 1
        novelty_reward = self.novelty_weight / np.sqrt(self.state_visit_count[state_tuple])
        reward += novelty_reward

        return state, reward


    @classmethod
    def get_action_space(cls, action_type: int) -> list[int]:
        """
        Returns the action space.

        Returns:
            list: List of possible actions.
        """
        if action_type == 0:
            return [0, 1, 2, 3]  # action_type 1
        elif action_type == 1:
            return [4, 5, 6, 7]  # action_type 2
    

    @classmethod
    def get_action_size(cls) -> int:
        """
        Returns the size of the action space.

        Returns:
            int: Size of the action space.
        """
        return 8


    def test_for_completion(self) -> bool:
        """
        Checks if the agent has reached the goal.

        Returns:
            bool: True if the agent has reached the goal, False otherwise.
        """
        return self.grid[self.position[ROW]][self.position[COL]] == 3
    

    def test_for_termination(self) -> bool:
        """
        Checks if the maximum number of steps has been taken.

        Returns:
            bool: True if the maximum steps have been taken, False otherwise.
        """
        return self.step >= self.max_steps
    

    def select_action(self, state: np.ndarray, model: Sequential, epsilon: float, action_type: int) -> int:
        if np.random.rand() < epsilon:
            return np.random.choice(self.get_action_space(action_type))
        else:
            state = state.reshape(1, -1)
            q_values = model(state)
            q_values = tf.squeeze(q_values)
            if action_type == 0:
                return np.argmax(q_values[:4])  # Player actions
            else:
                return np.argmax(q_values[4:]) + 4  # Goal actions
    

    def learn(self, prev_state: np.ndarray, action: int, reward: int, state: np.ndarray, game_over: bool) -> float:
        """
        Trains the model using the provided experience.

        Parameters:
            model (Sequential): The neural network model.
            prev_state (np.ndarray): The previous state.
            action (int): The action taken.
            reward (int): The reward received.
            state (np.ndarray): The current state.
            game_over (bool): Whether the game is over.

        Returns:
            float: The loss from training.
        """
        prev_state = tf.convert_to_tensor(prev_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        game_over = tf.convert_to_tensor(game_over, dtype=tf.float32)
        discount_factor = tf.convert_to_tensor(self.discount_factor, dtype=tf.float32)

        target = self.calculate_target(state, reward, discount_factor, game_over)

        loss = self.run_gradient(prev_state, action, target)

        return loss
    

    @tf.function
    def calculate_target(self, state: tf.Tensor, reward: tf.Tensor, discount: float, game_over: tf.Tensor) -> tf.Tensor:
        """
        Calculates the target value for training.

        Parameters:
            state (tf.Tensor): The current state.
            reward (tf.Tensor): The reward received.
            discount (float): The discount factor.
            game_over (tf.Tensor): Whether the game is over.
            model (Sequential): The neural network model.

        Returns:
            tf.Tensor: The target value.
        """
        target_qv = self.model(state)
        max_target_pv = tf.reduce_max(target_qv, axis=1)
        target = reward + (discount * max_target_pv) * game_over

        return target


    @tf.function
    def run_gradient(self, prev_states: tf.Tensor, actions: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        """
        Runs the gradient descent step for training.

        Parameters:
            model (Sequential): The neural network model.
            prev_states (tf.Tensor): The previous states.
            actions (tf.Tensor): The actions taken.
            target (tf.Tensor): The target values.

        Returns:
            tf.Tensor: The loss from training.
        """
        with tf.GradientTape(persistent=True) as tape:
            prediction = self.model(prev_states)
            
            prediction_add_action = tf.gather(prediction, actions, axis=1, batch_dims=0)

            loss = tf.square((prediction_add_action - target))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss


    def qtrain(self, model: Sequential, **opt) -> None:
        """
        Trains the model using Q-learning with experience replay.

        Parameters:
            model (Sequential): The neural network model.
            **opt: Optional parameters for training.
        """
        n_episodes = opt.get('n_episodes', 15000)
        max_memory = opt.get('max_memory', 15000)
        data_size = opt.get('data_size', 32)
        save_grids = opt.get('save_grids', False)
        start_time = time.time()

        model_file = 'best_dql_solver.h5'

        # NOTE: When changing the grid size, delete the old model file. Otherwise input shape mismatches will occur.
        if os.path.exists(model_file):
            model.load_weights(model_file)

        memory = ExperienceReplay(model, max_memory=max_memory)

        win_history = list()

        for episode in range(n_episodes):

            start_time_episode = time.time()
            mean_loss = 0.0
            self.grid, self.position, self.total_reward, self.step = self.env.reset()
            game_over = False
            episode_cost = 0
            self.state_visit_count = {}

            if save_grids:
                if os.path.exists('grids'):
                    np.save(f'grid{episode}.npy', self.grid)
                else:
                    os.mkdir('grids')
                    np.save(f'grids/grid{episode}.npy', self.grid)

            state = self.calculate_state()

            steps = 0
            while not game_over:
                losses = []
                if self.visualise:
                    self.visualiser.draw_maze(self.grid, self.position, round(episode_cost, 3), self.step, sum(win_history))

                action_type = 1 if self.step % self.action_type_switch != 0 else 0
                action = self.select_action(state, model, self.exploration_rate, action_type)

                next_state, reward = self.act(action, action_type)

                episode_cost += reward
                if self.test_for_completion():
                    win_history.append(1)
                    game_over = True
                elif self.test_for_termination():
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False
                
                experience = [state, action, reward, next_state, game_over]
                memory.remember(experience)

                if steps % 4 == 0:
                    prev_states, actions, rewards, states, game_overs = memory.get_data(data_size=data_size)
                    loss = self.learn(prev_states, actions, rewards, states, game_overs)
                    losses.append(loss)
                    mean_loss = tf.reduce_mean(losses)

                steps += 1
                state = next_state
                end_time_episode = time.time()
                epoch_time = end_time_episode - start_time_episode

            win_rate = sum(win_history[-100:]) / len(win_history[-100:])
            end_time = time.time()
            total_time = end_time - start_time
            template = "Epoch: {:05d}/{:d} | Mean loss: {:07.3f} | Steps: {:02d} | Win count: {:.2f} | Win rate [-100:]: {:.2f} | time (s): {:.2f} | total time (s): {:.2f}"
            print(template.format(episode+1, n_episodes, mean_loss, steps, sum(win_history), win_rate, epoch_time, total_time))

            if win_rate > 0.9:
                self.exploration_rate = 0.5

            if sum(win_history[-self.win_threshhold:]) == self.win_threshhold:
                print(f"Reached sufficient win rate at epoch: {episode+1}")
                break
        print(f'Overall win rate: {round(sum(win_history) / len(win_history), 3)}')
        model.save_weights('best_dql_solver.h5', True, 'h5')


    def build_model(self, env: MazeEnv) -> Sequential:
        """
        Builds the neural network model.

        Parameters:
            env (MazeEnv): The maze environment.

        Returns:
            Sequential: The neural network model.
        """
        model = Sequential()
        model.add(tf.keras.Input(shape=(env.get_state_size()),))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=self.get_action_size(), activation='softmax'))

        model.compile(optimizer=Adam())

        return model
    

    def run(self, env: MazeEnv) -> None:
        """
        Builds the model and starts the training process.

        Parameters:
            env (MazeEnv): The maze environment.
        """
        model = self.build_model(env)
        print(self.qtrain(model=model, n_episodes=1000))
  

if __name__ == '__main__':
    env = MazeEnv(5, 5)
    QLearn(env=env, visualise=False).run(env=env)


