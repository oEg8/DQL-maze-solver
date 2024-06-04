import numpy as np


class ExperienceReplay:
    def __init__(self, model, max_memory: int = 1000, discount: float = 0.95) -> None:
        """
        Initializes the ExperienceReplay object.

        Parameters:
            model (Any): The model used for Q-learning.
            max_memory (int): Maximum number of experiences to store in memory. Default is 1000.
            discount (float): Discount factor for future rewards. Default is 0.95.
        """
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()


    def remember(self, episode: tuple[np.ndarray, int, float, np.ndarray, bool]):
        """
        Appends the given episode to the memory and deletes the oldest entry when maximum length is reached.

        Parameters:
            episode (Tuple[np.ndarray, int, float, np.ndarray, bool]): A tuple containing 
                (previous state, action, reward, next state, game over flag).
        """
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)


    def get_data(self, data_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a sample of the memory for the Q-learning algorithm to learn on.

        Parameters:
            data_size (int): Number of experiences to return. Default is 32.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                - prev_states: Batch of previous states.
                - actions: Batch of actions taken.
                - rewards: Batch of rewards received.
                - states: Batch of next states.
                - game_overs: Batch of game over flags.
        """
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)

        # Determine the shape of states
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