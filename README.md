# Maze Navigation with Q-Learning

This project implements a maze navigation system using Q-learning with experience replay. It generates random mazes, trains a Q-learning model to navigate through them, and visualizes the learning process.

## Requirements

- Python 3.x
- Libraries: NumPy, TensorFlow, Keras, Pygame

## Usage

1. Clone the repository:

```bash
git clone https://github.com/oEg8/DQL-maze-solver.git
```

2. Navigate to the project directory:

```bash
cd yourdirectory
```

3. Run the main script:

```bash
python DQLsolver.py
```

## Files

- `main.py`: Main script to execute the maze navigation system.
- `DQLsolver.py`: Contains the Deep Q-Learning algorithm.
- `MazeEnv.py`: Defines the maze environment class.
- `Experience_Replay.py`: Implements experience replay for training the Q-learning model.
- `Visualiser.py`: Provides visualization capabilities for the maze navigation system.
- `MazeMaker.py`: Generates random mazes for navigation.
- `README.md`: This file, providing an overview of the project.

## How It Works

1. **Maze Generation**: The `MazeMaker` class generates random mazes with obstacles, a start point, and an end point. It ensures the existence of a valid route from the start to the end.

2. **Q-Learning**: The `QLearn` class implements Q-learning with experience replay. It learns to navigate the maze by taking actions and updating Q-values based on rewards and future states.

3. **Training**: The `main.py` script initializes the maze environment, trains the Q-learning model, and visualizes the learning process.

## Contributors

- https://github.com/oEg8

## License

This project is licensed under the MIT License