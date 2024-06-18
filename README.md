# Maze Navigation with Q-Learning

This project implements a maze navigation system using Q-learning with experience replay. It generates random mazes, trains a Q-learning model to navigate through them, and visualizes the learning process.

## Contents

- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
- [Files](#files)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

## Description

TODO

## Requirements

- Python 3.x
- NumPy
- TensorFlow
- Keras
- Pygame

You can install the dependencies using pip:

```
pip install numpy, tensorflow, keras, pygame
```

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
python main.py
```

## Files

- `main.py`: Main script to execute the maze navigation system.
- `DQLsolver.py`: Contains the Deep Q-Learning algorithm.
- `MazeEnv.py`: Defines the maze environment class.
- `Experience_Replay.py`: Implements experience replay for training the Q-learning model.
- `Visualiser.py`: Provides visualization capabilities for the maze navigation system.
- `MazeMaker.py`: Generates random mazes for navigation.
- `README.md`: This file, providing an overview of the project.

## Contributors

- https://github.com/oEg8

## License

This project is licensed under the MIT License

## Refrences

- Pygame Documentation: [Pygame](https://www.pygame.org/docs/)
- Tensorflow Documentation: [Tensorflow](https://www.tensorflow.org/api_docs)