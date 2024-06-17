from MazeEnv import MazeEnv
from DQLsolver import QLearn

env = MazeEnv(3, 3)
print(QLearn(env=env, visualise=False).run(env=env))