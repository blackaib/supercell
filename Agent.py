from ReplayMemory import ReplayMemory
from Environment import Environment


class Agent():
    def __init__(self):
        self.replaymemory = ReplayMemory()
        self.env = Environment()

    def reset(self):
        pass

    def step(self, action_index):
        pass

    def train(self):
        pass

# https://github.com/Allong12/py-scrcpy