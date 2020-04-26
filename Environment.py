import numpy as np


class Environment():
    WIDTH = 600 // 2
    HEIGHT = 336 // 2

    def __init__(self, scrcpy):
        self.state = None
        self.done = None
        self.scrcpy = scrcpy
        self.reward = None
        self.action_space = {
            0: [640, 720, 640, 720],  # stop
            1: [640, 720, 640, 220],  # up
            2: [640, 720, 640, 1220],  # down
            3: [640, 720, 140, 720],  # left
            4: [640, 720, 1140, 720],  # right
            5: [640, 720, 1140, 220],  # right up
            6: [640, 720, 1140, 1220],  # right down
            7: [640, 720, 140, 220],  # left up
            8: [640, 720, 140, 1220],  #left down
            9: [2200, 1000]  # shot
        }

    def send_action(self, index):
        print("send action: {}".format(index))
        self.scrcpy.send_action(self.action_space[index])

    def reset(self):
        print("reset")
        self.state = np.random.uniform(low=-0.05,
                                       high=0.05,
                                       size=(Environment.WIDTH, Environment.HEIGHT, 3))
        self.done = False
        self.reward = 0.0
        return np.array(self.state)

    def step(self, action_index):
        self.send_action(action_index)
        self.state = self.preprocessing(self.scrcpy.get_next_frame(most_recent=True))  # (336,600,3)
        self.reward = self.calculate_reward()
        self.done = self.check_done()
        return self.state, self.reward, self.done

    def preprocessing(self, X):
        X = X / 255 - 0.5
        return X

    def calculate_reward(self):
        # todo reward policy
        return 0.01

    def check_done(self):
        # todo done check
        return False
