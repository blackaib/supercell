from Environment import Environment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from SCRCPY_client import SCRCPY_client
import sys
import os
import subprocess
import time
import logging
import gym

SVR_maxSize = 600
SVR_bitRate = 8000000
SVR_tunnelForward = "true"
SVR_crop = "9999:9999:0:0"
SVR_sendFrameMeta = "true"

IP = 'localhost'
PORT = 8080
RECVSIZE = 0x10000
HEADER_SIZE = 12

cwd = os.getcwd()
SCRCPY_dir = '/snap/scrcpy/211/usr/bin/'
SCRCPY_SERVER_dir = '/home/cogitans/snap/scrcpy/common/scrcpy-server.jar'
FFMPEG_bin = 'ffmpeg'
ADB_bin = os.path.join(SCRCPY_dir, "adb")
#fd = open("savesocksession",'wb')

logger = logging.getLogger(__name__)

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 1 # 3
T_horizon = 1 # 20


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(Environment.WIDTH*Environment.HEIGHT*3, 256)
        self.fc_pi = nn.Linear(256, 10)  # direction(8), shot, stop
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = x.view(-1, Environment.WIDTH*Environment.HEIGHT*3)
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = x.view(-1, Environment.WIDTH * Environment.HEIGHT * 3)
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        print("save data")
        self.data.append(transition)

    def make_batch(self):
        print("gathering experiences")
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
            print(loss)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main(scrcpy):
    env = Environment(scrcpy)
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                action_index = m.sample().item()
                s_prime, r, done = env.step(action_index)

                model.put_data((s, action_index, r / 100.0, s_prime, prob[0][action_index].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


def connect_and_forward_scrcpy():
    try:
        logger.info("Upload JAR...")
        adb_push = subprocess.Popen(
            [ADB_bin, 'push', SCRCPY_SERVER_dir, '/data/local/tmp/'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd)
        adb_push_comm = ''.join([x.decode("utf-8") for x in adb_push.communicate() if x is not None])

        if "error" in adb_push_comm:
            logger.critical("Is your device/emulator visible to ADB?")
            raise Exception(adb_push_comm)
        '''
        ADB Shell is Blocking, don't wait up for it 
        Args for the server are as follows:
        maxSize         (integer, multiple of 8) 0
        bitRate         (integer)
        tunnelForward   (optional, bool) use "adb forward" instead of "adb tunnel"
        crop            (optional, string) "width:height:x:y"
        sendFrameMeta   (optional, bool) 

        '''
        logger.info("Run JAR")
        subprocess.Popen(
            [ADB_bin, 'shell',
             'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
             'app_process', '/', 'com.genymobile.scrcpy.Server',
             str(SVR_maxSize), str(SVR_bitRate),
             SVR_tunnelForward, SVR_crop, SVR_sendFrameMeta],
            cwd=cwd)
        time.sleep(0.05)

        logger.info("Forward Port")
        subprocess.Popen(
            [ADB_bin, 'forward',
             'tcp:8080', 'localabstract:scrcpy'],
            cwd=cwd).wait()
        time.sleep(0.05)
    except FileNotFoundError:
        raise FileNotFoundError("Couldn't find ADB at path ADB_bin: " +
                                str(ADB_bin))
    return True


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    assert connect_and_forward_scrcpy()

    SCRCPY = SCRCPY_client()
    SCRCPY.connect
    SCRCPY.start_processing()
    main(SCRCPY)

# https://github.com/Allong12/py-scrcpy
# https://pypi.org/project/pure-python-adb/