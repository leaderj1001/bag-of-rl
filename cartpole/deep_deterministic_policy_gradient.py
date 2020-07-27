import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

from collections import deque
import random
import numpy as np
import os


class ReplayBuffer(object):
    def __init__(self, args):
        self.memory = deque(maxlen=args.buffer_limit)

    def _put_data(self, transition):
        self.memory.append(transition)

    def _sample(self, n):
        mini_batch = random.sample(self.memory, n)
        states, next_states, rewards, dones, actions = [], [], [], [], []

        for transition in mini_batch:
            state, next_state, reward, done, action = transition
            states.append(state)
            next_states.append(next_state)
            rewards.append([reward])
            dones.append([done])
            actions.append([action])

        return torch.tensor(states, dtype=torch.float), torch.tensor(next_states, dtype=torch.float), torch.tensor(rewards), \
               torch.tensor(dones), torch.tensor(actions)

    def size(self):
        return len(self.memory)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class MuNet(nn.Module):
    def __init__(self, in_channels, args):
        super(MuNet, self).__init__()
        self.hidden = 128

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 1)
        )

    def forward(self, x):
        mu = torch.tanh(self.fc1(x)) * 2
        return mu


class QNet(nn.Module):
    def __init__(self, in_channels, args):
        super(QNet, self).__init__()
        self.hidden = 64

        self.fc_s = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU()
        )

        self.fc_a = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.ReLU()
        )

        self.fc_q = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 1),
        )

    def forward(self, x, a):
        s = self.fc_s(x)
        a = self.fc_a(a)

        out = torch.cat([s, a], dim=1)
        out = self.fc_q(out)

        return out


def _train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, args):
    states, next_states, rewards, dones, actions = memory._sample(args.batch_size)

    target = rewards + args.gamma * q_target(next_states, mu_target(next_states))
    q_loss = F.smooth_l1_loss(q(states, actions), target.detach())

    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(states, mu(states)).mean()

    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    tau = 0.005
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target * (1.0 - tau) + param.data * tau)


def main(args):
    lr_mu = 0.0005
    lr_q = 0.001
    gamma = 0.99
    batch_size = 32
    buffer_limit = 50000
    tau = 0.005  # for target network soft update

    args.env = 'Pendulum-v0'
    env = load_env(args)
    in_channels = env.observation_space.shape[0]
    memory = ReplayBuffer(args)

    q = QNet(in_channels, args)
    mu = MuNet(in_channels, args)

    q_target = QNet(in_channels, args)
    mu_target = MuNet(in_channels, args)

    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())

    q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    ou_noise = OrnsteinUhlenbeckNoise(np.zeros(1))

    score = 0.
    for episode in range(1, args.episodes):
        state = env.reset()

        for t in range(300):
            state = torch.from_numpy(state).float()
            action = mu(state)
            action = action.item() + ou_noise()[0]

            next_state, reward, done, info = env.step([action])
            memory._put_data((state.numpy(), next_state, reward / 100., done, action))
            score += reward
            state = next_state

            if done:
                break

        if memory.size() > 2000:
            for i in range(10):
                _train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, args)
                soft_update(mu, mu_target)
                soft_update(q, q_target)

        if episode % 20 == 0 and episode != 0:
            print("# of episode :{}, avg score : {:.1f}".format(episode, score / 20))
            score = 0.0


if __name__ == '__main__':
    args = load_args()
    main(args)
