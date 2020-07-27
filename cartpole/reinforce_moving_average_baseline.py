import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env

import os
from collections import deque
import numpy as np


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()

        self.args = args
        self.hidden = 256

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.data = []
        self.baseline = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)

        return F.softmax(out, dim=-1)

    def _put_data(self, reward, prob):
        self.data.append([reward, prob])

    def _make_batch(self):
        rewards, probs = [], []

        for reward, prob in self.data:
            rewards.append([reward])
            probs.append(prob.view(-1, 1))

        self.data = []
        return torch.tensor(rewards), torch.cat(probs)

    def _train(self):
        rewards, probs = self._make_batch()
        returns = torch.zeros([rewards.size(0), 1])

        # -log(pi) * (Returns - baseline)

        R = 0.
        for t in range(rewards.size(0) - 1, -1, -1):
            R = rewards[t] + R * self.args.gamma
            returns[t] = R

        if self.baseline is None:
            self.baseline = returns
        else:
            self.baseline = 0.99 * returns.mean() + (1 - 0.99) * self.baseline.mean()

        advantage = returns - self.baseline
        loss = -torch.log(probs) * advantage.detach()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main(args):
    env = load_env(args)
    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    dq = deque(maxlen=100)
    dq.append(0)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        score = 0.

        while not done:
            state = torch.from_numpy(state).float()

            prob = model(state)
            m = Categorical(prob)
            action = m.sample().item()

            next_state, reward, done, _ = env.step(action)
            score += 1.

            model._put_data(reward, prob[action])
            state = next_state
        model._train()
        dq.append(score)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), axis=-1))


if __name__ == '__main__':
    args = load_args()
    main(args)
