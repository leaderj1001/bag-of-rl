import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

import numpy as np
import os
from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.memory = deque(maxlen=args.buffer_limit)
        self.batch_size = 4

    def _put_data(self, transition):
        self.memory.append(transition)

    def _sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.memory[-1]]
        else:
            mini_batch = random.sample(self.memory, self.batch_size)

        states, next_states, rewards, dones, actions, probs, is_firsts = [], [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True
            for transition in seq:
                state, next_state, reward, done, action, prob = transition
                states.append(state)
                next_states.append(next_state)
                rewards.append([reward])
                done_mask = 0. if done else 1.
                dones.append([done_mask])
                actions.append([action])
                probs.append(prob.view(-1, 1))
                is_firsts.append(is_first)
                is_first = False

        return torch.tensor(states), torch.tensor(next_states), torch.tensor(rewards), torch.tensor(dones), torch.tensor(actions), torch.cat(probs), is_firsts

    def _size(self):
        return len(self.memory)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()
        self.args = args
        self.hidden = 256

        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

        self.q_value = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

    def _pi(self, x):
        out = self.fc(x)
        out = self.actor(out)
        return F.softmax(out, dim=-1)

    def _q(self, x):
        out = self.fc(x)
        out = self.q_value(out)
        return out


def _train(model, optimizer, memory, on_policy=False):
    pass


def main(args):
    env = load_env(args)
    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    memory = ReplayBuffer(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False

        while not done:
            seq_data = []
            for _ in range(args.n_rollout):
                prob = model._pi(torch.from_numpy(state).float())
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                seq_data.append((state, next_state, reward / 100., done, action, prob))
                state = next_state

                if done:
                    break
            memory._put_data(seq_data)
            if memory._size() > 500:
                _train(model, optimizer, memory, on_policy=True)
                _train(model, optimizer, memory)


if __name__ == '__main__':
    args = load_args()
    main(args)