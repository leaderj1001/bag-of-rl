import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.distributions import Categorical

from environment.load_environment import load_env
from config.cartpole_config import load_args

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

        self.actor = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.data = []

        # self.apply(self._init_weights)

    def _pi(self, x):
        out = self.fc1(x)
        out = self.actor(out)

        return F.softmax(out, dim=-1)

    def _v(self, x):
        out = self.fc1(x)
        out = self.critic(out)

        return out

    def _put_data(self, state, next_state, reward, done, action, prob):
        self.data.append([state, next_state, reward, done, action, prob])

    def _train(self):
        states, next_states, rewards, dones, actions, probs = self._make_batch()

        next_values = self._v(next_states)
        values = self._v(states)

        target, advantages = self._compute_returns(values, next_values, rewards, dones)

        log_policy = torch.log(self._pi(states)[0])[actions]
        actor_loss = -log_policy * advantages.detach()
        critic_loss = F.mse_loss(values, target.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def _make_batch(self):
        states, next_states, rewards, dones, actions, probs = [], [], [], [], [], []
        for state, next_state, reward, done, action, prob in self.data:
            states.append(state)
            next_states.append(next_state)
            rewards.append([reward / 100.])
            done_mask = 0. if done else 1.
            dones.append([done_mask])
            actions.append([action])
            probs.append([prob])

        self.data = []
        return torch.tensor(states, dtype=torch.float), torch.tensor(next_states, dtype=torch.float), \
               torch.tensor(rewards, dtype=torch.float), torch.tensor(dones), torch.tensor(actions), torch.tensor(probs)

    def _compute_returns(self, values, next_values, rewards, dones):
        returns = torch.zeros([self.args.n_rollout, 1], dtype=torch.float)
        advantages = torch.zeros([self.args.n_rollout, 1], dtype=torch.float)

        _next_value = next_values[-1]
        for t in range(self.args.n_rollout - 1, -1, -1):
            returns[t] = rewards[t] + next_values[t] * self.args.gamma * dones[t]
            # _next_value = rewards[t] + _next_value * self.args.gamma * dones[t]
            # returns[t] = _next_value
            delta = returns[t] - values[t].detach()
            advantages[t] = advantages[t] * self.args.gamma * self.args.lambd * dones[t] + delta
        return returns, advantages

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)


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
            for _ in range(args.n_rollout):
                state = torch.from_numpy(state).float()

                prob = model._pi(state)

                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                model._put_data(state.numpy(), next_state, reward, done, action, prob[action])

                score += 1

                state = next_state
            model._train()
        dq.append(score)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), end=' '))

if __name__ == '__main__':
    args = load_args()
    main(args)
