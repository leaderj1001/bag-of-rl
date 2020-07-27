import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env

import os
import numpy as np
from collections import deque


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
            nn.Linear(self.hidden, out_channels),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.data = []

    def _pi(self, x):
        out = self.fc1(x)
        return F.softmax(self.actor(out), dim=-1)

    def _v(self, x):
        out = self.fc1(x)
        return self.critic(out)

    def _train(self):
        states, next_states, rewards, dones, actions, probs = self._make_batch()
        next_values = self._v(next_states)
        target = self._compute_returns(next_values, rewards, dones)
        values = self._v(states)
        td_error = target - values

        log_policy = torch.log(probs)
        actor_loss = -log_policy * td_error.detach()

        critic_loss = F.mse_loss(values, target.detach())

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

    def _make_batch(self):
        states, next_states, rewards, dones, actions, probs = [], [], [], [], [], []
        for state, next_state, reward, done, action, prob in self.data:
            states.append(state)
            next_states.append(next_state)
            rewards.append([reward / 100.])
            done_mask = 0. if done else 1.
            dones.append([done_mask])
            actions.append([action])
            probs.append(prob.view(-1, 1))

        self.data = []
        return torch.tensor(states, dtype=torch.float), torch.tensor(next_states, dtype=torch.float), torch.tensor(rewards, dtype=torch.float), \
               torch.tensor(dones), torch.tensor(actions), torch.cat(probs)

    def _put_data(self, state, next_state, reward, done, action, prob):
        self.data.append([state, next_state, reward, done, action, prob])

    def _compute_returns(self, next_values, rewards, dones):
        returns = torch.zeros([rewards.size(0), 1], dtype=torch.float)
        R = next_values[-1]
        for t in range(rewards.size(0) - 1, -1, -1):
            # _next_value = rewards[t] + self.args.gamma * next_values[t] * dones[t]
            R = rewards[t] + self.args.gamma * R * dones[t]
            returns[t] = R
        return returns


def main(args):
    env = load_env(args)
    in_channels, out_channels = env.observation_space.shape[0], env.action_space.n
    print(in_channels, out_channels)
    dq = deque(maxlen=100)
    dq.append(0)

    model = Model(in_channels, out_channels, args)
    if args.is_cuda:
        model = model.cuda()

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

                state = next_state
                score += 1

                if done:
                    break
            model._train()
        dq.append(score)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), end=' '))


if __name__ == '__main__':
    args = load_args()
    main(args)
