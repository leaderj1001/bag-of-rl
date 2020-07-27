import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from environment.load_environment import load_env
from config.cartpole_config import load_args
from draw_plot.draw_plot import draw_plot

import os
from collections import deque
import numpy as np


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()
        self.hidden = 128
        self.args = args

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

    def _pi(self, x):
        out = self.fc1(x)
        out = self.actor(out)
        return F.softmax(out, dim=-1)

    def _v(self, x):
        out = self.fc1(x)
        return self.critic(out)

    def _train(self):
        states, next_states, rewards, dones, actions, probs = self._make_batch()

        td_target = rewards + self.args.gamma * self._v(next_states) * dones
        values = self._v(states)
        delta = td_target - values

        log_probs = torch.log(probs)
        actor_loss = -log_probs * delta.detach()

        critic_loss = F.mse_loss(values, td_target.detach())

        # entropy = -pi * log(pi)
        entropies = (-probs * torch.log(probs)).sum(dim=-1)
        # entropies = m.entropy()

        loss = actor_loss + critic_loss - 0.1 * entropies.detach()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def _put_data(self, state, next_state, reward, done, action, prob):
        self.data.append([state, next_state, reward, done, action, prob])

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
        return torch.tensor(states, dtype=torch.float), torch.tensor(next_states, dtype=torch.float), \
               torch.tensor(rewards, dtype=torch.float), torch.tensor(dones), torch.tensor(actions), torch.cat(probs)


def main(args):
    env = load_env(args)
    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    dq = deque(maxlen=100)
    dq.append(0)

    episodes_list = []
    rewards_list = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        score = 0.

        while not done:
            for _ in range(1):
                state = torch.from_numpy(state).float()

                prob = model._pi(state)
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                score += 1.
                model._put_data(state.numpy(), next_state, reward, done, action, prob[action])
                state = next_state
            model._train()
        dq.append(score)
        episodes_list.append(episode)
        rewards_list.append(score)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), end=' '))

        if not os.path.isdir('entropy'):
            os.mkdir('entropy')

        draw_plot(episodes_list, rewards_list, rewards_list, path='entropy/loss_reward_plot.jpg')


if __name__ == '__main__':
    args = load_args()
    main(args)
