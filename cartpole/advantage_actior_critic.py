import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch.optim as optim

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

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
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            nn.Linear(self.hidden, out_channels),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden, 1),
        )

        self.data = []
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

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

    def _compute_returns(self, next_states, rewards, dones):
        returns = torch.zeros([rewards.size(0), 1])
        next_values = self._v(next_states)

        R = next_values[-1]
        for t in range(rewards.size(0) - 1, -1, -1):
            R = rewards[t] + self.args.gamma * R * dones[t]
            returns[t] = R

        return returns

    def _train(self):
        states, next_states, rewards, dones, actions, probs = self._make_batch()

        value = self._v(states)

        target = self._compute_returns(next_states, rewards, dones)
        td_error = target - value

        log_policy = torch.log(probs)
        actor_loss = -log_policy * td_error.detach()

        critic_loss = F.smooth_l1_loss(value, target.detach())

        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

    def _make_batch(self):
        states, next_states, rewards, dones, actions, probs = [], [], [], [], [], []
        for state, next_state, reward, done, action, prob in self.data:
            states.append(state.numpy())
            next_states.append(next_state)
            rewards.append([reward / 100.])
            mask_done = 0. if done else 1.
            dones.append([mask_done])
            actions.append(action)
            probs.append(prob.view(-1, 1))

        self.data = []
        return torch.tensor(states).float(), torch.tensor(next_states).float(), torch.tensor(rewards).float(), torch.tensor(dones).float(), torch.tensor(actions), torch.cat(probs)


def main(args):
    dq = deque(maxlen=100)
    dq.append(0)

    env = load_env(args)
    in_channels, out_channels = env.observation_space.shape[0], env.action_space.n
    model = Model(in_channels, out_channels, args)
    if args.is_cuda:
        model = model.cuda()

    rewards_list = []
    episodes_list = []
    losses_list = []
    for episode in range(args.episodes):
        score = 0.
        state = env.reset()
        done = False

        while not done:
            for _ in range(args.n_rollout):
                state = torch.from_numpy(state).float()
                if args.is_cuda:
                    state = state.cuda()

                prob = model._pi(state)
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                model._put_data(state, next_state, reward, done, action, prob[action])
                score += reward

                state = next_state

                if done:
                    break
            loss = model._train()
            losses_list.append(loss)
        dq.append(score)
        episodes_list.append(episode)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), end=' '))
            print(np.mean(np.array(losses_list)))


if __name__ == '__main__':
    args = load_args()
    main(args)
