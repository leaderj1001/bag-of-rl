import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

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

    def _put_data(self, state, next_state, reward, done, action, prob):
        self.data.append([state, next_state, reward, done, action, prob])

    def _train(self):
        states, next_states, rewards, dones, actions, old_probs = self._make_batch()

        losses = 0.
        for i in range(self.args.train_epochs):
            values = self._v(states)
            if self.args.method == 'returns':
                td_target = self._compute_returns(next_states, rewards, dones)
                advantage = td_target - values
            else:
                td_target = rewards + self.args.gamma * self._v(next_states) * dones
                delta = td_target - values
                delta = delta.detach().numpy()

                advantage_list = []
                advantage = 0.
                for delta_t in delta[::-1]:
                    advantage = delta_t[0] + self.args.gamma * self.args.lmbda * advantage
                    advantage_list.append([advantage])
                advantage_list.reverse()
                advantage = torch.tensor(advantage_list, dtype=torch.float)

            new_probs = self._pi(states).gather(1, actions)
            ratio = torch.exp(torch.log(new_probs) - torch.log(old_probs))

            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, min=1-self.args.epsilon, max=1+self.args.epsilon) * advantage.detach()
            loss = -torch.min(surr1, surr2) + F.mse_loss(values, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            losses += loss.mean()

        return losses / self.args.train_epochs

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
               torch.tensor(rewards), torch.tensor(dones), \
               torch.tensor(actions), torch.tensor(probs)

    def _compute_returns(self, next_states, rewards, dones):
        returns = torch.zeros([next_states.size(0), 1])
        next_values = self._v(next_states)

        R = next_values[-1]
        for t in range(next_values.size(0) - 1, -1, -1):
            R = rewards[t] + self.args.gamma * R * dones[t]
            returns[t] = R

        return returns


def main(args):
    env = load_env(args)
    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    dq = deque(maxlen=100)
    dq.append(0)

    episodes_list = []
    losses_list = []
    rewards_list = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        score = 0.
        losses = 0.

        while not done:
            for _ in range(args.n_rollout):
                state = torch.from_numpy(state).float()

                prob = model._pi(state)
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                score += 1.
                model._put_data(state.numpy(), next_state, reward, done, action, prob[action])
                state = next_state

                if done:
                    break
            losses += model._train()
        dq.append(score)

        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), end=' '))

        episodes_list.append(episode)
        losses_list.append(losses)
        rewards_list.append(score)

        if not os.path.isdir('./ppo'):
            os.mkdir('ppo')
        draw_plot(episodes_list, losses_list, rewards_list, path='./ppo/loss_reward_plot.jpg')


if __name__ == '__main__':
    args = load_args()
    main(args)
