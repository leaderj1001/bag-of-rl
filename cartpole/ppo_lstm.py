import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

import numpy as np
from collections import deque


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()

        self.args = args
        self.hidden_size = 64
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.hidden_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2)

        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size // 2, out_channels)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.data = []

    def _pi(self, x, hidden):
        out = self.fc(x).view(-1, 1, self.hidden_size)
        out, hidden = self.lstm(out, hidden)
        out = self.actor(out)
        return F.softmax(out.view(-1, self.out_channels), dim=-1), hidden

    def _v(self, x, hidden):
        out = self.fc(x).view(-1, 1, self.hidden_size)
        out, hidden = self.lstm(out, hidden)
        return self.critic(out).view(-1, 1)

    def _put_data(self, state, next_state, reward, done, action, prob, hidden_in, hidden_out):
        self.data.append([state, next_state, reward, done, action, prob, hidden_in, hidden_out])

    def _make_batch(self):
        states, next_states, rewards, dones, actions, probs, hiddens_in, hiddens_out = [], [], [], [], [], [], [], []

        for state, next_state, reward, done, action, prob, hidden_in, hidden_out in self.data:
            states.append(state)
            next_states.append(next_state)
            rewards.append([reward / 100.])
            done_mask = 0. if done else 1.
            dones.append([done_mask])
            actions.append([action])
            probs.append([prob])
            hiddens_in.append(hidden_in)
            hiddens_out.append(hidden_out)

        self.data = []
        return torch.tensor(states).float(), torch.tensor(next_states).float(), torch.tensor(rewards), torch.tensor(dones), torch.tensor(actions), \
               torch.tensor(probs), hiddens_in[0], hiddens_out[0]

    def _train(self):
        states, next_states, rewards, dones, actions, probs, hiddens_in, hiddens_out = self._make_batch()
        hiddens_in = (hiddens_in[0].detach(), hiddens_in[1].detach())
        hiddens_out = (hiddens_out[0].detach(), hiddens_out[1].detach())

        for _ in range(self.args.train_epochs):
            next_values = self._v(next_states, hiddens_out)
            values = self._v(states, hiddens_in)

            # returns = self._compute_returns(next_values, rewards, dones)
            # advantage = returns - values

            td_target, advantage = self._compute_gae(values, next_values, rewards, dones)

            # td_target = rewards + self.args.gamma * self._v(next_states, hiddens_out) * dones
            # td_error = td_target - self._v(states, hiddens_in)

            new_probs = self._pi(states, hiddens_in)[0].gather(1, actions)
            ratio = torch.exp(torch.log(new_probs) - torch.log(probs.detach()))

            loss1 = ratio * advantage
            loss2 = torch.clamp(ratio, min=1-self.args.epsilon, max=1+self.args.epsilon) * advantage

            actor_loss = -torch.min(loss1, loss2)
            critic_loss = F.smooth_l1_loss(self._v(states, hiddens_out), td_target.detach())

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def _compute_returns(self, next_values, rewards, dones):
        returns = torch.zeros([dones.size(0), 1], dtype=torch.float)

        R = next_values[-1]
        for t in range(dones.size(0) - 1, -1, -1):
            R = rewards[t] + self.args.gamma * R * dones[t]
            returns[t] = R

        return returns

    def _compute_gae(self, values, next_values, rewards, dones):
        td_target = rewards + self.args.gamma * next_values * dones
        delta = (td_target - values).detach()
        advantage = torch.zeros([dones.size(0), 1], dtype=torch.float)

        d = 0.
        for t in range(dones.size(0) - 1, -1, -1):
            d = self.args.gamma * self.args.lmbda * d + delta[t]
            advantage[t] = d

        return td_target, advantage


def main(args):
    env = load_env(args)
    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    dq = deque(maxlen=100)
    dq.append(0)

    for episode in range(1, args.episodes + 1):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        state = env.reset()
        done = False
        score = 0.

        while not done:
            for _ in range(args.n_rollout):
                h_in = h_out
                prob, hidden_out = model._pi(torch.from_numpy(state).float(), h_in)
                prob = prob.squeeze()
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                model._put_data(state, next_state, reward, done, action, prob[action], hidden_in=h_in, hidden_out=hidden_out)
                state = next_state
                score += reward
                if done:
                    break
            model._train()
        dq.append(score)

        if episode % args.print_intervals == 0:
            print('Episode: {}, Score mean: {}'.format(episode, np.mean(dq)))


if __name__ == '__main__':
    args = load_args()
    main(args)
