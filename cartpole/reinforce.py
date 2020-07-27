import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from collections import deque

# Todo:
# - Converge
# - Get more rewards
# - Why loss is positive num?

# - [64], 336
# - [128], 376
# - [64, 128, 256], 366
# - [128, 256], 353
# - [256, 512], 281


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()
        self.hidden = [64]
        self.args = args

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden[0]),
            nn.ReLU()
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(self.hidden[0], self.hidden[1]),
        #     nn.ReLU()
        # )

        self.out = nn.Sequential(
            nn.Linear(self.hidden[0], out_channels),
        )

        self.data = []
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def forward(self, x):
        out = self.fc1(x)
        return F.softmax(self.out(out), dim=-1)

    def _put_data(self, state, reward, prob, done):
        self.data.append([state, reward, prob, done])

    def _make_batch(self):
        states, rewards, probs, dones = [], [], [], []
        for state, reward, prob, done in self.data:
            states.append(state)
            rewards.append([reward])
            probs.append(prob.view(-1, 1))
            done_mask = 0. if done else 1.
            dones.append(done_mask)

        self.data = []
        return torch.tensor(states), torch.tensor(rewards), torch.cat(probs), torch.tensor(dones)

    def _train(self):
        states, rewards, probs, dones = self._make_batch()

        # -log(pi) * returns

        returns = torch.zeros([rewards.size(0), 1])
        R = 0.
        for t in range(rewards.size(0) - 1, -1, -1):
            R = rewards[t] + R * self.args.gamma
            returns[t] = R

        # Normalize,
        returns = (returns - returns.mean()) / returns.std()
        loss = -torch.log(probs) * returns.detach()

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # R = 0
        # losses = []
        # self.optimizer.zero_grad()
        # for state, r, prob in self.data[::-1]:
        #     R = r + R * self.args.gamma
        #     loss = -torch.log(prob) * R
        #     loss.backward()
        #     losses.append(loss.item())
        # self.optimizer.step()
        # self.data = []
        #
        # losses = np.array(losses)
        #
        # return np.mean(losses, axis=-1)
        return loss.mean().item()


def main(args):
    env = load_env(args)
    dq = deque(maxlen=100)
    dq.append(0)

    model = Model(env.observation_space.shape[0], env.action_space.n, args)
    if args.is_cuda:
        model = model.cuda()
    episodes_list = []
    losses_list = []
    rewards_list = []

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        score = 0.

        while not done:
            # env.render()
            state = torch.from_numpy(state).float()
            if args.is_cuda:
                state = state.cuda()

            prob = model(state)
            m = Categorical(prob)
            action = m.sample()

            next_state, reward, done, info = env.step(action.item())
            model._put_data(state.numpy(), reward, prob[action], done)
            score += reward

            state = next_state
        losses = model._train()
        dq.append(score)
        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{}, avg score : {}".format(episode, np.mean(dq), axis=-1))

        episodes_list.append(episode)
        losses_list.append(losses)
        rewards_list.append(score)

    if not os.path.isdir('./reinforce'):
        os.mkdir('reinforce')
    draw_plot(episodes_list, losses_list, rewards_list, path='./reinforce/loss_reward_plot.jpg')


if __name__ == '__main__':
    args = load_args()
    main(args)
