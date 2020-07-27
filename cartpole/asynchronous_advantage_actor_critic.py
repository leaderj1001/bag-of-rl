import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

import numpy as np
import os
import time
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

    def _train(self, optimizer):
        states, next_states, rewards, dones, action, probs = self._make_batch()

        td_target = rewards + self.args.gamma * self._v(next_states) * dones
        values = self._v(states)
        td_error = td_target - values

        actor_loss = -torch.log(probs) * td_error.detach()
        critic_loss = F.smooth_l1_loss(values, td_target.detach())

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

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
                torch.tensor(rewards), torch.tensor(dones), torch.tensor(actions), torch.cat(probs)


def _train(global_model, rank, args):
    env = load_env(args)
    local_model = Model(env.observation_space.shape[0], env.action_space.n, args)

    local_model.load_state_dict(global_model.state_dict())
    optimizer = optim.Adam(global_model.parameters(), lr=args.lr)
    dq = deque(maxlen=100)
    dq.append(0)

    for episode in range(args.episodes):
        done = False
        state = env.reset()
        score = 0.

        while not done:
            for _ in range(args.n_rollout):
                state = torch.from_numpy(state).float()

                prob = local_model._pi(state)
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                local_model._put_data(state.numpy(), next_state, reward, done, action, prob[action])
                score += 1
                if done:
                    break
                state = next_state
            local_model._train(optimizer)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
        dq.append(score)
        local_model.load_state_dict(global_model.state_dict())
        if episode % args.print_intervals == 0:
            print('[Episode: {}, Rank: {}] score: {}'.format(episode, rank, np.mean(dq)))


def main(args):
    env = load_env(args)

    global_model = Model(env.observation_space.shape[0], env.action_space.n, args)
    global_model.share_memory()

    processes = []
    for rank in range(args.world_size):
        p = mp.Process(target=_train, args=(global_model, rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    args = load_args()
    main(args)
