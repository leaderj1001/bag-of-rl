import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical, Normal

from environment.load_environment import load_env

import os
import numpy as np
from collections import deque
import argparse
import random


def hard_target_update(net, target_net):
    target_net.load_state_dict(net.state_dict())


def soft_target_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def get_action(mu, std):
    normal = Normal(mu, std)
    z = normal.rsample()
    action = torch.tanh(z)

    return action.data.numpy()


def eval_action(mu, std, epsilon=1e-6):
    normal = Normal(mu, std)
    z = normal.rsample()
    action = torch.tanh(z)
    log_prob = normal.log_prob(z)

    log_prob -= torch.log(1 - action.pow(2) + epsilon)
    log_policy = log_prob.sum(1, keepdim=True)

    return action, log_policy


class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, args, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden = 64

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU()
        )

        self.mu_fc = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

        self.log_std_fc = nn.Sequential(
            nn.Linear(self.hidden, out_channels)
        )

    def forward(self, x):
        out = self.fc1(x)

        mu = self.mu_fc(out)
        log_std = self.log_std_fc(out)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = torch.exp(log_std)

        return mu, std


class Critic(nn.Module):
    def __init__(self, in_channels, action_size, args):
        super(Critic, self).__init__()
        self.hidden = 64

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels + action_size, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_channels + action_size, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)

        q_value1 = self.fc1(x)
        q_value2 = self.fc2(x)

        return q_value1, q_value2


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--alpha_lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--max_iter_num', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--goal_score', type=int, default=-300)

    return parser.parse_args()


def _train_model(actor, critic, target_critic, mini_batch, actor_optimizer, critic_optimizer, alpha_optimizer, target_entropy, log_alpha, alpha, args):
    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    next_states = np.vstack(mini_batch[:, 3])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    masks = list(mini_batch[:, 4])

    actions = torch.tensor(actions).squeeze(dim=1)
    rewards = torch.tensor(rewards).squeeze(dim=1)
    masks = torch.tensor(masks)

    q_value1, q_value2 = critic(torch.tensor(states).float(), actions)

    mu, std = actor(torch.tensor(next_states).float())
    next_policy, next_log_policy = eval_action(mu, std)
    target_next_q_value1, target_next_q_value2 = target_critic(torch.tensor(next_states).float(), next_policy)

    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    min_target_next_q_value = min_target_next_q_value.squeeze(dim=1) - alpha * next_log_policy.squeeze(dim=1)
    target = rewards + args.gamma * min_target_next_q_value * masks

    critic_loss1 = F.mse_loss(q_value1.squeeze(dim=1), target.detach())
    critic_optimizer.zero_grad()
    critic_loss1.backward()
    critic_optimizer.step()

    critic_loss2 = F.mse_loss(q_value2.squeeze(dim=1), target.detach())
    critic_optimizer.zero_grad()
    critic_loss2.backward()
    critic_optimizer.step()

    mu, std = actor(torch.tensor(states).float())
    policy, log_policy = eval_action(mu, std)

    q_value1, q_value2 = critic(torch.tensor(states).float(), policy)
    min_q_value = torch.min(q_value1, q_value2)

    actor_loss = ((alpha * log_policy) - min_q_value).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_policy + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    alpha = torch.exp(log_alpha)

    return alpha


def main(args):
    env = load_env(args)
    env.seed(500)
    torch.manual_seed(500)

    in_channels, out_channels = env.observation_space.shape[0], env.action_space.shape[0]
    print(in_channels, out_channels)

    actor = Actor(in_channels, out_channels, args)
    critic = Critic(in_channels, out_channels, args)
    target_critic = Critic(in_channels, out_channels, args)

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    hard_target_update(critic, target_critic)

    target_entropy = -torch.prod(torch.tensor(out_channels)).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha = torch.exp(log_alpha)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    replay_buffer = deque(maxlen=100000)
    recent_rewards = deque(maxlen=100)
    steps = 0

    for episode in range(args.max_iter_num):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, in_channels])

        while not done:
            steps += 1

            mu, std = actor(torch.tensor(state).float())
            action = get_action(mu, std)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, in_channels])
            mask = 0. if done else 1.

            replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state
            score += reward

            if steps > args.batch_size:
                mini_batch = random.sample(replay_buffer, args.batch_size)

                actor.train(), critic.train(), target_critic.train()
                alpha = _train_model(actor, critic, target_critic, mini_batch, actor_optimizer, critic_optimizer, alpha_optimizer, target_entropy, log_alpha, alpha, args)

                soft_target_update(critic, target_critic, args.tau)

            if done:
                recent_rewards.append(score)

        if episode % args.log_interval == 0:
            print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))

        if np.mean(recent_rewards) > args.goal_score:
            # if not os.path.isdir(args.save_path):
            #     os.makedirs(args.save_path)

            # ckpt_path = args.save_path + 'model.pth.tar'
            # torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed -300. So end')
            break


if __name__ == '__main__':
    args = load_args()
    main(args)
