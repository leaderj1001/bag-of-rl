import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from environment.load_environment import load_env
from config.cartpole_config import load_args

from draw_plot.draw_plot import draw_plot

import numpy as np
from collections import deque
import os


class ActorCritic(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(ActorCritic, self).__init__()
        self.hidden = [256]
        self.args = args

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden[0]),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(self.hidden[0], out_channels)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden[0], 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        self.data = []

    def _pi(self, x):
        out = self.fc1(x)
        return F.softmax(self.actor(out), dim=-1)

    def _v(self, x):
        out = self.fc1(x)
        return self.critic(out)

    def _train(self):
        states, next_states, rewards, dones, actions, probs = self._make_batch()

        values = self._v(states)

        td_target = rewards + self.args.gamma * self._v(next_states) * dones
        delta = td_target - values

        # actor loss: -log(pi) * delta
        # - delta: reward + gamma * next_value * done - value
        # critic loss: (value - td_error)^2
        actor_loss = -torch.log(probs) * delta.detach()
        critic_loss = F.smooth_l1_loss(self._v(states), td_target.detach())

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss.mean().item()

    def _history(self, state, next_state, reward, done, action, prob):
        self.data.append([state, next_state, reward, done, action, prob])

    def _make_batch(self):
        states, next_states, rewards, dones, actions, probs = [], [], [], [], [], []
        for state, next_state, reward, done, action, prob in self.data:
            states.append(state.numpy())
            next_states.append(next_state)
            rewards.append([reward / 100.])
            done_mask = 0. if done else 1.
            dones.append([done_mask])
            actions.append([action])
            probs.append(prob.view(-1, 1))

        self.data = []
        return torch.tensor(states).float(),\
               torch.tensor(next_states).float(),\
               torch.tensor(rewards).float(),\
               torch.tensor(dones).float(),\
               torch.tensor(actions), torch.cat(probs)


def main(args):
    env = load_env(args)

    in_channels, out_channels = env.observation_space.shape[0], env.action_space.n
    print(in_channels, out_channels)
    dq = deque(maxlen=100)
    dq.append(0)

    model = ActorCritic(in_channels, out_channels, args)
    if args.is_cuda:
        model = model.cuda()

    rewards_list = []
    episodes_list = []
    losses_list = []
    for episode in range(args.episodes):
        score = 0.
        done = False
        state = env.reset()

        loss = 0.
        while not done:
            for t in range(args.n_rollout):
                state = torch.from_numpy(state).float()
                if args.is_cuda:
                    state = state.cuda()

                prob = model._pi(state)
                m = Categorical(prob)
                action = m.sample().item()

                next_state, reward, done, _ = env.step(action)
                model._history(state, next_state, reward, done, action, prob[action])
                state = next_state

                score += reward
                if done:
                    break
            loss += model._train()
        rewards_list.append(score)
        dq.append(score)
        losses_list.append(loss)
        episodes_list.append(episode)
        if episode % args.print_intervals == 0 and episode != 0:
            print("# of episode :{0}, avg score : {1:.2f}".format(episode, np.mean(dq), axis=-1))

        if not os.path.isdir('./actor_critic'):
            os.mkdir('actor_critic')
        draw_plot(episodes_list, losses_list, rewards_list, path='./actor_critic/loss_reward_plot.jpg')


if __name__ == '__main__':
    args = load_args()
    main(args)
