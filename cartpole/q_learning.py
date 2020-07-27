import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

from config.cartpole_config import load_args
from environment.load_environment import load_env
from draw_plot.draw_plot import draw_plot

from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, args):
        self.memory = deque(maxlen=args.buffer_limit)

    def _put_data(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.memory, n)

        states, next_states, rewards, actions, dones = [], [], [], [], []
        for transition in mini_batch:
            state, next_state, reward, action, done = transition
            states.append(state)
            next_states.append(next_state)
            actions.append([action])
            rewards.append([reward])
            dones.append([done])

        return torch.tensor(states), torch.tensor(next_states, dtype=torch.float), torch.tensor(rewards), torch.tensor(actions), torch.tensor(dones)

    def size(self):
        return len(self.memory)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(Model, self).__init__()
        self.hidden = 64
        self.args = args

        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, out_channels)
        )

    def forward(self, x):
        out = self.fc1(x)
        return out

    def _sample_action(self, state, epsilon):
        out = self.forward(state)
        p = random.random()
        if p < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def _train(q, q_target, memory, optimizer, args):
    for i in range(args.train_epochs):
        states, next_states, rewards, actions, dones = memory.sample(args.batch_size)

        q_out = q(states)
        q_a = q_out.gather(1, actions)

        max_next_q = q_target(next_states).max(1)[0].unsqueeze(1)
        target = rewards + args.gamma * max_next_q * dones

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):
    env = load_env(args)
    in_channels, out_channels = env.observation_space.shape[0], env.action_space.n

    q = Model(in_channels, out_channels, args)
    q_target = Model(in_channels, out_channels, args)

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer(args)

    score = 0.
    optimizer = optim.Adam(q.parameters(), lr=args.lr)

    for episode in range(1, args.episodes + 1):
        epsilon = max(0.01, 0.08 - 0.01 * (episode / 200))

        state = env.reset()
        done = False

        while not done:
            state = torch.from_numpy(state).float()
            action = q._sample_action(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            done_mask = 0. if done else 1.
            memory._put_data((state.numpy(), next_state, reward / 100., action, done_mask))

            state = next_state

            score += 1.
            if done:
                break

        if memory.size() > 2000:
            _train(q, q_target, memory, optimizer, args)

        if episode % args.print_intervals == 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                episode, score / args.print_intervals, memory.size(), epsilon * 100))
            score = 0.


if __name__ == '__main__':
    args = load_args()
    main(args)
