import argparse


def load_args():
    parser = argparse.ArgumentParser('CartPole')

    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--print_intervals', type=int, default=40)
    parser.add_argument('--n_rollout', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--method', type=str, default='delta')
    parser.add_argument('--buffer_limit', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--world_size', type=int, default=4)

    return parser.parse_args()
