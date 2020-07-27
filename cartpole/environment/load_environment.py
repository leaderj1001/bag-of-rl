import gym


def load_env(args):
    env = gym.make(args.env)
    return env
