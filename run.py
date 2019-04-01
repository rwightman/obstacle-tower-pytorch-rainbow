import argparse
from datetime import datetime
import numpy as np
import torch
import sys

from agent_eval import AgentEval
from env import create_env


def run_episode(env, agent):
    reward_sum = 0
    done = True
    while True:
        if done:
            state = env.reset()
            reward_sum = 0
            done = False

        action = agent.act_e_greedy(state, epsilon=0.0005)  # Choose an action ε-greedily
        state, reward, done, _ = env.step(action)  # Step
        reward_sum += reward

        if done:
            break
    return reward_sum


def run_evaluation(env, agent):
    while not env.unwrapped.done_grading():
        run_episode(env, agent)
        env.reset()


parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--worker', type=int, default=0, help='worker id (base port offset)')

#FIXME move to default class init
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')


def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        # Disable nondeterministic ops (not sure if critical but better safe than sorry)
        #torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    # Environment
    env = create_env(
        args.environment_filename,
        custom=False,
        skip_frames=1,
        realtime=args.render,
        worker_id=args.worker,
        device=args.device)

    agent = AgentEval(args, env)

    if env.unwrapped.is_grading():
        print('grading...')
        run_evaluation(env, agent)
    else:
        print('testing...')
        rewards = []
        for _ in range(10):
            episode_rewards = run_episode(env, agent)
            rewards.append(episode_rewards)
        env.close()
        print(sum(rewards) / len(rewards))


if __name__ == '__main__':
    main()
