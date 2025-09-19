import numpy as np
from collections import deque
import torch
import argparse
from buffer import ReplayBuffer
from utils import save, collect_random
import random
from agent import SAC
import os

from lb_env.lb_cluster_sac import ClusterEnv
# from lb_env.eval.lb_cluster_t_sac_eval import ClusterEnv_eval

def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="MCRL2", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="ClusterEnv", help="Gym environment name, default: ClusterEnv")
    parser.add_argument("--episodes", type=int, default=4000, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Maximal training dataset size, default: 100000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=1, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    
    args = parser.parse_args()
    return args

def train(config, env):

    env = env
    
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    # average10 = deque(maxlen=10)
    total_steps = 0

    agent = SAC(state_size=34,
                     action_size=15,
                     device=device)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

    collect_random(env=env, dataset=buffer, num_samples=512)

    for i in range(1, config.episodes+1):
        # print("Episode {} starts".format(i))
        state = env._reset()
        episode_steps = 0
        rewards = 0
        while True:
            action = agent.get_action(state)[0]
            # print("step action is {}".format(action))
            steps += 1
            next_state, reward, done = env._step(action)
            buffer.add(state, action, reward, next_state, done)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(buffer.sample(), gamma=0.99)
            state = next_state
            rewards += reward
            episode_steps += 1
            if done:
                break

        # average10.append(rewards)
        total_steps += episode_steps
        print("Episode: {} | Reward: {} | Policy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))

        if i % config.save_every == 0:
            save(config, save_name="SAC_discrete", model=agent.actor_local, ep=i)

if __name__ == "__main__":

    setup_seed(3407)

    env = ClusterEnv()

    config = get_config()
    train(config, env)


