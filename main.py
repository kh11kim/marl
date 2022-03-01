import argparse
#from rxbot.rxbot_reach import RxbotReachEnv
from pybullet_wrapper import *
from agent import Agent
from common.replay_buffer import Buffer
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="panda_joint1", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.2, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--hidden-layer", type=int, default=128, help="number of hidden layer nodes")
    
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args

env = PandaGymNewEnv(render=True, reward_type="col")

args = get_args()
args.n_agents = 7
args.obs_shape = [8 for i in range(args.n_agents)]
action_shape = []
for i in range(args.n_agents):
    action_shape.append(1)
args.action_shape = action_shape[:args.n_agents] 
args.high_action = 1
args.low_action = -1


agents = []
for i in range(args.n_agents):
    agent = Agent(i, args)
    agents.append(agent)
buffer = Buffer(args)

def evaluate(env, args, agents):
    returns = []
    for episode in range(args.evaluate_episodes):
        # reset the environment
        obs = env.reset()
        new_obs = []
        for i in range(7):
            new_obs.append([*obs[:7], obs[7+i]])
        obs = np.array(new_obs)
        rewards = 0
        for time_step in range(args.evaluate_episode_len):
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(obs[agent_id], args.noise_rate, args.epsilon)
                    actions.append(action)
            actions = np.array(actions).flatten()
            obs_, r, done, info = env.step(actions)
            new_obs = []
            for i in range(7):
                new_obs.append([*obs_[:7], obs_[7+i]])
            obs_ = np.array(new_obs)
            #obs_ = np.array([obs_]*args.n_agents)
            rewards += r
            obs = obs_
        returns.append(rewards)
        print('Returns is', rewards)
    return sum(returns) / args.evaluate_episodes    

returns = []
for time_step in tqdm(range(args.time_steps)):
    if time_step % args.max_episode_len == 0:
        obs = env.reset()
        new_obs = []
        for i in range(7):
            new_obs.append([*obs[:7], obs[7+i]])
        obs = np.array(new_obs)
        #obs = np.array([obs]*args.n_agents)
    
    actions = []
    u = []
    with torch.no_grad():
        for agent_id, agent in enumerate(agents):
            action = agent.select_action(obs[agent_id], args.noise_rate, args.epsilon)
            u.append(action)
            actions.append(action)
    actions = np.array(actions).flatten()
    obs_, r, done, info = env.step(actions)
    new_obs = []
    for i in range(7):
        new_obs.append([*obs_[:7], obs_[7+i]])
    obs_ = np.array(new_obs)
    #obs_ = np.array([obs_]*args.n_agents)
    r = np.array([r]*args.n_agents)
    buffer.store_episode(obs, actions, r, obs_)
    obs = obs_
    if buffer.current_size >= args.batch_size:
        transitions = buffer.sample(args.batch_size)
        for agent in agents:
            other_agents = agents.copy()
            other_agents.remove(agent)
            agent.learn(transitions, other_agents)
    save_path = args.save_dir + '/' + args.scenario_name
    if time_step > 0 and time_step % args.evaluate_rate == 0:
        returns.append(evaluate(env, args, agents))
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.xlabel('episode * ' + str(args.evaluate_rate / args.max_episode_len))
        plt.ylabel('average returns')
        plt.savefig(save_path + '/plt.png', format='png')
        plt.close()

    np.save(save_path + '/returns.pkl', returns)


input()