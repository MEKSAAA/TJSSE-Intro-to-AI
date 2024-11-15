import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2", render_mode="human")

# Seeding the environment and other random generators
env.reset(seed=0)
env.action_space.seed(0)
np.random.seed(0)
random.seed(0)

from dqn_agent import Agent

agent = Agent(state_size=8, action_size=4, seed=0)

def dqn(n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    global env  # 声明使用全局变量env

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    average_scores = []  # list containing average scores from each episode
    eps = eps_start  # initialize epsilon
    render = False  # flag for rendering
    best_avg_score = -np.inf  # track the best average score

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()  # Unpack the tuple to get the initial observation
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, = env.step(action)[:4]  # Unpack the first four values
            agent.step(state, action, reward, next_state, done)  # update QNetwork
            state = next_state
            score += reward
            if render:
                env.render()
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        average_scores.append(np.mean(scores_window))  # save average score of the last 100 episodes
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        avg_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            torch.save(agent.qnetwork_local.state_dict(), 'best_checkpoint.pth')

        # 达到平均分数100后启用渲染
        if avg_score >= 100.0 and not render:
            render = True
            env = gym.make("LunarLander-v2", render_mode="human")  # 重新创建环境以启用渲染
            env.reset(seed=0)  # 重新设置种子
            env.action_space.seed(0)
            print('\nAverage Score exceeded 200! Enabling rendering...')

        if avg_score >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


    return scores,average_scores


scores,average_scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='Score')
plt.plot(np.arange(len(average_scores)), average_scores, label='Average Score')
plt.axhline(y=200, color='r', linestyle='--', label='Solved Requirement')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='lower right')
plt.show()

agent.qnetwork_local.load_state_dict(torch.load('best_checkpoint.pth'))

for i in range(3):
    state, _ = env.reset()  # Unpack the tuple to get the initial observation
    for j in range(2000):
        action = agent.act(state)
        env.render()
        state, reward, done, _, = env.step(action)[:4]  # Unpack the first four values
        if done:
            state, _ = env.reset()

env.close()