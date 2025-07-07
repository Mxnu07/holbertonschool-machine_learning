#!/usr/bin/env python3
"""
Q-learning algorithm for training on the FrozenLake environment
"""

import numpy as np
import gymnasium as gym


def epsilon_greedy(Q, state, epsilon):
    """
    Selects an action using the epsilon-greedy policy
    """
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])


def train(
    env,
    Q,
    episodes=1000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    min_epsilon=0.01,
    epsilon_decay=0.995,
):
    """
    Trains the Q-learning algorithm on the given environment

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        episodes: the total number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon: the minimum value that epsilon should decay to
        epsilon_decay: the decay rate for updating epsilon between episodes
    Returns:
        Q: the updated Q-table
        total_rewards: a list containing the rewards per episode
    """
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        rewards_current_episode = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, _, info = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            state = new_state
            rewards_current_episode += reward
            if done:
                break
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_rewards.append(rewards_current_episode)
    return Q, total_rewards
