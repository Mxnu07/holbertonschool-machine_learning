#!/usr/bin/env python3
"""
Training loop for MC policy gradient
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train the policy using Monte-Carlo policy gradient.
    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True render the environment every 1000 episodes
    Returns: list of the scores obtained after each episode
    """
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[0]
        episode_gradients = []
        episode_rewards = []
        done = False

        if show_result and episode % 1000 == 0:
            env.render()

        while not done:
            action, grad = policy_gradient(state, weights)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            episode_gradients.append(grad)
            state = next_state
            done = terminated or truncated

        score = sum(episode_rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

        for i, gradient in enumerate(episode_gradients):
            reward = sum([R * gamma ** R for R in episode_rewards[i:]])
            weights += alpha * gradient * reward

    return scores
