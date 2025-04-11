# eval.py
import numpy as np

def evaluate_agent(env, model, num_episodes=10):
    """
    Evaluate the agent in the given environment for a specified number of episodes.
    
    Args:
        env: The environment to evaluate the agent in.
        model: The trained agent model to be evaluated.
        num_episodes (int): Number of episodes to evaluate over.

    Returns:
        float: Mean reward over the evaluation episodes.
    """
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            done = done.any()
        rewards.append(episode_reward)
    mean_reward = np.mean(rewards)
    print(f"Mean Reward: {mean_reward}")
    return mean_reward
