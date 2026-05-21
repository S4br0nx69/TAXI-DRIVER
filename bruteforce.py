"""Brute-force (random agent) baseline pour Taxi-v3/v4.
Sert de point de comparaison pour les algorithmes RL.
Usage : python3 bruteforce.py
"""

import gymnasium as gym
import numpy as np
import time


def run_bruteforce(test_episodes=100, max_steps=500, render=False):
    """Exécute un agent aléatoire sur Taxi et retourne les métriques."""
    try:
        env = gym.make("Taxi-v4", render_mode="ansi" if render else None)
    except (gym.error.DeprecatedEnv, gym.error.VersionNotFound):
        env = gym.make("Taxi-v3", render_mode="ansi" if render else None)

    all_steps = []
    all_rewards = []
    all_penalties = []

    for ep in range(test_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        penalties = 0

        while not done and steps < max_steps:
            action = env.action_space.sample()  # Action aléatoire
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            if reward == -10:
                penalties += 1
            total_reward += reward
            steps += 1

            if render:
                print(env.render())

        all_steps.append(steps)
        all_rewards.append(total_reward)
        all_penalties.append(penalties)

    env.close()

    results = {
        "episodes": test_episodes,
        "mean_steps": np.mean(all_steps),
        "std_steps": np.std(all_steps),
        "mean_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "mean_penalties": np.mean(all_penalties),
        "success_rate": sum(1 for r in all_rewards if r > 0) / test_episodes * 100,
    }
    return results


if __name__ == "__main__":
    print("=== Brute-force Baseline (Random Agent) ===\n")

    start = time.time()
    results = run_bruteforce(test_episodes=1000)
    elapsed = round(time.time() - start, 2)

    print(f"Épisodes        : {results['episodes']}")
    print(f"Mean steps      : {results['mean_steps']:.2f} (±{results['std_steps']:.2f})")
    print(f"Mean reward     : {results['mean_reward']:.2f} (±{results['std_reward']:.2f})")
    print(f"Mean penalties  : {results['mean_penalties']:.2f}")
    print(f"Success rate    : {results['success_rate']:.1f}%")
    print(f"Execution time  : {elapsed}s")