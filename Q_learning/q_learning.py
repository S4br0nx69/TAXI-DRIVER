import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
try:
    from IPython.display import clear_output, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
import random


class Taxi:
    def __init__(self, render_mode="rgb_array", env_version="v4"):
        self.render_mode = render_mode
        try:
            self.env = gym.make(f"Taxi-{env_version}", render_mode=render_mode)
        except (gym.error.DeprecatedEnv, gym.error.VersionNotFound):
            fallback = "v4" if env_version == "v3" else "v3"
            print(f"Taxi-{env_version} indisponible, fallback sur Taxi-{fallback}")
            self.env = gym.make(f"Taxi-{fallback}", render_mode=render_mode)
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        # Hyperparamètres communs
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def train(self, train_episodes=25000, training_graph=False):
        """Entraîne l'agent via Q-Learning avec epsilon-greedy décroissant.
        Retourne un np.array des rewards par épisode."""
        reward_per_episode = np.zeros(train_episodes)
        steps_per_episode = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)

        for i in range(train_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            penalties = 0

            while not done:
                # Epsilon-greedy
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                # Bellman update
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = (1 - self.alpha) * old_value + \
                    self.alpha * (reward + self.gamma * next_max)

                if reward == -10:
                    penalties += 1

                total_reward += reward
                steps += 1
                state = next_state

            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            reward_per_episode[i] = total_reward
            steps_per_episode[i] = steps
            penalties_per_episode[i] = penalties

            if i % 1000 == 0:
                avg_reward = reward_per_episode[max(0, i - 1000):i + 1].mean() if i > 0 else total_reward
                print(f"Episode {i:>6}/{train_episodes} | "
                      f"ε={self.epsilon:.4f} | "
                      f"Avg reward (last 1k): {avg_reward:.2f} | "
                      f"Steps: {steps}")

        print(f"\nTraining terminé — {train_episodes} épisodes")
        print(f"  Mean reward : {reward_per_episode.mean():.2f}")
        print(f"  Mean steps  : {steps_per_episode.mean():.2f}")
        print(f"  Mean penalties : {penalties_per_episode.mean():.2f}")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode)

        return reward_per_episode

    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False, final_frame_pause=0):
        """Évalue l'agent entraîné (greedy policy, ε=0)."""
        total_rewards = []
        total_steps = []
        total_penalties = []

        for i in range(test_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            penalties = 0

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                if reward == -10:
                    penalties += 1
                episode_reward += reward
                steps += 1

                if not fast_testing:
                    self._render_frame(state, action, reward, episode_reward, i + 1)
                    sleep(timestamp)

            if not fast_testing and final_frame_pause > 0:
                sleep(final_frame_pause)

            total_rewards.append(episode_reward)
            total_steps.append(steps)
            total_penalties.append(penalties)

        if not fast_testing:
            plt.close()

        avg_steps = np.mean(total_steps)
        avg_penalties = np.mean(total_penalties)
        avg_reward = np.mean(total_rewards)

        print(f"\nRésultats après {test_episodes} épisodes de test :")
        print(f"  Average steps    : {avg_steps:.2f}")
        print(f"  Average penalties: {avg_penalties:.2f}")
        print(f"  Average reward   : {avg_reward:.2f}")

        return avg_steps, avg_penalties, avg_reward

    def _render_frame(self, state, action, reward, episode_reward, episode):
        """Affiche une frame selon le render_mode."""
        frame = self.env.render()
        if isinstance(frame, np.ndarray) and HAS_IPYTHON:
            plt.imshow(frame)
            plt.title(f"Episode: {episode} | Action: {action} | "
                      f"Reward: {reward} | Total: {episode_reward}")
            plt.axis('off')
            clear_output(wait=True)
            display(plt.gcf())
            plt.clf()
        else:
            if HAS_IPYTHON:
                clear_output(wait=True)
            print(f"\r State: {state} | Action: {action} | "
                  f"Reward: {reward} | Total: {episode_reward} | "
                  f"Episode: {episode}", end="", flush=True)

    def _plot_training(self, rewards, steps):
        """Graphiques de convergence post-entraînement."""
        window = 500
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Rewards
        avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(avg_rewards, linewidth=0.8)
        axes[0].set_title(f"Mean Reward (rolling {window})")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)

        # Steps
        avg_steps = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(avg_steps, linewidth=0.8, color='orange')
        axes[1].set_title(f"Mean Steps (rolling {window})")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_metrics.png", dpi=150)
        plt.show()
        print("Graphique sauvegardé : training_metrics.png")