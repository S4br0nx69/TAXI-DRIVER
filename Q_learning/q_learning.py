"""Agent Q-Learning pour Taxi-v3/v4.
Méthode off-policy par différence temporelle (TD) : la mise à jour utilise
le max sur les actions de l'état suivant, indépendamment de la politique suivie.

Corrections apportées (cf. revue de code) :
- env_version par défaut passé à "v3" (évite de déclencher systématiquement
  l'exception de fallback si Taxi-v4 n'existe pas dans l'environnement
  Gymnasium installé — c'est "v3" qui est utilisé dans tout le reste du projet).
- Ajout d'un paramètre `seed` optionnel sur train() et test() pour la
  reproductibilité (random, numpy, et RNG interne de l'environnement
  Gymnasium, initialisé une seule fois au premier reset).
- test() retourne maintenant un dict avec les métriques agrégées ET les
  listes brutes par épisode. Avec l'ancienne signature (un tuple de moyennes),
  il était impossible de calculer un écart-type ou un intervalle de confiance
  après coup — la donnée brute n'existait nulle part.
- Avertissement explicite si epsilon n'est pas descendu sous un seuil bas en
  fin d'entraînement (signe que epsilon_decay est trop lent pour le budget
  d'épisodes alloué — confond la comparaison d'hyperparamètres avec un simple
  problème de convergence inachevée).
"""

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
    def __init__(self, render_mode="rgb_array", env_version="v3"):
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

        # Hyperparamètres individuels Q-Learning
        self.optimistic_init = 0.0   # init Q-table (0=zéros, >0=optimiste → force l'exploration)
        self.double_q        = False  # Double Q-Learning (deux Q-tables pour réduire l'overestimation)
        self.q_table2        = None   # deuxième Q-table, créée dans train() si double_q=True

    def train(self, train_episodes=25000, training_graph=False, seed=None):
        """Entraîne l'agent via Q-Learning avec epsilon-greedy décroissant.
        Retourne un np.array des rewards par épisode.

        seed : si fourni, fixe random/numpy et initialise le RNG de
        l'environnement une seule fois au premier reset (convention
        Gymnasium), pour rendre l'entraînement reproductible.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.q_table = np.full(
            [self.env.observation_space.n, self.env.action_space.n],
            self.optimistic_init, dtype=np.float64
        )
        if self.double_q:
            self.q_table2 = np.full(
                [self.env.observation_space.n, self.env.action_space.n],
                self.optimistic_init, dtype=np.float64
            )

        reward_per_episode = np.zeros(train_episodes)
        steps_per_episode = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)

        for i in range(train_episodes):
            if seed is not None and i == 0:
                state, _ = self.env.reset(seed=seed)
            else:
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
                    if self.double_q and self.q_table2 is not None:
                        action = np.argmax(self.q_table[state] + self.q_table2[state])
                    else:
                        action = np.argmax(self.q_table[state])

                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                # Bellman update
                if self.double_q and self.q_table2 is not None:
                    if random.random() < 0.5:
                        best_a = np.argmax(self.q_table[next_state])
                        target = reward + self.gamma * self.q_table2[next_state, best_a]
                        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
                    else:
                        best_a = np.argmax(self.q_table2[next_state])
                        target = reward + self.gamma * self.q_table[next_state, best_a]
                        self.q_table2[state, action] += self.alpha * (target - self.q_table2[state, action])
                else:
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

        # FIX : diagnostic de convergence d'epsilon. Un epsilon_decay trop lent
        # pour le budget d'épisodes alloué fausse la comparaison entre
        # hyperparamètres (l'agent n'a pas fini d'explorer, indépendamment de
        # la qualité de alpha/gamma).
        if self.epsilon > 0.05:
            print(f"  ATTENTION : epsilon final = {self.epsilon:.4f} (> 0.05). "
                  f"epsilon_decay={self.epsilon_decay} est probablement trop lent "
                  f"pour {train_episodes} épisodes : l'agent explore encore "
                  f"significativement à la fin de l'entraînement.")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode)

        return reward_per_episode

    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False,
             final_frame_pause=0, seed=None):
        """Évalue l'agent entraîné (greedy policy, ε=0).

        Retourne un dict avec les métriques agrégées ET les listes brutes par
        épisode, pour permettre le calcul d'écart-type / intervalle de
        confiance — impossible avec une simple moyenne.
        """
        total_rewards = []
        total_steps = []
        total_penalties = []
        total_completions = []

        for i in range(test_episodes):
            # Décalage du seed pour ne pas reproduire exactement la même
            # séquence de resets qu'en entraînement.
            if seed is not None and i == 0:
                state, _ = self.env.reset(seed=seed + 10_000)
            else:
                state, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            penalties = 0
            completed = False

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if terminated:
                    completed = True
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
            total_completions.append(1 if completed else 0)

        if not fast_testing:
            plt.close()

        avg_steps = float(np.mean(total_steps))
        avg_penalties = float(np.mean(total_penalties))
        avg_reward = float(np.mean(total_rewards))
        completion_rate = float(np.mean(total_completions)) * 100

        print(f"\nRésultats après {test_episodes} épisodes de test :")
        print(f"  Average steps    : {avg_steps:.2f}")
        print(f"  Average penalties: {avg_penalties:.2f}")
        print(f"  Average reward   : {avg_reward:.2f} (± {np.std(total_rewards):.2f})")
        print(f"  Completion rate  : {completion_rate:.1f}%")

        return {
            'reward': avg_reward,
            'steps': avg_steps,
            'penalties': avg_penalties,
            'completion_rate': completion_rate,
            'reward_std': float(np.std(total_rewards)),
            'steps_std': float(np.std(total_steps)),
            'episode_rewards': total_rewards,
            'episode_steps': total_steps,
            'episode_penalties': total_penalties,
            'episode_completions': total_completions,
        }

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