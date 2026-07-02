"""Agent SARSA (State-Action-Reward-State-Action) pour Taxi-v3/v4.
Algorithme on-policy : la mise à jour utilise l'action réellement choisie au step suivant (≠ Q-Learning qui prend le max).
Historique des corrections (bug policy_type ignoré en n-step, seed, test() en dict) : voir BENCHMARK.md.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random

try:
    from IPython.display import clear_output, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class Sarsa:
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
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        # Hyperparamètres individuels SARSA
        self.policy_type = 'epsilon_greedy'  # 'epsilon_greedy', 'softmax', 'expected'
        self.temperature = 1.0               # pour softmax (Boltzmann)
        self.n_steps = 1                     # 1 = SARSA standard, n > 1 = n-step SARSA
        self.lambda_ = 0.0                   # SARSA(λ) : 0=standard, 0<λ<1=traces d'éligibilité, 1≈MC

    def _choose_action(self, state):
        """Sélection selon la politique choisie."""
        if self.policy_type == 'softmax':
            q_values = self.q_table[state]
            scaled = q_values / max(self.temperature, 1e-8)
            exp_q = np.exp(scaled - np.max(scaled))
            probs = exp_q / exp_q.sum()
            return np.random.choice(self.env.action_space.n, p=probs)
        # epsilon_greedy et expected utilisent la même sélection
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def _expected_value(self, state):
        """Valeur espérée sous politique epsilon-greedy (pour Expected SARSA) — non définie pour 'softmax'."""
        q_values = self.q_table[state]
        n_actions = self.env.action_space.n
        best_action = np.argmax(q_values)
        probs = np.ones(n_actions) * self.epsilon / n_actions
        probs[best_action] += 1 - self.epsilon
        return np.dot(probs, q_values)

    def train(self, train_episodes=25000, training_graph=False, seed=None):
        """Entraîne l'agent via SARSA. Supporte standard, Expected et n-step."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.lambda_ > 0:
            return self._train_lambda(train_episodes, training_graph, seed=seed)

        if self.n_steps > 1:
            return self._train_nstep(train_episodes, training_graph, seed=seed)

        reward_per_episode = np.zeros(train_episodes)
        steps_per_episode = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)

        for i in range(train_episodes):
            if seed is not None and i == 0:
                state, _ = self.env.reset(seed=seed)
            else:
                state, _ = self.env.reset()
            action = self._choose_action(state)
            done = False
            total_reward = 0
            steps = 0
            penalties = 0

            while not done:
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                next_action = self._choose_action(next_state)

                old_value = self.q_table[state, action]
                if self.policy_type == 'expected':
                    next_value = self._expected_value(next_state)
                else:
                    next_value = self.q_table[next_state, next_action]

                self.q_table[state, action] = old_value + \
                    self.alpha * (reward + self.gamma * next_value - old_value)

                if reward == -10:
                    penalties += 1

                total_reward += reward
                steps += 1
                state = next_state
                action = next_action

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

        if self.epsilon > 0.05:
            print(f"  ATTENTION : epsilon final = {self.epsilon:.4f} (> 0.05). "
                  f"epsilon_decay={self.epsilon_decay} est probablement trop lent "
                  f"pour {train_episodes} épisodes.")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode)

        return reward_per_episode

    def _train_lambda(self, train_episodes, training_graph, seed=None):
        """SARSA(λ) avec traces d'éligibilité (Sutton & Barto ch.12) — généralise SARSA (λ=0) et Monte Carlo (λ→1)."""
        n_states  = self.env.observation_space.n
        n_actions = self.env.action_space.n

        reward_per_episode    = np.zeros(train_episodes)
        steps_per_episode     = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)

        for ep in range(train_episodes):
            eligibility = np.zeros((n_states, n_actions))

            if seed is not None and ep == 0:
                state, _ = self.env.reset(seed=seed)
            else:
                state, _ = self.env.reset()
            action = self._choose_action(state)

            done         = False
            total_reward = 0
            steps        = 0
            penalties    = 0

            while not done:
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                next_action = self._choose_action(next_state)

                if self.policy_type == 'expected':
                    next_value = self._expected_value(next_state)
                else:
                    next_value = self.q_table[next_state, next_action]

                delta = reward + self.gamma * next_value - self.q_table[state, action]

                eligibility[state, action] += 1.0
                self.q_table += self.alpha * delta * eligibility
                eligibility  *= self.gamma * self.lambda_

                if reward == -10:
                    penalties += 1
                total_reward += reward
                steps        += 1
                state  = next_state
                action = next_action

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            reward_per_episode[ep]    = total_reward
            steps_per_episode[ep]     = steps
            penalties_per_episode[ep] = penalties

            if ep % 1000 == 0:
                avg_reward = reward_per_episode[max(0, ep - 1000):ep + 1].mean() if ep > 0 else total_reward
                print(f"Episode {ep:>6}/{train_episodes} | "
                      f"ε={self.epsilon:.4f} | "
                      f"Avg reward (last 1k): {avg_reward:.2f} | "
                      f"Steps: {steps}")

        print(f"\nTraining terminé — {train_episodes} épisodes (SARSA λ={self.lambda_})")
        print(f"  Mean reward : {reward_per_episode.mean():.2f}")
        print(f"  Mean steps  : {steps_per_episode.mean():.2f}")
        print(f"  Mean penalties : {penalties_per_episode.mean():.2f}")

        if self.epsilon > 0.05:
            print(f"  ATTENTION : epsilon final = {self.epsilon:.4f} (> 0.05). "
                  f"epsilon_decay={self.epsilon_decay} est probablement trop lent "
                  f"pour {train_episodes} épisodes.")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode)

        return reward_per_episode

    def _train_nstep(self, train_episodes, training_graph, seed=None):
        """Entraînement n-step SARSA (algorithme tabulaire, Sutton & Barto ch.7)."""
        n = self.n_steps
        reward_per_episode = np.zeros(train_episodes)
        steps_per_episode = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)

        for ep in range(train_episodes):
            if seed is not None and ep == 0:
                state, _ = self.env.reset(seed=seed)
            else:
                state, _ = self.env.reset()
            action = self._choose_action(state)

            states = [state]
            actions = [action]
            rewards = [0.0]  # rewards[0] dummy, rewards[t] = récompense au pas t

            T = float('inf')
            t = 0
            total_reward = 0
            penalties = 0

            while True:
                if t < T:
                    next_state, reward, done, truncated, _ = self.env.step(actions[t])
                    done = done or truncated

                    states.append(next_state)
                    rewards.append(reward)

                    if reward == -10:
                        penalties += 1
                    total_reward += reward

                    if done:
                        T = t + 1
                    else:
                        next_action = self._choose_action(next_state)
                        actions.append(next_action)

                tau = t - n + 1
                if tau >= 0:
                    G = sum(self.gamma ** (i - tau - 1) * rewards[i]
                            for i in range(tau + 1, min(tau + n, T) + 1))
                    if tau + n < T:
                        # bootstrap conforme à policy_type, y compris en n-step
                        if self.policy_type == 'expected':
                            bootstrap = self._expected_value(states[tau + n])
                        else:
                            bootstrap = self.q_table[states[tau + n], actions[tau + n]]
                        G += self.gamma ** n * bootstrap

                    s_tau, a_tau = states[tau], actions[tau]
                    self.q_table[s_tau, a_tau] += self.alpha * (G - self.q_table[s_tau, a_tau])

                t += 1
                if tau == T - 1:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            reward_per_episode[ep] = total_reward
            steps_per_episode[ep] = T
            penalties_per_episode[ep] = penalties

            if ep % 1000 == 0:
                avg_reward = reward_per_episode[max(0, ep - 1000):ep + 1].mean() if ep > 0 else total_reward
                print(f"Episode {ep:>6}/{train_episodes} | "
                      f"ε={self.epsilon:.4f} | "
                      f"Avg reward (last 1k): {avg_reward:.2f} | "
                      f"Steps: {int(T)}")

        print(f"\nTraining terminé — {train_episodes} épisodes ({n}-step SARSA)")
        print(f"  Mean reward : {reward_per_episode.mean():.2f}")
        print(f"  Mean steps  : {steps_per_episode.mean():.2f}")
        print(f"  Mean penalties : {penalties_per_episode.mean():.2f}")

        if self.epsilon > 0.05:
            print(f"  ATTENTION : epsilon final = {self.epsilon:.4f} (> 0.05). "
                  f"epsilon_decay={self.epsilon_decay} est probablement trop lent "
                  f"pour {train_episodes} épisodes.")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode)

        return reward_per_episode

    def test(self, test_episodes=1, timestamp=0.2, fast_testing=False,
             final_frame_pause=0, seed=None):
        """Évalue l'agent entraîné (greedy policy, ε=0).

        Retourne un dict avec métriques agrégées + listes brutes par épisode.
        """
        total_rewards = []
        total_steps = []
        total_penalties = []
        total_completions = []

        for i in range(test_episodes):
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

        avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(avg_rewards, linewidth=0.8)
        axes[0].set_title(f"SARSA — Mean Reward (rolling {window})")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)

        avg_steps = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(avg_steps, linewidth=0.8, color='orange')
        axes[1].set_title(f"SARSA — Mean Steps (rolling {window})")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("sarsa_training_metrics.png", dpi=150)
        plt.show()
        print("Graphique sauvegardé : sarsa_training_metrics.png")