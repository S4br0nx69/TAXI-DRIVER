"""Agent Deep Q-Learning (DQN) pour Taxi-v3/v4.
Réseau de neurones PyTorch pour approximer la Q-function, replay buffer + target network pour stabiliser l'entraînement.
Historique des corrections (seed torch, test() en dict, use_factored_encoding) : voir BENCHMARK.md.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from IPython.display import clear_output, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


class QNetwork(nn.Module):
    """Réseau de neurones pour approximer Q(s, a)."""
    def __init__(self, state_size, action_size, hidden_sizes=(128, 64)):
        super().__init__()
        layers = []
        prev_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN : Q(s,a) = V(s) + A(s,a) - mean(A) — efficace quand plusieurs actions ont la même valeur."""
    def __init__(self, state_size, action_size, hidden_sizes=(128, 64)):
        super().__init__()
        shared_layers = []
        in_size = state_size
        for h in hidden_sizes[:-1]:
            shared_layers.extend([nn.Linear(in_size, h), nn.ReLU()])
            in_size = h
        self.shared = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        last_h = hidden_sizes[-1]
        self.value_stream = nn.Sequential(
            nn.Linear(in_size, last_h), nn.ReLU(), nn.Linear(last_h, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_size, last_h), nn.ReLU(), nn.Linear(last_h, action_size)
        )

    def forward(self, x):
        shared    = self.shared(x)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Experience replay buffer pour décorréler les échantillons."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, render_mode="rgb_array", env_version="v3"):
        self.render_mode = render_mode
        try:
            self.env = gym.make(f"Taxi-{env_version}", render_mode=render_mode)
        except (gym.error.DeprecatedEnv, gym.error.VersionNotFound):
            fallback = "v4" if env_version == "v3" else "v3"
            print(f"Taxi-{env_version} indisponible, fallback sur Taxi-{fallback}")
            self.env = gym.make(f"Taxi-{fallback}", render_mode=render_mode)

        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Option d'encodage : False = comportement original (one-hot pur,
        # aucune généralisation inter-états possible). True = encodage
        # factorisé (position taxi/passager/destination), qui restaure une
        # vraie capacité de généralisation pour le réseau. Décrit en tête de
        # fichier — désactivé par défaut pour ne rien changer au comportement
        # existant tant que ce n'est pas explicitement demandé.
        self.use_factored_encoding = False
        self.encoded_state_size = self.state_size

        # Hyperparamètres communs
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

        # Hyperparamètres individuels DQN
        self.lr = 0.001
        self.batch_size = 64
        self.target_update = 10
        self.buffer_size = 50000
        self.optimizer_type = 'adam'  # 'adam', 'rmsprop', 'sgd'
        self.hidden_sizes   = (128, 64)  # architecture réseau (couches cachées)
        self.double_dqn     = False       # Double DQN : policy_net choisit, target_net évalue
        self.dueling        = False       # Dueling DQN : streams V(s) et A(s,a) séparés
        self.tau            = 0.0         # soft update : 0=hard update périodique, >0=τ·θ+(1-τ)·θ_target

        # Composants (réseaux + optimiseur construits en dernier, après les hyperparams)
        self.loss_fn = nn.HuberLoss()
        self.memory = ReplayBuffer(capacity=self.buffer_size)
        self._build_networks()
        self.optimizer = self._build_optimizer()

    def _build_networks(self):
        """Instancie policy_net et target_net selon dueling et hidden_sizes."""
        net_cls = DuelingQNetwork if self.dueling else QNetwork
        self.policy_net = net_cls(self.encoded_state_size, self.action_size,
                                  self.hidden_sizes).to(self.device)
        self.target_net = net_cls(self.encoded_state_size, self.action_size,
                                  self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _build_optimizer(self):
        """Construit l'optimiseur selon optimizer_type."""
        if self.optimizer_type == 'rmsprop':
            return optim.RMSprop(self.policy_net.parameters(), lr=self.lr)
        elif self.optimizer_type == 'sgd':
            return optim.SGD(self.policy_net.parameters(), lr=self.lr)
        return optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def _rebuild_networks_for_encoding(self):
        """À appeler après un changement de use_factored_encoding/dueling : reconstruit réseaux + optimiseur."""
        self.encoded_state_size = 4 if self.use_factored_encoding else self.state_size
        self._build_networks()
        self.optimizer = self._build_optimizer()

    def _encode_state(self, state):
        """Encode un état entier en one-hot ou en vecteur factorisé selon self.use_factored_encoding."""
        if self.use_factored_encoding:
            taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(state)
            return np.array([
                taxi_row / 4.0,
                taxi_col / 4.0,
                pass_idx / 4.0,
                dest_idx / 3.0,
            ], dtype=np.float32)
        one_hot = np.zeros(self.state_size, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

    def _choose_action(self, state):
        """Sélection epsilon-greedy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._encode_state(state)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def _learn(self):
        """Mise à jour du réseau via mini-batch du replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Encodage
        state_batch = torch.FloatTensor(
            np.array([self._encode_state(s) for s in states])).to(self.device)
        next_state_batch = torch.FloatTensor(
            np.array([self._encode_state(s) for s in next_states])).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)

        # Q(s, a) courant
        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Q cible
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN : policy_net sélectionne l'action, target_net l'évalue
                best_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
                next_q = self.target_net(next_state_batch).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(next_state_batch).max(1)[0]
            next_q[done_batch] = 0.0
            target_q = reward_batch + self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, train_episodes=25000, training_graph=False, seed=None):
        """Entraîne l'agent DQN, retourne un np.array des rewards (seed fixe aussi torch/CUDA)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        reward_per_episode = np.zeros(train_episodes)
        steps_per_episode = np.zeros(train_episodes)
        penalties_per_episode = np.zeros(train_episodes)
        losses = []

        for i in range(train_episodes):
            if seed is not None and i == 0:
                state, _ = self.env.reset(seed=seed)
            else:
                state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            penalties = 0
            episode_loss = 0

            while not done:
                action = self._choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                self.memory.push(state, action, reward, next_state, done)
                loss = self._learn()
                episode_loss += loss

                if reward == -10:
                    penalties += 1
                total_reward += reward
                steps += 1
                state = next_state

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Mise à jour du target network
            if self.tau > 0:
                # Soft update : θ_target ← τ·θ_policy + (1-τ)·θ_target
                for param, target_param in zip(self.policy_net.parameters(),
                                               self.target_net.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )
            elif i % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            reward_per_episode[i] = total_reward
            steps_per_episode[i] = steps
            penalties_per_episode[i] = penalties
            if steps > 0:
                losses.append(episode_loss / steps)

            if i % 1000 == 0:
                avg_reward = reward_per_episode[max(0, i - 1000):i + 1].mean() if i > 0 else total_reward
                avg_loss = np.mean(losses[-1000:]) if losses else 0
                print(f"Episode {i:>6}/{train_episodes} | "
                      f"ε={self.epsilon:.4f} | "
                      f"Avg reward (last 1k): {avg_reward:.2f} | "
                      f"Steps: {steps} | "
                      f"Loss: {avg_loss:.4f}")

        print(f"\nTraining terminé — {train_episodes} épisodes")
        print(f"  Mean reward : {reward_per_episode.mean():.2f}")
        print(f"  Mean steps  : {steps_per_episode.mean():.2f}")
        print(f"  Mean penalties : {penalties_per_episode.mean():.2f}")

        if self.epsilon > 0.05:
            print(f"  ATTENTION : epsilon final = {self.epsilon:.4f} (> 0.05). "
                  f"epsilon_decay={self.epsilon_decay} est probablement trop lent "
                  f"pour {train_episodes} épisodes.")

        if training_graph:
            self._plot_training(reward_per_episode, steps_per_episode, losses)

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
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(
                        self._encode_state(state)).unsqueeze(0).to(self.device)
                    action = self.policy_net(state_tensor).argmax().item()

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

    def save(self, path="dqn_model.pth"):
        """Sauvegarde le modèle entraîné."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"Modèle sauvegardé : {path}")

    def load(self, path="dqn_model.pth"):
        """Charge un modèle pré-entraîné."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Modèle chargé : {path}")

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

    def _plot_training(self, rewards, steps, losses):
        """Graphiques de convergence post-entraînement."""
        window = 500
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Rewards
        avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(avg_rewards, linewidth=0.8)
        axes[0].set_title(f"DQN — Mean Reward (rolling {window})")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(True, alpha=0.3)

        # Steps
        avg_steps = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(avg_steps, linewidth=0.8, color='orange')
        axes[1].set_title(f"DQN — Mean Steps (rolling {window})")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].grid(True, alpha=0.3)

        # Loss
        if losses:
            loss_window = min(window, len(losses))
            avg_loss = np.convolve(losses, np.ones(loss_window) / loss_window, mode='valid')
            axes[2].plot(avg_loss, linewidth=0.8, color='red')
            axes[2].set_title(f"DQN — Mean Loss (rolling {loss_window})")
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Loss")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("dqn_training_metrics.png", dpi=150)
        plt.show()
        print("Graphique sauvegardé : dqn_training_metrics.png")