<div align="center">

# 🚕 Taxi Driver

**Reinforcement Learning agent for Gymnasium Taxi-v3**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3.0-0081A5?style=for-the-badge)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.9-11557C?style=for-the-badge)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

R�solution de l'environnement **Taxi-v3** par apprentissage par renforcement model-free.<br>
Deux approches implémentées : **Q-Learning tabulaire** et **Deep Q-Learning (DQN)**.

<br>

[Installation](#-installation) •
[Q-Learning](#-q-learning-tabulaire) •
[Deep Q-Learning](#-deep-q-learning-dqn) •
[Benchmark](#-benchmark) •
[Architecture](#-architecture)

</div>

---

## 📦 Installation

> [!IMPORTANT]
> Requiert **Python 3.11+** et `pip`.

```bash
git clone <repo-url> && cd taxi-driver
pip3 install -r requirements.txt
```

<details>
<summary><b>📋 Dépendances détaillées</b></summary>

| Package | Version | Usage |
|---------|---------|-------|
| `gymnasium` | `1.3.0` | Environnement Taxi-v3 |
| `numpy` | `2.4.6` | Q-table, calculs matriciels |
| `matplotlib` | `3.10.9` | Visualisation, graphiques de benchmark |
| `torch` | `2.12.0` | Réseau de neurones (DQN) |

</details>

---

## 🏗 Architecture

```
taxi-driver/
├── Q_learning/
│   └── q_learning.py           # Agent Q-Learning tabulaire
├── deep_Q_learning/
│   └── deep_q_learning.py      # Agent DQN (PyTorch)
├── img/                        # Captures & graphiques
├── requirements.txt
└── README.md
```

---

## 🧠 Q-Learning tabulaire

### Quickstart

```python
import q_learning as Taxi

taxi = Taxi.Taxi('rgb_array')
taxi.train(train_episodes=35000)
taxi.test(test_episodes=5, timestamp=0.1, fast_testing=False, final_frame_pause=0)
```

### API Reference

<details>
<summary><code>Taxi(render_mode)</code> — Constructeur</summary>

| Paramètre | Type | Description |
|-----------|------|-------------|
| `render_mode` | `str` | `"rgb_array"` → fenêtre matplotlib · `"ansi"` → rendu terminal |

</details>

<details>
<summary><code>taxi.train(**kwargs)</code> — Entraînement</summary>

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `train_episodes` | `int` | `25000` | Nombre d'épisodes d'entraînement |

</details>

<details>
<summary><code>taxi.test(**kwargs)</code> — Évaluation</summary>

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `test_episodes` | `int` | `1` | Nombre d'épisodes de test |
| `timestamp` | `float` | `0.2` | Délai (s) entre chaque action |
| `fast_testing` | `bool` | `False` | Désactive le rendu graphique |
| `final_frame_pause` | `float` | `0` | Pause (s) sur la dernière frame |

</details>

### Outputs temps réel

![Fenêtre de test](./img/testing_execution_window.png)

| Métrique | Description |
|----------|-------------|
| `Action` | Dernière action exécutée |
| `Reward` | Récompense instantanée |
| `Episode reward` | Récompense cumulée de l'épisode |
| `Episode` | Index de l'épisode en cours |

### Résultat final

![Résultats](./img/final_result.png)

> Affiche le **mean steps**, **mean penalties** et **mean reward** sur tous les épisodes de test.

---

## 🔥 Deep Q-Learning (DQN)

### Quickstart

```python
import deep_q_learning as Taxi

taxi = Taxi.QAgent()
taxi.compile()
taxi.fit()
```

### API Reference

<details>
<summary><code>QAgent.__init__()</code> — Hyperparamètres</summary>

| Variable | Type | Description |
|----------|------|-------------|
| `model_class` | `nn.Module` | Architecture du réseau de neurones |
| `memory` | `ReplayBuffer` | Gestionnaire du replay buffer (experience replay) |
| `loss` | `callable` | Fonction de perte (`HuberLoss` par défaut) |
| `lr` | `float` | Learning rate |
| `gamma` | `float` | Facteur de discount |
| `epsilon_decay` | `float` | Décroissance de l'exploration ε-greedy |
| `batch_size` | `int` | Taille des mini-batches |

</details>

### Monitoring

L'entraînement produit un dashboard matplotlib en temps réel :

![Métriques DQN](./img/Metrics%20graph%20deep%20Q%20learning%20(3%20layers%20V2).png)

> [!NOTE]
> Le modèle est sérialisé automatiquement à la fin du training. Pas de fonction de test graphique dédiée — les métriques en temps réel et la barre de progression couvrent le suivi de convergence.

---

## 🎯 Système de récompenses

> Environnement : `Taxi-v3` — Grille 5×5, 4 destinations (R, G, Y, B), 6 actions.

| Événement | Reward |
|-----------|--------|
| Step standard | `-1` |
| Livraison réussie | `+20` |
| Pickup / drop-off illégal | `-10` |

> **State space :** 500 états discrets `(taxi_row, taxi_col, passenger_loc, destination)`
> **Action space :** 6 actions `(South, North, East, West, Pickup, Dropoff)`

---

## 📊 Benchmark

| Algorithme | Mean Steps | Mean Reward | Convergence |
|------------|-----------|-------------|-------------|
| 🔴 Brute-force (random) | ~350 | — | — |
| 🟡 Q-Learning (baseline) | ~50 | — | ~5k épisodes |
| 🟢 Q-Learning (optimisé) | ~13 | ~7.5 | ~15k épisodes |
| 🔵 Deep Q-Learning (DQN) | ~15 | ~7.0 | ~10k épisodes |

> [!TIP]
> Les valeurs ci-dessus sont indicatives. Consultez le rapport de benchmark pour les résultats détaillés avec variance et intervalles de confiance.

<details>
<summary><b>📈 Métriques de benchmark collectées</b></summary>

- Mean reward per episode
- Mean steps per episode
- Mean penalties (illegal actions) per episode
- Convergence speed (épisodes avant stabilisation)
- Reward variance & écart-type
- Temps d'entraînement (wall-clock)
- Impact des hyperparamètres (α, γ, ε) sur la convergence

</details>

---

## ⚙️ Stack technique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Langage | Python | `3.11+` |
| Environnement RL | Gymnasium | `1.3.0` |
| Calcul numérique | NumPy | `2.4.6` |
| Visualisation | Matplotlib | `3.10.9` |
| Deep Learning | PyTorch | `2.12.0` |

---

<div align="center">

**Epitech** — Taxi Driver v1.4

</div>