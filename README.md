<div align="center">

# 🚕 Taxi Driver

**Reinforcement Learning agent for Gymnasium Taxi-v3**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3.0-0081A5?style=for-the-badge)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.9-11557C?style=for-the-badge)](https://matplotlib.org)

<br>

R�solution de l'environnement **Taxi-v3** par apprentissage par renforcement model-free.<br>
5 approches implémentées et comparées : **Brute-force**, **Q-Learning**, **SARSA**, **Monte Carlo** et **Deep Q-Learning (DQN)**.

<br>

[Installation](#-installation) •
[Utilisation](#-utilisation) •
[Algorithmes](#-algorithmes) •
[Benchmark](#-benchmark) •
[Architecture](#-architecture)

</div>

---

## 📦 Installation

> [!IMPORTANT]
> Requiert **Python 3.11+** et `pip`.

```bash
git clone git@github.com:S4br0nx69/TAXI-DRIVER.git
cd TAXI-DRIVER
python3 -m venv .venv
source .venv/bin/activate
pip install gymnasium numpy matplotlib torch
```

---

## 🚀 Utilisation

Chaque algorithme se lance depuis son dossier :

```bash
source .venv/bin/activate

# Q-Learning
cd Q_learning && python3 main.py

# SARSA
cd Sarsa && python3 main.py

# Monte Carlo
cd MonteCarlo && python3 main.py

# Deep Q-Learning
cd deep_Q_learning && python3 main.py

# Brute-force (baseline)
cd .. && python3 bruteforce.py
```

### Modes d'exécution

Au lancement, le programme propose deux modes :

```
╔══════════════════════════════════╗
║     TAXI DRIVER — Q-Learning     ║
╚══════════════════════════════════╝

1. Mode utilisateur (réglage des hyperparamètres)
2. Mode time-limited (paramètres optimisés)

Choix [1/2] :
```

<details>
<summary><b>Mode 1 — Utilisateur</b></summary>

L'utilisateur saisit chaque hyperparamètre manuellement (α, γ, ε, decay, etc.). Une valeur par défaut est proposée entre crochets — Entrée l'applique directement.

```
Nombre d'épisodes d'entraînement [25000] :
Nombre d'épisodes de test [25] :
Learning rate α [0.1] :
Discount factor γ [0.6] :
Epsilon initial [1.0] :
Epsilon minimum [0.01] :
Epsilon decay [0.9995] :
Afficher les épisodes de test ? (o/n) [o] :
```

</details>

<details>
<summary><b>Mode 2 — Time-limited</b></summary>

Paramètres pré-optimisés. L'utilisateur choisit uniquement le nombre d'épisodes et l'affichage visuel.

```
Nombre d'épisodes d'entraînement [10000] :
Nombre d'épisodes de test [25] :
Afficher les épisodes de test ? (o/n) [o] :

Paramètres optimisés : α=0.1, γ=0.6, ε=1.0→0.01, decay=0.9995
```

</details>

### Affichage visuel

Lorsque l'affichage est activé (`o`), la grille Taxi-v3 est rendue en temps réel via matplotlib à chaque step de chaque épisode de test.

### Outputs

Le programme affiche :
- Les logs d'entraînement tous les 1 000 épisodes (ε, reward moyen, steps)
- Le bilan du training (mean reward, mean steps, mean penalties)
- Les graphiques de convergence (sauvegardés en PNG)
- Les résultats de test (average steps, penalties, reward)
- Le temps d'exécution total
- Le CVaR à 95% de niveau de confiance

---

## 🧠 Algorithmes

<details>
<summary><b>Brute-force (baseline)</b></summary>

Agent aléatoire sans apprentissage. Sert de plancher de performance.
- **Résultat** : 196 steps, -769 reward, 0% de succès
- **Fichier** : `bruteforce.py` (racine du projet)

</details>

<details>
<summary><b>Q-Learning tabulaire</b></summary>

Algorithme off-policy. Q-table 500×6 mise à jour par la formule de Bellman à chaque step.
- **Hyperparamètres** : α=0.1, γ=0.6, ε decay 1.0→0.01
- **Résultat** : 27.84 steps, -8.52 reward, 0 pénalités (10k épisodes)
- **Fichiers** : `Q_learning/q_learning.py`, `Q_learning/main.py`

</details>

<details>
<summary><b>SARSA</b></summary>

Algorithme on-policy. Mise à jour avec l'action réellement choisie au step suivant.
- **Hyperparamètres** : α=0.2, γ=0.9, ε decay 1.0→0.01
- **Résultat** : 13.72 steps, +7.28 reward, 0 pénalités (10k épisodes)
- **Fichiers** : `Sarsa/sarsa.py`, `Sarsa/main.py`

</details>

<details>
<summary><b>Monte Carlo (first-visit)</b></summary>

Algorithme épisodique pur. Mise à jour en fin d'épisode avec le retour cumulé réel.
- **Hyperparamètres** : α=0.05, γ=0.95, ε decay 1.0→0.01
- **Résultat** : 95.60 steps, -83.84 reward, 0 pénalités (50k épisodes)
- **Fichiers** : `MonteCarlo/monte_carlo.py`, `MonteCarlo/main.py`

</details>

<details>
<summary><b>Deep Q-Learning (DQN)</b></summary>

R�seau de neurones PyTorch (128→64, ReLU) avec experience replay et target network.
- **Hyperparamètres** : lr=0.001, γ=0.99, ε decay 1.0→0.01, batch=64
- **Résultat** : 12.56 steps, +8.44 reward, 0 pénalités (10k épisodes)
- **Fichiers** : `deep_Q_learning/deep_q_learning.py`, `deep_Q_learning/main.py`
- **Modèle** : sérialisé dans `dqn_model.pth` après entraînement

</details>

---

## 📊 Benchmark

| Algorithme | Épisodes | Mean Steps | Mean Reward | Pénalités | Temps |
|---|---|---|---|---|---|
| 🔴 Brute-force | — | 196.42 | -769.59 | 63.78 | 1.9s |
| 🟡 Monte Carlo | 50 000 | 95.60 | -83.84 | 0.00 | 53s |
| 🔵 Q-Learning | 10 000 | 27.84 | -8.52 | 0.00 | 27s |
| 🟢 SARSA | 10 000 | 13.72 | +7.28 | 0.00 | 27s |
| 🟣 DQN | 10 000 | 12.56 | +8.44 | 0.00 | 476s |

> Le DQN obtient les meilleurs résultats (12.56 steps, +8.44 reward) mais SARSA offre le meilleur rapport performance/coût (97% de la performance du DQN en 17× moins de temps).

---

## 🏗 Architecture

```
TAXI-DRIVER/
├── Q_learning/
│   ├── main.py              # Point d'entrée (2 modes)
│   └── q_learning.py        # Classe Taxi — Q-Learning tabulaire
├── Sarsa/
│   ├── main.py
│   └── sarsa.py             # Classe Sarsa — algorithme on-policy
├── MonteCarlo/
│   ├── main.py
│   └── monte_carlo.py       # Classe MonteCarlo — first-visit
├── deep_Q_learning/
│   ├── main.py
│   └── deep_q_learning.py   # Classe DQNAgent — PyTorch
├── bruteforce.py             # Baseline random agent
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🎯 Système de récompenses (Taxi-v3)

| Événement | Reward |
|-----------|--------|
| Step standard | `-1` |
| Livraison réussie | `+20` |
| Pickup / drop-off illégal | `-10` |

> **State space** : 500 états discrets `(taxi_row, taxi_col, passenger_loc, destination)`
> **Action space** : 6 actions `(South, North, East, West, Pickup, Dropoff)`

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

**Sabri** — Taxi Driver v1.0*

</div>