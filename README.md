<div align="center">

# 🚕 Taxi Driver

**Reinforcement Learning agent for Gymnasium Taxi-v3**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3.0-0081A5?style=for-the-badge)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.4.6-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.9-11557C?style=for-the-badge)](https://matplotlib.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Portainer](https://img.shields.io/badge/Portainer-CE-13BEF9?style=for-the-badge&logo=portainer&logoColor=white)](https://portainer.io)
[![CI](https://github.com/S4br0nx69/TAXI-DRIVER/actions/workflows/ci.yml/badge.svg)](https://github.com/S4br0nx69/TAXI-DRIVER/actions/workflows/ci.yml)

<br>

Résolution de l'environnement **Taxi-v3** par apprentissage par renforcement model-free.<br>
5 approches implémentées et comparées : **Brute-force**, **Q-Learning**, **SARSA**, **Monte Carlo** et **Deep Q-Learning (DQN)**.

<br>

[Installation](#-installation) •
[Docker](#-docker) •
[Utilisation](#-utilisation) •
[Algorithmes](#-algorithmes) •
[Benchmark](#-benchmark) •
[Tests & CI/CD](#-tests--cicd) •
[Sécurité](#-sécurité) •
[Architecture](#-architecture)

</div>

---

## 📦 Installation

### Option 1 — Docker (recommandé)

```bash
git clone git@github.com:S4br0nx69/TAXI-DRIVER.git
cd TAXI-DRIVER
docker compose build
docker compose run --rm q-learning
```

### Option 2 — Local

> [!IMPORTANT]
> Requiert **Python 3.11+** et `pip`.

```bash
git clone git@github.com:S4br0nx69/TAXI-DRIVER.git
cd TAXI-DRIVER
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🐳 Docker

Le projet est entièrement containerisé. Chaque algorithme est un service Docker indépendant.

```bash
# Build de l'image
docker compose build

# Lancer un algorithme
docker compose run --rm q-learning
docker compose run --rm sarsa
docker compose run --rm montecarlo
docker compose run --rm dqn
docker compose run --rm bruteforce
```

> [!TIP]
> Répondre **n** à "Afficher les épisodes de test ?" dans Docker (pas de display graphique dans un container).

<details>
<summary><b>📊 Supervision avec Portainer</b></summary>

Portainer CE est intégré pour superviser les containers en temps réel (logs, CPU, RAM).

```bash
# Démarrer Portainer
docker compose up -d portainer

# Accéder à l'interface
# https://localhost:9443
```

Les containers d'entraînement apparaissent dans Portainer pendant leur exécution.

</details>

<details>
<summary><b>📁 Récupérer les fichiers générés</b></summary>

Les graphiques et modèles sont accessibles via les volumes Docker :

```
Q_learning/training_metrics.png
Sarsa/sarsa_training_metrics.png
MonteCarlo/monte_carlo_training_metrics.png
deep_Q_learning/dqn_training_metrics.png
deep_Q_learning/dqn_model.pth
```

</details>

---

## 🚀 Utilisation

Chaque algorithme se lance depuis son dossier :

```bash
source .venv/bin/activate

cd Q_learning && python3 main.py
cd Sarsa && python3 main.py
cd MonteCarlo && python3 main.py
cd deep_Q_learning && python3 main.py
python3 bruteforce.py  # depuis la racine
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

R�glage manuel de chaque hyperparamètre. Valeurs par défaut entre crochets — Entrée les applique. Les entrées sont validées avec bornes de sécurité (pas de crash sur saisie invalide).

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

Paramètres pré-optimisés. L'utilisateur choisit uniquement le nombre d'épisodes.

```
Nombre d'épisodes d'entraînement [10000] :
Nombre d'épisodes de test [25] :
Afficher les épisodes de test ? (o/n) [o] :

Paramètres optimisés : α=0.1, γ=0.6, ε=1.0→0.01, decay=0.9995
```

</details>

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
<summary><b>🔴 Brute-force (baseline)</b></summary>

Agent aléatoire sans apprentissage. Sert de plancher de performance.
- **Résultat** : 196 steps, -769 reward, 0% de succès
- **Fichier** : `bruteforce.py`

</details>

<details>
<summary><b>🔵 Q-Learning tabulaire</b></summary>

Algorithme off-policy. Q-table 500×6 mise à jour par la formule de Bellman à chaque step.
- **Hyperparamètres** : α=0.1, γ=0.6, ε decay 1.0→0.01
- **Résultat** : 27.84 steps, -8.52 reward, 0 pénalités (10k épisodes)
- **Fichiers** : `Q_learning/q_learning.py`, `Q_learning/main.py`

</details>

<details>
<summary><b>🟢 SARSA</b></summary>

Algorithme on-policy. Mise à jour avec l'action réellement choisie au step suivant.
- **Hyperparamètres** : α=0.2, γ=0.9, ε decay 1.0→0.01
- **Résultat** : 13.72 steps, +7.28 reward, 0 pénalités (10k épisodes)
- **Fichiers** : `Sarsa/sarsa.py`, `Sarsa/main.py`

</details>

<details>
<summary><b>🟡 Monte Carlo (first-visit)</b></summary>

Algorithme épisodique pur. Mise à jour en fin d'épisode avec le retour cumulé réel.
- **Hyperparamètres** : α=0.05, γ=0.95, ε decay 1.0→0.01
- **Résultat** : 95.60 steps, -83.84 reward, 0 pénalités (50k épisodes)
- **Fichiers** : `MonteCarlo/monte_carlo.py`, `MonteCarlo/main.py`

</details>

<details>
<summary><b>🟣 Deep Q-Learning (DQN)</b></summary>

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

> **DQN** obtient les meilleurs résultats (12.56 steps, +8.44 reward) mais **SARSA** offre le meilleur rapport performance/coût : 97% de la performance du DQN en 17× moins de temps.

---

## 🧪 Tests & CI/CD

### Tests unitaires

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

<details>
<summary><b>7 tests — détail</b></summary>

| Test | Cible | Vérifie |
|------|-------|---------|
| `test_bruteforce` | `bruteforce.py` | Résultats cohérents (steps > 0, reward < 0, 0% succès) |
| `test_qlearning_train` | `q_learning.py` | `train()` retourne un `np.ndarray` de la bonne taille |
| `test_qlearning_convergence` | `q_learning.py` | Après 10k épisodes : steps < 100, 0 pénalités |
| `test_sarsa_train` | `sarsa.py` | `train()` retourne un `np.ndarray` de la bonne taille |
| `test_montecarlo_train` | `monte_carlo.py` | `train()` retourne un `np.ndarray` de la bonne taille |
| `test_utils_safe_input_int` | `utils.py` | Gère texte, hors bornes, valeur valide |
| `test_utils_safe_input_float` | `utils.py` | Gère texte, hors bornes, valeur valide |

</details>

### CI/CD — GitHub Actions

Chaque push sur `main` et chaque pull request déclenche automatiquement :
- Installation des dépendances (Gymnasium, NumPy, Matplotlib, PyTorch CPU, pytest)
- Exécution des 7 tests unitaires
- Smoke test du brute-force
- Smoke test du Q-Learning (5000 épisodes d'entraînement + validation)

> Le workflow est défini dans [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Grid Search

Optimisation automatisée des hyperparamètres du Q-Learning (48 combinaisons de α, γ, decay) :

```bash
python grid_search.py
```

<details>
<summary><b>Paramètres testés</b></summary>

| Paramètre | Valeurs |
|-----------|---------|
| α (learning rate) | 0.05, 0.1, 0.2, 0.3 |
| γ (discount factor) | 0.6, 0.8, 0.9, 0.99 |
| ε decay | 0.999, 0.9995, 0.9999 |

Le script affiche la meilleure combinaison trouvée avec les steps et reward correspondants.

</details>

---

## 🔒 Sécurité

<details>
<summary><b>Mesures appliquées</b></summary>

- **Désérialisation sécurisée** : `torch.load()` utilise `weights_only=True` pour bloquer l'exécution de code arbitraire via des fichiers `.pth` malveillants
- **Validation des entrées** : toutes les saisies utilisateur sont validées avec bornes (`utils.py`) — pas de crash sur entrée invalide, pas de déni de service par valeurs excessives
- **Isolation Docker** : exécution dans des containers éphémères, aucune installation système requise
- **Pas de secrets en dur** : aucune clé API, token ou credential dans le code source
- **Exécution 100% locale** : aucun appel réseau, aucune API externe

</details>

> Voir [`SECURITY.md`](SECURITY.md) pour la politique de signalement de vulnérabilités.

---

## 🏗 Architecture

```
TAXI-DRIVER/
├── .github/
│   └── workflows/
│       └── ci.yml            # Pipeline CI/CD GitHub Actions
├── tests/
│   └── test_agents.py        # 7 tests unitaires (pytest)
├── Q_learning/
│   ├── main.py               # Point d'entrée (2 modes)
│   └── q_learning.py         # Classe Taxi — Q-Learning tabulaire
├── Sarsa/
│   ├── main.py
│   └── sarsa.py              # Classe Sarsa — algorithme on-policy
├── MonteCarlo/
│   ├── main.py
│   └── monte_carlo.py        # Classe MonteCarlo — first-visit
├── deep_Q_learning/
│   ├── main.py
│   └── deep_q_learning.py    # Classe DQNAgent — PyTorch
├── bruteforce.py              # Baseline random agent
├── grid_search.py             # Optimisation des hyperparamètres
├── utils.py                   # Validation sécurisée des entrées
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── SECURITY.md
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
| Containerisation | Docker | `latest` |
| Supervision | Portainer CE | `latest` |
| Tests | pytest | `9.0.3` |
| CI/CD | GitHub Actions | — |

---

## 🔬 Fine-tuning (branche `features/fine-tunning`)

### Objectif

L'objectif de cette branche est d'aller plus loin que les paramètres par défaut : identifier les meilleurs hyperparamètres pour chaque modèle, mesurer objectivement le gain apporté, et produire un comparatif visuel avant/après utilisable en présentation.

Le travail se décompose en deux axes :
1. **Extension des hyperparamètres** — chaque modèle expose de nouveaux réglages spécifiques à son algorithme
2. **Infrastructure de benchmark** — un script dédié capture les métriques à trois moments clés (baseline, grid search, final) et génère les traces (JSON, PNG, tableau terminal)

---

### Hyperparamètres ajoutés par modèle

#### 🟢 SARSA — `Sarsa/sarsa.py`

| Paramètre | Type | Valeur par défaut | Rôle |
|---|---|---|---|
| `policy_type` | `str` | `'epsilon_greedy'` | Politique de sélection d'action : `epsilon_greedy`, `softmax` (Boltzmann) ou `expected` (Expected SARSA) |
| `temperature` | `float` | `1.0` | Facteur d'échelle pour la politique softmax — plus bas = plus exploitant |
| `n_steps` | `int` | `1` | Horizon de retour : `1` = SARSA standard, `n > 1` = n-step SARSA (Sutton & Barto ch.7) |

**Méthodes ajoutées :**
- `_choose_action(state)` — remplace le code epsilon-greedy inline ; dispatche vers softmax, epsilon-greedy ou expected selon `policy_type`
- `_expected_value(state)` — calcule la valeur espérée sous politique epsilon-greedy, utilisée par Expected SARSA à la place de `Q(s', a')`
- `_train_nstep(train_episodes, training_graph)` — boucle d'entraînement n-step dédiée ; appelée automatiquement par `train()` quand `n_steps > 1`

**Pourquoi :** SARSA est on-policy, ce qui en fait le candidat naturel pour tester des variantes de politique. Expected SARSA réduit la variance en moyennant sur toutes les actions possibles. Le n-step permet de trouver le bon équilibre entre TD (1-step, biais fort) et Monte Carlo (retour complet, variance forte).

---

#### 🟡 Monte Carlo — `MonteCarlo/monte_carlo.py`

| Paramètre | Type | Valeur par défaut | Rôle |
|---|---|---|---|
| `visit_mode` | `str` | `'first_visit'` | `first_visit` = mise à jour uniquement à la première visite d'une paire (s, a) dans l'épisode ; `every_visit` = mise à jour à chaque passage |

**Pourquoi :** En first-visit, chaque paire (s, a) ne contribue qu'une fois au retour G de l'épisode — estimateur non biaisé mais plus lent à converger. Every-visit exploite davantage les données de chaque épisode au prix d'un léger biais sur les états fréquemment revisités.

---

#### 🟣 DQN — `deep_Q_learning/deep_q_learning.py`

| Paramètre | Type | Valeur par défaut | Rôle |
|---|---|---|---|
| `optimizer_type` | `str` | `'adam'` | Optimiseur PyTorch : `adam`, `rmsprop` ou `sgd` |

**Méthode ajoutée :**
- `_build_optimizer()` — instancie l'optimiseur selon `optimizer_type` et `lr` ; appelée dans `__init__` et à réappeler manuellement si les deux paramètres sont modifiés après construction

**Pourquoi :** Adam est le défaut raisonnable pour le DQN, mais RMSprop est historiquement recommandé dans le papier original DeepMind (2013). Exposer ce choix permet de le valider empiriquement sur Taxi.

---

#### 🔵 Q-Learning — `Q_learning/q_learning.py`

Pas de nouvel hyperparamètre individuel : Q-Learning sert de référence algorithmique (off-policy pur, sans variante). Les labels des hyperparamètres ont été harmonisés (`# Hyperparamètres communs`) pour la cohérence avec les autres modèles.

---

### Script de benchmark — `benchmark.py`

Nouveau script racine qui orchestre la capture de métriques en trois phases indépendantes et génère les traces visuelles.

#### Principe des phases

```
baseline  →  snapshot avec params par défaut       →  results/baseline.json
   ↓
grid      →  exploration de l'espace de params      →  results/grid_search.json
   ↓
final     →  snapshot avec meilleurs params         →  results/final.json
   ↓
compare   →  comparatif avant/après                 →  results/plots/comparison.png
```

Chaque phase est rejouable indépendamment. Les résultats persistent en JSON entre les runs.

#### Utilisation

Le benchmark tourne dans Docker via le service dédié `benchmark` défini dans `docker-compose.yml` :

```bash
docker compose run --rm benchmark benchmark.py baseline

docker compose run --rm benchmark benchmark.py grid

docker compose run --rm benchmark benchmark.py final

docker compose run --rm benchmark benchmark.py compare
```

> [!TIP]
> `--no-interactive` désactive les prompts et utilise directement les valeurs CLI — utile pour scripter :
> ```bash
> docker compose run --rm benchmark benchmark.py baseline --train-episodes 2000 --test-episodes 50 --no-interactive
> ```

#### Flow interactif (phases baseline, grid, final)

Au lancement, le script pose deux questions dans l'ordre :

**1. Sélection des modèles** — menu curses navigable au clavier :

```
Modèles à exécuter
  ↑↓ naviguer   Espace cocher/décocher   Entrée confirmer

    [x]  Q-Learning
    [x]  SARSA
    [x]  Monte Carlo
    [ ]  DQN
```

**2. Configuration des épisodes** — une seule saisie, appliquée à tous les modèles cochés :

```
--- Configuration des épisodes (tous les modèles) ---

  Épisodes d'entraînement [10000] :
  Épisodes de test        [100] :
```

Appuyer sur Entrée sans saisir applique la valeur par défaut entre crochets.

#### Espace de recherche du grid search

| Modèle | Paramètres explorés | Combinaisons |
|---|---|---|
| Q-Learning | α × γ × ε_decay | 27 |
| SARSA | α × γ × ε_decay × policy_type × n_steps | 32 |
| Monte Carlo | α × γ × ε_decay × visit_mode | 36 |
| DQN | lr × γ × batch_size × optimizer_type | 16 |

Le critère de sélection est le **reward moyen maximum** sur les épisodes de test.

#### Sorties produites

| Fichier | Contenu |
|---|---|
| `results/baseline.json` | Métriques + params de chaque modèle avant fine-tuning |
| `results/grid_search.json` | Toutes les combinaisons testées + meilleurs params par modèle |
| `results/final.json` | Métriques + params de chaque modèle après fine-tuning |
| `results/plots/baseline.png` | Bar chart des 3 métriques à la baseline |
| `results/plots/final.png` | Bar chart des 3 métriques après fine-tuning |
| `results/plots/comparison.png` | Bar chart avant/après côte à côte pour les 4 modèles |

Le script affiche également un tableau terminal avec les deltas Δ (reward, steps, penalties) permettant de lire le gain en un coup d'œil :

```
======================================================
                  Reward          Steps       Penalties
Modèle         Avant  Après    Δ  Avant  Après    Δ  ...
======================================================
Q-Learning      x.x    x.x  +x.x  xx.x   xx.x  -x.x  ...
SARSA           x.x    x.x  +x.x  xx.x   xx.x  -x.x  ...
...
```

#### Architecture des fichiers de sortie

```
results/
├── baseline.json
├── grid_search.json
├── final.json
└── plots/
    ├── baseline.png
    ├── final.png
    └── comparison.png
```

---

<div align="center">

**Sabri Hammi** — Taxi Driver v2.0

</div>
