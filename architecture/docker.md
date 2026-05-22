# Rapport technique — Dockerisation du projet Taxi Driver

## 1. Contexte et objectif

Le projet Taxi Driver fonctionne dans un environnement Python avec plusieurs dépendances (Gymnasium, NumPy, Matplotlib, PyTorch). Lors du développement, plusieurs problèmes de compatibilité ont été rencontrés : le PEP 668 bloquant les installations pip sur Ubuntu 24, l'absence de `python3-tk` pour le rendu matplotlib, les conflits de versions PyTorch, et la nécessité de créer un venv manuellement à chaque nouvelle machine.

L'objectif de la dockerisation est de garantir la reproductibilité de l'exécution indépendamment de la machine hôte. Un collaborateur doit pouvoir lancer le projet en deux commandes sans installer Python ni aucune dépendance.

## 2. Architecture Docker

### 2.1 Fichiers créés

Quatre fichiers ont été ajoutés à la racine du projet :

| Fichier | Rôle |
|---------|------|
| `Dockerfile` | Définition de l'image (base, dépendances, config) |
| `docker-compose.yml` | Orchestration des 5 services (un par algorithme) |
| `requirements.txt` | Versions fixées des dépendances Python |
| `.dockerignore` | Exclusion des fichiers inutiles pour alléger l'image |

### 2.2 Arborescence résultante

```
TAXI-DRIVER/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── bruteforce.py
├── Q_learning/
│   ├── main.py
│   └── q_learning.py
├── Sarsa/
│   ├── main.py
│   └── sarsa.py
├── MonteCarlo/
│   ├── main.py
│   └── monte_carlo.py
└── deep_Q_learning/
    ├── main.py
    └── deep_q_learning.py
```

## 3. Dockerfile — Détail de l'image

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-tk && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

ENTRYPOINT ["python3"]
```

### 3.1 Choix techniques

**Image de base : `python:3.12-slim`**
L'image slim est basée sur Debian avec le strict minimum. Elle pèse ~120 Mo contre ~900 Mo pour l'image complète. Python 3.12 est choisi pour sa compatibilité avec toutes les dépendances du projet.

**Installation de `python3-tk`**
Nécessaire pour que matplotlib puisse fonctionner même si le backend Agg est utilisé par défaut. Sans ce paquet, l'import de matplotlib échoue dans certaines configurations. Le `rm -rf /var/lib/apt/lists/*` supprime le cache apt pour réduire la taille de l'image.

**`COPY requirements.txt` avant `COPY .`**
Cette séparation exploite le cache Docker : tant que `requirements.txt` ne change pas, le layer `pip install` est mis en cache. Seul le code source est recopié à chaque build, ce qui accélère les rebuilds successifs.

**`--no-cache-dir`**
Empêche pip de stocker les paquets téléchargés dans le cache du container, réduisant la taille finale de l'image.

**`PYTHONUNBUFFERED=1`**
Force Python à écrire directement dans stdout sans buffering. Sans cette variable, les logs d'entraînement (affichés tous les 1 000 épisodes) ne s'affichent qu'à la fin de l'exécution au lieu d'apparaître en temps réel.

**`MPLBACKEND=Agg`**
Force matplotlib en mode non-interactif. Un container Docker n'a pas de serveur graphique (pas de display X11). Sans cette variable, matplotlib tente d'ouvrir une fenêtre et crashe. En mode Agg, les graphiques de convergence sont sauvegardés en PNG au lieu d'être affichés.

**`ENTRYPOINT ["python3"]`**
Permet de passer le script à exécuter directement en argument : `docker compose run --rm q-learning Q_learning/main.py`.

## 4. Requirements — Gestion des dépendances

```
gymnasium[toy-text]==1.3.0
numpy==2.4.6
matplotlib==3.10.9
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.5.1+cpu
```

### 4.1 Choix techniques

**Versions fixées**
Chaque dépendance est épinglée à une version exacte (`==`) pour garantir la reproductibilité. Un `pip install gymnasium` sans version pourrait installer une version future incompatible.

**`gymnasium[toy-text]`**
L'extra `[toy-text]` installe pygame, nécessaire pour le rendu visuel de l'environnement Taxi-v3 via `env.render()`. Sans pygame, Gymnasium 1.3.0 lève une `DependencyNotInstalled` lors du rendu.

**PyTorch CPU (`torch==2.5.1+cpu`)**
La version complète de PyTorch inclut les drivers CUDA et pèse plus de 2 Go. La version CPU-only pèse ~200 Mo. Sur un environnement à 500 états discrets, un GPU n'apporte aucun gain de performance. L'index `--extra-index-url https://download.pytorch.org/whl/cpu` pointe vers le dépôt PyTorch dédié aux builds CPU.

**Version `2.5.1+cpu` au lieu de `2.12.0`**
La version 2.12.0 complète causait des conflits de dépendances dans l'image Docker slim. La version 2.5.1+cpu est la dernière version stable disponible sur l'index CPU PyTorch compatible avec Python 3.12.

## 5. Docker Compose — Orchestration des services

```yaml
services:
  q-learning:
    build: .
    command: ["Q_learning/main.py"]
    volumes:
      - ./Q_learning:/app/Q_learning
    stdin_open: true
    tty: true

  sarsa:
    build: .
    command: ["Sarsa/main.py"]
    volumes:
      - ./Sarsa:/app/Sarsa
    stdin_open: true
    tty: true

  montecarlo:
    build: .
    command: ["MonteCarlo/main.py"]
    volumes:
      - ./MonteCarlo:/app/MonteCarlo
    stdin_open: true
    tty: true

  dqn:
    build: .
    command: ["deep_Q_learning/main.py"]
    volumes:
      - ./deep_Q_learning:/app/deep_Q_learning
    stdin_open: true
    tty: true

  bruteforce:
    build: .
    command: ["bruteforce.py"]
    stdin_open: true
    tty: true
```

### 5.1 Choix techniques

**Un service par algorithme**
Chaque algorithme est isolé dans son propre service. Cela permet de lancer un seul algo à la fois (`docker compose run --rm q-learning`) sans interférence.

**`volumes`**
Chaque service monte son dossier local en volume dans le container. Les fichiers générés pendant l'entraînement (graphiques PNG, modèles `.pth`) sont ainsi directement accessibles sur la machine hôte sans copie manuelle.

**`stdin_open: true` + `tty: true`**
Ces options sont indispensables pour la saisie interactive. Sans elles, les `input()` dans les `main.py` (choix du mode, nombre d'épisodes, affichage visuel) bloquent immédiatement avec un `EOFError`.

**Image partagée**
Tous les services utilisent la même image Docker (`build: .`). Docker Compose ne la construit qu'une fois, les 5 services la réutilisent.

## 6. Dockerignore — Optimisation de l'image

```
.venv/
__pycache__/
*.pyc
*.pyo
*.pth
.git/
.gitignore
*.md
LICENSE
TAXI-DRIVER_AI.pdf
img/
```

Le `.dockerignore` exclut tous les fichiers inutiles à l'exécution : l'environnement virtuel local (`.venv/`), le cache Python, l'historique git, la documentation, le rapport PDF et les images. Seul le code source Python et le `requirements.txt` sont embarqués dans l'image.

Sans `.dockerignore`, le `COPY . .` du Dockerfile copierait le `.venv/` (plusieurs centaines de Mo) et le `.git/` dans l'image, augmentant sa taille inutilement.

## 7. Utilisation

### 7.1 Build

```bash
docker compose build
```

Construit l'image Docker une seule fois. Les builds suivants sont quasi-instantanés grâce au cache des layers.

### 7.2 Lancer un algorithme

```bash
docker compose run --rm q-learning
docker compose run --rm sarsa
docker compose run --rm montecarlo
docker compose run --rm dqn
docker compose run --rm bruteforce
```

Le flag `--rm` supprime automatiquement le container après exécution pour ne pas accumuler des containers arrêtés.

### 7.3 Récupérer les fichiers générés

Les graphiques de convergence et modèles entraînés sont disponibles directement dans les dossiers locaux grâce aux volumes :

```
Q_learning/training_metrics.png
Sarsa/sarsa_training_metrics.png
MonteCarlo/monte_carlo_training_metrics.png
deep_Q_learning/dqn_training_metrics.png
deep_Q_learning/dqn_model.pth
```

## 8. Tests et résultats

### 8.1 Validation

Chaque service a été testé individuellement en mode time-limited :

| Service | Statut | Steps | Reward | Temps |
|---------|--------|-------|--------|-------|
| bruteforce | ✅ | 197.60 | -774.21 | 2.1s |
| q-learning | ✅ | 13.64 | +7.36 | 8.13s |
| sarsa | ✅ | 13.72 | +7.28 | ~8s |
| montecarlo | ✅ | 95.60 | -83.84 | ~53s |
| dqn | ✅ | 12.56 | +8.44 | ~480s |

### 8.2 Problèmes rencontrés et solutions

| Problème | Cause | Solution |
|----------|-------|----------|
| `Dockerfile cannot be empty` | Fichier créé vide via l'IDE | Réécriture du contenu via `cat >` |
| `pip install failed` | PyTorch 2.12.0 trop lourd pour l'image slim | Passage à `torch==2.5.1+cpu` via l'index CPU |
| `ModuleNotFoundError: pygame` | `gymnasium[toy-text]` non installé | Ajout de l'extra `[toy-text]` dans requirements.txt |
| `DependencyNotInstalled: pygame` | Affichage visuel activé par défaut | Répondre "n" à l'affichage dans un container |

## 9. Workflow Git

La dockerisation a été réalisée sur une branche dédiée `features/docker`, la branche a été mergée sur `main` après validation de tous les services.