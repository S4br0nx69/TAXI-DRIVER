# Rapport technique : Mise en place MLOps du projet Taxi Driver

## 1. Contexte et objectif

Le projet Taxi Driver comparait 5 algorithmes d'apprentissage par renforcement
(Brute-force, Q-Learning, SARSA, Monte Carlo, DQN) via des scripts interactifs
(`main.py`) lancés à la main. Les hyperparamètres étaient codés en dur dans
chaque agent, les résultats imprimés dans le terminal sans persistance, et le
modèle DQN sauvegardé dans un `.pth` isolé sans métadonnées.

L'objectif de la mise en place MLOps est d'industrialiser le cycle de vie des
modèles : **reproductibilité**, **traçabilité des expériences**, **versioning
des modèles**, **promotion automatique** du meilleur, **serving** via une API,
et **automatisation** dans la CI. Le tout reste **100 % local**, aucun serveur
distant, aucun appel réseau sortant pour rester cohérent avec la posture de
sécurité du projet (voir `SECURITY.md`).

La mise en place a été menée **phase par phase**, chaque phase validée par
exécution réelle dans un container avant de passer à la suivante.

## 2. Architecture de la couche MLOps

Toute la logique transverse est regroupée dans le package `mlops/`, sans
modifier le code des agents existants (importés dynamiquement) :

```
mlops/
├── __init__.py        # registre ALGORITHMS + sémantique des actions
├── config.yaml        # config centralisée (hyperparamètres par algo)
├── config.py          # résolution defaults <- algo <- overrides CLI
├── seeding.py         # seeds globaux (random / numpy / torch / env)
├── tracking.py        # helpers MLflow (backend fichier local)
├── train.py           # entrypoint d'entraînement unifié instrumenté
├── registry.py        # model registry fichier + versioning + promotion
├── pipeline.py        # orchestration train -> register -> select
└── serve/
    └── app.py         # API d'inférence FastAPI
```

| Fichier ajouté | Rôle |
|----------------|------|
| `requirements-mlops.txt` | dépendances d'entraînement instrumenté (mlflow, pyyaml) |
| `requirements-serve.txt` | dépendances de l'API (fastapi, uvicorn, numpy) |
| `Dockerfile.serve` | image légère de serving (sans torch ni gymnasium) |
| `Makefile` | orchestration des commandes via Docker |
| `tests/test_mlops.py` | 8 tests unitaires de la couche MLOps |

## 3. Phase 1 : Reproductibilité & configuration

### 3.1 Seeds globaux (`seeding.py`)

`set_global_seeds(seed, env)` fixe tous les générateurs pseudo-aléatoires
impliqués : `random` (epsilon-greedy, `action_space.sample`), NumPy, PyTorch
(import paresseux, seul le DQN en dépend) et l'environnement Gymnasium
(`env.action_space.seed` + `env.reset(seed=...)`). Deux runs avec le même seed
produisent désormais des résultats identiques, condition de base du MLOps.

### 3.2 Configuration centralisée (`config.yaml` + `config.py`)

Les hyperparamètres, auparavant codés en dur dans chaque agent, sont centralisés
dans `config.yaml`. `resolve(algorithm, overrides)` applique une précédence
claire : **defaults globaux < hyperparamètres de l'algo < overrides CLI**. Les
valeurs `None` (options CLI non fournies) sont ignorées pour ne pas écraser les
défauts.

## 4. Phase 2 : Suivi d'expériences (MLflow)

### 4.1 Backend fichier local

`tracking.py` configure MLflow avec un **store fichier** (`file:./mlruns`) :
aucun serveur de tracking distant. L'UI se consulte hors-ligne avec
`mlflow ui --backend-store-uri file:./mlruns` (service `mlflow-ui` du compose,
port 5000).

### 4.2 Entrypoint unifié (`train.py`)

`python -m mlops.train <algo>` :
1. résout la config et importe dynamiquement l'agent (registre `ALGORITHMS`) ;
2. applique les hyperparamètres et sème tous les RNG ;
3. ouvre un run MLflow, logge les **params** ;
4. entraîne, mesure le temps, évalue en test ;
5. logge les **métriques** (`mean_reward`, `mean_steps`, `mean_penalties`,
   `train_cvar_95`, `final_epsilon`, `train_time_s`) ;
6. logge un **artefact** (courbe de reward lissée).

Le choix `GIT_PYTHON_REFRESH=quiet` neutralise l'introspection git de MLflow,
inutile et bruyante en container.

## 5. Phase 3 : Model registry & versioning

### 5.1 Pourquoi un registry fichier maison

Le Model Registry natif de MLflow exige un backend base de données
(sqlite/MySQL/…), **incompatible avec un store 100 % fichier**. On implémente
donc un registry fichier inspectable sous `models/<algo>/`, avec un index
`registry.json` (liste des versions + champion courant) et un dossier `vN/` par
version (`policy.npy`, modèle brut, `metadata.json`).

### 5.2 Policy greedy uniforme la clé du serving léger

`extract_policy(agent)` extrait, quel que soit l'algorithme, une **policy
greedy** : un vecteur `[n_states]` donnant l'action optimale par état discret.
- agents tabulaires : `argmax(q_table, axis=1)` ;
- DQN : argmax du réseau évalué sur l'encodage one-hot des 500 états.

Conséquence majeure : le serving n'a besoin que de `policy.npy` (4 Ko) + numpy ,
**ni torch, ni l'architecture du réseau**, quel que soit l'algorithme d'origine.

### 5.3 Promotion automatique

Une version devient **champion** (`stage=Production`) si :
1. elle passe les **garde-fous** (`mean_steps ≤ 30`, `mean_penalties = 0`), ET
2. elle améliore la métrique de promotion (`mean_reward`) par rapport au
   champion courant (le premier modèle valide est promu d'office).

L'ancien champion est rétrogradé. Ces seuils sont déclaratifs dans
`config.yaml` (section `promotion`).

## 6. Phase 4 : Serving (API FastAPI)

### 6.1 Image dédiée et légère

`Dockerfile.serve` produit une image minimale (fastapi + uvicorn + numpy +
pyyaml, **sans torch ni gymnasium**), ne copiant que le package `mlops/`. Le
dossier `models/` est monté en volume **lecture seule** au runtime.

### 6.2 Endpoints

| Méthode | Route | Rôle |
|---------|-------|------|
| GET | `/health` | liveness + indication du modèle chargé |
| GET | `/model/info` | métadonnées du champion (version, métriques, params, run) |
| POST | `/predict` | `{"state": int}` → action greedy + nom de l'action |
| POST | `/reload` | recharge le champion à chaud (après un retraining) |

La variable `TAXI_SERVE_ALGORITHM` choisit le modèle servi : un algo précis, ou
`auto` (meilleur champion tous algos confondus, via `registry.best_algorithm`).
Les états hors bornes renvoient `422`, l'absence de champion `503`.

## 7. Phase 5 : Pipeline & CI/CD

### 7.1 Pipeline d'orchestration (`pipeline.py`)

`python -m mlops.pipeline` enchaîne l'entraînement+enregistrement de plusieurs
algorithmes, affiche un **leaderboard** trié par `mean_reward`, puis désigne le
**champion global** (celui que sert l'API).

### 7.2 Makefile

Toutes les opérations passent par Docker, sans dépendance Python sur l'hôte :
`make build | train | pipeline | serve | mlflow-ui | test | clean`.

### 7.3 CI GitHub Actions

Un job `mlops` a été ajouté à `.github/workflows/ci.yml` :
1. installe les dépendances MLOps ;
2. exécute les tests unitaires `tests/test_mlops.py` ;
3. entraîne+enregistre SARSA (converge sous le garde-fou de façon fiable) ;
4. **smoke test de serving** : démarre l'API (TestClient), vérifie `/health`,
   une prédiction valide (`200`) et le rejet d'un état hors bornes (`422`).

## 8. Utilisation

```bash
# Build des images MLOps
make build

# Entraîner + enregistrer un algo
make train ALGO=sarsa EPISODES=10000

# Pipeline complet (qlearning + sarsa + montecarlo) -> champion global
make pipeline

# Explorer les runs dans l'UI MLflow (http://localhost:5000)
make mlflow-ui

# Servir le champion (http://localhost:8000)
make serve
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' -d '{"state": 328}'
```

## 9. Tests et résultats

| Validation | Statut |
|------------|--------|
| Déterminisme des seeds (random/numpy/env) | ✅ |
| Résolution de config + overrides | ✅ |
| Tracking MLflow (params/métriques/artefacts dans `mlruns/`) | ✅ |
| Versioning + promotion + remplacement de champion | ✅ |
| Extraction de policy greedy (tabular + DQN) | ✅ |
| API de serving (HTTP live + TestClient, bornes 422/503) | ✅ |
| Pipeline + leaderboard + champion global | ✅ |
| Suite pytest complète | ✅ 15 tests |

### 8.1 Choix techniques notables

| Décision | Justification |
|----------|---------------|
| MLflow en backend **fichier** | 100 % local, aucun serveur ni réseau |
| Registry **fichier maison** | le registry MLflow natif exige une base de données |
| **Policy greedy** comme artefact de serving | serving uniforme et torch-free (4 Ko/modèle) |
| Image de serving **séparée** | pas de torch/gymnasium → image légère et surface réduite |
| Agents **non modifiés** | import dynamique : la couche MLOps reste découplée |

## 10. Workflow Git

La mise en place MLOps a été réalisée sur une branche dédiée `MLops`,
phase par phase, chaque phase validée par exécution en container avant la
suivante.
