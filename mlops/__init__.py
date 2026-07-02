"""Couche MLOps du projet Taxi Driver.

Regroupe les briques transverses à tous les algorithmes RL :
- reproductibilité (seeds globaux)
- configuration centralisée des hyperparamètres
- suivi d'expériences (MLflow, backend fichier local)
- registry / versioning des modèles
- serving (API FastAPI)
- pipeline d'orchestration (train -> eval -> register)

Tout est 100% local : aucun appel réseau, aucun serveur distant.
"""

__all__ = [
    "ALGORITHMS",
    "ACTION_NAMES",
]

# Registre des algorithmes : nom -> (sous-dossier, module, classe).
# Source unique de vérité utilisée par train.py, pipeline.py et serve.
ALGORITHMS = {
    "qlearning": ("Q_learning", "q_learning", "Taxi"),
    "sarsa": ("Sarsa", "sarsa", "Sarsa"),
    "montecarlo": ("MonteCarlo", "monte_carlo", "MonteCarlo"),
    "dqn": ("deep_Q_learning", "deep_q_learning", "DQNAgent"),
}

# Sémantique des 6 actions discrètes de Taxi-v3/v4.
ACTION_NAMES = ["South", "North", "East", "West", "Pickup", "Dropoff"]
