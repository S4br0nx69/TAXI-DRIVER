"""Suivi d'expériences via MLflow — backend fichier 100% local.

Aucun serveur de tracking distant : MLflow écrit dans `./mlruns` (ou le
répertoire passé en paramètre). L'UI se lance ensuite hors-ligne avec
`mlflow ui --backend-store-uri ./mlruns`.

Ce module isole le projet de l'API MLflow : si on changeait de backend de
tracking, seul ce fichier serait à modifier.
"""

import os

# MLflow tente d'introspecter le dépôt git via GitPython ; inutile ici et
# bruyant si git est absent du container. À neutraliser avant l'import mlflow.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import matplotlib
matplotlib.use("Agg")  # pas de display dans un container / CI
import matplotlib.pyplot as plt
import mlflow
import numpy as np

DEFAULT_EXPERIMENT = "taxi-driver"


def configure(tracking_dir="mlruns", experiment=DEFAULT_EXPERIMENT):
    """Pointe MLflow vers un store fichier local et sélectionne l'expérience.

    Returns:
        l'URI de tracking absolu (file:...), utile pour l'affichage.
    """
    abs_dir = os.path.abspath(tracking_dir)
    os.makedirs(abs_dir, exist_ok=True)
    uri = f"file:{abs_dir}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def start_run(run_name=None, tags=None):
    """Démarre un run MLflow (à utiliser comme context manager)."""
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params(params):
    """Logge un dict d'hyperparamètres."""
    mlflow.log_params(params)


def log_metrics(metrics, step=None):
    """Logge un dict de métriques scalaires."""
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)


def log_artifact(path, artifact_path=None):
    """Logge un fichier (plot, modèle, métadonnées) dans le run courant."""
    if os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)


def reward_curve(reward_array, out_path, title="Training reward", window=500):
    """Trace la courbe de reward lissée (moyenne glissante) et la sauvegarde.

    Renvoie le chemin du PNG produit, prêt à être loggé comme artefact.
    """
    rewards = np.asarray(reward_array, dtype=float)
    win = min(window, max(1, len(rewards)))
    smoothed = np.convolve(rewards, np.ones(win) / win, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smoothed, linewidth=0.9)
    ax.set_title(f"{title} (rolling {win})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
