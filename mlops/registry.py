"""Model registry & versioning — backend fichier local.

Le Model Registry natif de MLflow exige un backend base de données (sqlite/…),
incompatible avec notre store 100% fichier. On implémente donc un registry
fichier simple, inspectable et versionné, sous `models/<algorithme>/` :

    models/
      qlearning/
        registry.json          # index : versions + champion courant
        v1/
          policy.npy           # action greedy par état  (shape: [n_states])
          model.npy|model.pth  # modèle brut (Q-table / poids DQN)
          metadata.json        # métriques, params, run_id, format
        v2/ ...

Point clé : quel que soit l'algorithme, on extrait une **policy greedy**
(une action optimale par état discret). Le serving n'a alors besoin que de
`policy.npy` + numpy — ni torch, ni l'architecture du réseau.

Promotion : une version devient "champion" si elle passe les garde-fous
(steps/pénalités) ET améliore la métrique de promotion (mean_reward) par
rapport au champion courant. Voir `promotion` dans config.yaml.
"""

import json
import os
import time

import numpy as np

from mlops.config import promotion_rules

MODELS_DIR = os.environ.get("TAXI_MODELS_DIR", "models")


# --------------------------------------------------------------------------- #
# Extraction de la policy greedy (uniforme tous algorithmes)
# --------------------------------------------------------------------------- #
def extract_policy(agent):
    """Retourne (policy, raw_model, raw_format).

    policy : np.ndarray int [n_states] — action greedy par état.
    raw_model / raw_format : le modèle brut et son format de sérialisation,
    pour retraining/audit ('q_table' en .npy, 'dqn_state' en .pth).
    """
    if hasattr(agent, "q_table"):
        q = np.asarray(agent.q_table)
        return np.argmax(q, axis=1).astype(np.int64), q, "q_table"

    if hasattr(agent, "policy_net"):
        return _extract_dqn_policy(agent), agent, "dqn_state"

    raise TypeError(
        f"Agent {type(agent).__name__} sans q_table ni policy_net : "
        "extraction de policy impossible."
    )


def _extract_dqn_policy(agent):
    """Calcule l'action greedy du DQN pour chacun des n_states états."""
    import torch

    n_states = agent.state_size
    eye = torch.eye(n_states, device=agent.device)  # encodage one-hot de tous les états
    agent.policy_net.eval()
    with torch.no_grad():
        q_values = agent.policy_net(eye)
    return q_values.argmax(dim=1).cpu().numpy().astype(np.int64)


def _env_dims(agent):
    """(n_states, n_actions) déduits de l'environnement de l'agent."""
    return int(agent.env.observation_space.n), int(agent.env.action_space.n)


# --------------------------------------------------------------------------- #
# Index du registry (par algorithme)
# --------------------------------------------------------------------------- #
def _algo_dir(algorithm, models_dir=MODELS_DIR):
    return os.path.join(models_dir, algorithm)


def _index_path(algorithm, models_dir=MODELS_DIR):
    return os.path.join(_algo_dir(algorithm, models_dir), "registry.json")


def _load_index(algorithm, models_dir=MODELS_DIR):
    path = _index_path(algorithm, models_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"algorithm": algorithm, "champion": None, "versions": []}


def _save_index(algorithm, index, models_dir=MODELS_DIR):
    os.makedirs(_algo_dir(algorithm, models_dir), exist_ok=True)
    with open(_index_path(algorithm, models_dir), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# --------------------------------------------------------------------------- #
# Enregistrement & promotion
# --------------------------------------------------------------------------- #
def register_from_run(agent, algorithm, metrics, params, run_id, models_dir=MODELS_DIR):
    """Enregistre une nouvelle version du modèle et évalue sa promotion.

    Appelé depuis un run MLflow actif (train.py) : la version est aussi loggée
    comme artefact du run pour la traçabilité.

    Returns:
        dict {name, version, stage, path, promoted}.
    """
    index = _load_index(algorithm, models_dir)
    version = len(index["versions"]) + 1
    version_dir = os.path.join(_algo_dir(algorithm, models_dir), f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    policy, raw_model, raw_format = extract_policy(agent)
    n_states, n_actions = _env_dims(agent)

    # policy.npy : action greedy par état (utilisée par le serving).
    np.save(os.path.join(version_dir, "policy.npy"), policy)

    # Modèle brut (pour retraining / audit).
    if raw_format == "q_table":
        np.save(os.path.join(version_dir, "model.npy"), raw_model)
        model_file = "model.npy"
    else:  # dqn_state
        raw_model.save(os.path.join(version_dir, "model.pth"))
        model_file = "model.pth"

    metadata = {
        "algorithm": algorithm,
        "version": version,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "format": raw_format,
        "model_file": model_file,
        "n_states": n_states,
        "n_actions": n_actions,
        "metrics": metrics,
        "params": params,
    }
    with open(os.path.join(version_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Décision de promotion.
    promoted = _evaluate_promotion(index, metadata)
    stage = "Production" if promoted else "None"
    entry = {"version": version, "metrics": metrics, "stage": stage,
             "run_id": run_id, "path": version_dir}
    index["versions"].append(entry)
    if promoted:
        # L'ancien champion repasse en "None".
        for v in index["versions"]:
            if v["version"] != version:
                v["stage"] = "None"
        index["champion"] = version
    _save_index(algorithm, index, models_dir)

    # Traçabilité MLflow : logge la version comme artefact du run actif.
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_artifact(os.path.join(version_dir, "metadata.json"),
                                artifact_path=f"model/v{version}")
            mlflow.log_artifact(os.path.join(version_dir, "policy.npy"),
                                artifact_path=f"model/v{version}")
            mlflow.set_tag("registered_version", version)
            mlflow.set_tag("promoted", str(promoted))
    except Exception:
        pass  # le tracking ne doit jamais faire échouer un enregistrement

    return {"name": algorithm, "version": version, "stage": stage,
            "path": version_dir, "promoted": promoted}


def _evaluate_promotion(index, metadata):
    """True si la version candidate doit devenir champion."""
    rules = promotion_rules()
    metrics = metadata["metrics"]

    # Garde-fous absolus.
    if metrics.get("mean_steps", float("inf")) > rules.get("max_steps", float("inf")):
        return False
    if metrics.get("mean_penalties", float("inf")) > rules.get("max_penalties", float("inf")):
        return False

    metric_key = rules.get("metric", "mean_reward")
    candidate = metrics.get(metric_key, float("-inf"))

    champion_version = index.get("champion")
    if champion_version is None:
        return True  # premier modèle valide -> champion d'office

    champion = next((v for v in index["versions"]
                     if v["version"] == champion_version), None)
    if champion is None:
        return True
    return candidate > champion["metrics"].get(metric_key, float("-inf"))


# --------------------------------------------------------------------------- #
# Lecture (serving / inspection)
# --------------------------------------------------------------------------- #
def get_champion(algorithm, models_dir=MODELS_DIR):
    """Retourne (version_dir, metadata) du champion, ou (None, None)."""
    index = _load_index(algorithm, models_dir)
    champ = index.get("champion")
    if champ is None:
        return None, None
    version_dir = os.path.join(_algo_dir(algorithm, models_dir), f"v{champ}")
    with open(os.path.join(version_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return version_dir, json.load(f)


def get_latest(algorithm, models_dir=MODELS_DIR):
    """Retourne (version_dir, metadata) de la dernière version enregistrée.

    Sert au mode "test" de la console : permet d'inspecter un algorithme qui
    n'a pas (encore) de champion validé. Retourne (None, None) si aucune version.
    """
    index = _load_index(algorithm, models_dir)
    if not index["versions"]:
        return None, None
    latest = max(index["versions"], key=lambda v: v["version"])
    version_dir = os.path.join(_algo_dir(algorithm, models_dir), f"v{latest['version']}")
    meta_path = os.path.join(version_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        return version_dir, json.load(f)


def load_policy(version_dir):
    """Charge la policy greedy (np.ndarray) d'une version donnée."""
    return np.load(os.path.join(version_dir, "policy.npy"))


def list_versions(algorithm, models_dir=MODELS_DIR):
    """Liste les versions enregistrées (pour inspection / CLI)."""
    return _load_index(algorithm, models_dir)


def best_algorithm(models_dir=MODELS_DIR):
    """Parmi tous les algorithmes, retourne celui dont le champion est le
    meilleur selon la métrique de promotion. (None, None) si aucun.
    """
    rules = promotion_rules()
    metric_key = rules.get("metric", "mean_reward")
    best_algo, best_meta, best_score = None, None, float("-inf")

    if not os.path.isdir(models_dir):
        return None, None
    for algorithm in sorted(os.listdir(models_dir)):
        version_dir, meta = get_champion(algorithm, models_dir)
        if meta is None:
            continue
        score = meta["metrics"].get(metric_key, float("-inf"))
        if score > best_score:
            best_algo, best_meta, best_score = algorithm, meta, score
    return best_algo, best_meta
