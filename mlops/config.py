"""Chargement et résolution de la configuration des expériences.

Lit `mlops/config.yaml` et fusionne, pour un algorithme donné, les valeurs
`defaults` avec ses hyperparamètres spécifiques. Les surcharges CLI ont la
priorité la plus haute.
"""

import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=CONFIG_PATH):
    """Charge le YAML de configuration brut."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(algorithm, overrides=None, path=CONFIG_PATH):
    """Résout la configuration effective d'un algorithme.

    Précédence (faible -> forte) :
      defaults globaux  <  hyperparamètres de l'algo  <  overrides CLI.

    Args:
        algorithm: clé de l'algo (qlearning, sarsa, montecarlo, dqn).
        overrides: dict de surcharges (les valeurs None sont ignorées).

    Returns:
        dict plat des paramètres effectifs (inclut seed, env_version,
        train_episodes, test_episodes et les hyperparamètres de l'algo).
    """
    cfg = load_config(path)
    algos = cfg.get("algorithms", {})
    if algorithm not in algos:
        raise KeyError(
            f"Algorithme inconnu : '{algorithm}'. "
            f"Disponibles : {sorted(algos)}"
        )

    resolved = dict(cfg.get("defaults", {}))
    resolved.update(algos[algorithm] or {})

    if overrides:
        resolved.update({k: v for k, v in overrides.items() if v is not None})

    resolved["algorithm"] = algorithm
    return resolved


def promotion_rules(path=CONFIG_PATH):
    """Retourne les règles de promotion du registry (Phase 3)."""
    return load_config(path).get("promotion", {})
