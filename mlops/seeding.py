"""Reproductibilité : seeds globaux pour des runs déterministes.

Tous les algorithmes utilisent `random` (epsilon-greedy / action_space.sample),
NumPy, et — pour le DQN — PyTorch. La randomness de l'environnement Gymnasium
est portée par son propre générateur, semé via `env.reset(seed=...)`.

`set_global_seeds()` est non-invasif : il agit sur les modules globaux et sur
l'environnement d'un agent existant, sans modifier le code des agents.
"""

import os
import random

import numpy as np


def set_global_seeds(seed, env=None):
    """Fixe tous les générateurs pseudo-aléatoires impliqués dans un run.

    Args:
        seed: graine entière (>= 0).
        env: environnement Gymnasium optionnel à semer (action_space + RNG interne).

    Retourne le seed appliqué (pratique pour le logging MLflow).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch est optionnel (seul le DQN en dépend) -> import paresseux.
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Déterminisme cuDNN (no-op en CPU, utile si GPU présent).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    if env is not None:
        seed_env(env, seed)

    return seed


def seed_env(env, seed):
    """Sème un environnement Gymnasium de façon déterministe.

    `env.reset(seed=...)` initialise le générateur interne ; les resets
    suivants (sans seed) poursuivent alors une séquence reproductible.
    """
    try:
        env.action_space.seed(seed)
    except (AttributeError, TypeError):
        pass
    try:
        env.reset(seed=seed)
    except TypeError:
        # Très anciennes API Gym sans paramètre seed.
        env.reset()
