"""Entrypoint d'entraînement unifié et instrumenté MLflow.

Entraîne n'importe lequel des agents du projet à partir de la configuration
centralisée, logge hyperparamètres / métriques / artefacts dans MLflow
(backend fichier local), et retourne un résumé du run.

Usage :
    python -m mlops.train qlearning
    python -m mlops.train dqn --episodes 2000 --seed 7
    python -m mlops.train sarsa --register        # promotion via le registry

Les agents existants ne sont pas modifiés : ils sont importés dynamiquement
depuis leur sous-dossier via le registre ALGORITHMS.
"""

import argparse
import importlib
import os
import sys
import tempfile
import time

import numpy as np

# Permet `python -m mlops.train` comme `python mlops/train.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlops import ALGORITHMS
from mlops.config import resolve
from mlops.seeding import set_global_seeds
from mlops import tracking

# Attributs d'hyperparamètres susceptibles d'exister sur un agent.
HYPERPARAM_KEYS = (
    "alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
    "lr", "batch_size",
)


def load_agent(algorithm, render_mode="ansi", env_version="v4"):
    """Importe dynamiquement la classe d'agent et l'instancie."""
    if algorithm not in ALGORITHMS:
        raise KeyError(f"Algorithme inconnu : {algorithm}. Choix : {sorted(ALGORITHMS)}")
    subdir, module_name, class_name = ALGORITHMS[algorithm]

    module_dir = os.path.join(_ROOT, subdir)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module = importlib.import_module(module_name)
    agent_cls = getattr(module, class_name)
    return agent_cls(render_mode, env_version=env_version)


def apply_hyperparams(agent, cfg):
    """Pousse les hyperparamètres de la config sur l'agent (si l'attribut existe)."""
    applied = {}
    for key in HYPERPARAM_KEYS:
        if key in cfg and hasattr(agent, key):
            setattr(agent, key, cfg[key])
            applied[key] = cfg[key]
    return applied


def cvar(reward_array, confidence=0.95):
    """Conditional Value at Risk : moyenne du pire (1-confidence) des rewards."""
    rewards = np.sort(np.asarray(reward_array, dtype=float))
    idx = int((1 - confidence) * len(rewards))
    return float(rewards[: idx + 1].mean())


def run(algorithm, overrides=None, tracking_dir="mlruns", register=False):
    """Exécute un entraînement complet instrumenté et retourne le résumé.

    Returns:
        dict : params effectifs, métriques, run_id MLflow et (si register)
        infos de version du modèle dans le registry.
    """
    cfg = resolve(algorithm, overrides=overrides)
    agent = load_agent(algorithm, env_version=cfg["env_version"])

    applied = apply_hyperparams(agent, cfg)
    seed = set_global_seeds(cfg["seed"], agent.env)

    params = {
        "algorithm": algorithm,
        "seed": seed,
        "env_version": cfg["env_version"],
        "train_episodes": cfg["train_episodes"],
        "test_episodes": cfg["test_episodes"],
        **applied,
    }

    tracking.configure(tracking_dir)
    with tracking.start_run(run_name=algorithm, tags={"algorithm": algorithm}) as active:
        run_id = active.info.run_id
        tracking.log_params(params)

        start = time.time()
        reward_array = agent.train(
            train_episodes=cfg["train_episodes"], training_graph=False
        )
        train_time = round(time.time() - start, 2)

        avg_steps, avg_penalties, avg_reward = agent.test(
            test_episodes=cfg["test_episodes"], fast_testing=True
        )

        metrics = {
            "mean_reward": avg_reward,
            "mean_steps": avg_steps,
            "mean_penalties": avg_penalties,
            "train_mean_reward": float(np.mean(reward_array)),
            "train_cvar_95": cvar(reward_array),
            "final_epsilon": float(getattr(agent, "epsilon", 0.0)),
            "train_time_s": train_time,
        }
        tracking.log_metrics(metrics)

        # Artefact : courbe de convergence.
        with tempfile.TemporaryDirectory() as tmp:
            curve = tracking.reward_curve(
                reward_array,
                os.path.join(tmp, f"{algorithm}_reward.png"),
                title=f"{algorithm} — training reward",
            )
            tracking.log_artifact(curve, artifact_path="plots")

        result = {
            "algorithm": algorithm,
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
        }

        # Phase 3 : enregistrement / promotion dans le registry.
        if register:
            from mlops import registry

            version_info = registry.register_from_run(
                agent=agent,
                algorithm=algorithm,
                metrics=metrics,
                params=params,
                run_id=run_id,
            )
            result["registry"] = version_info

    _print_summary(result)
    return result


def _print_summary(result):
    """Affiche un récapitulatif lisible en fin de run."""
    m = result["metrics"]
    print("\n" + "=" * 56)
    print(f"  Run MLflow : {result['run_id']}  ({result['algorithm']})")
    print(f"  mean_reward={m['mean_reward']:.2f}  "
          f"mean_steps={m['mean_steps']:.2f}  "
          f"penalties={m['mean_penalties']:.2f}")
    print(f"  train_cvar_95={m['train_cvar_95']:.2f}  "
          f"time={m['train_time_s']}s")
    if "registry" in result:
        reg = result["registry"]
        print(f"  registry : {reg['name']} v{reg['version']} "
              f"(stage={reg['stage']})")
    print("=" * 56)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entraînement instrumenté MLflow — Taxi Driver")
    p.add_argument("algorithm", choices=sorted(ALGORITHMS),
                   help="Algorithme à entraîner")
    p.add_argument("--episodes", type=int, default=None,
                   help="Nombre d'épisodes d'entraînement (override config)")
    p.add_argument("--test-episodes", type=int, default=None,
                   help="Nombre d'épisodes de test (override config)")
    p.add_argument("--seed", type=int, default=None, help="Graine (override config)")
    p.add_argument("--tracking-dir", default="mlruns",
                   help="Répertoire du store MLflow (défaut: mlruns)")
    p.add_argument("--register", action="store_true",
                   help="Enregistre/promeut le modèle dans le registry")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    overrides = {
        "train_episodes": args.episodes,
        "test_episodes": args.test_episodes,
        "seed": args.seed,
    }
    run(args.algorithm, overrides=overrides,
        tracking_dir=args.tracking_dir, register=args.register)


if __name__ == "__main__":
    main()
