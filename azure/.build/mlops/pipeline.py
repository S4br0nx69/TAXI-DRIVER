"""Pipeline d'orchestration MLOps : train -> evaluate -> register -> select.

Enchaîne l'entraînement instrumenté de plusieurs algorithmes, enregistre
chacun dans le registry (avec promotion automatique), puis désigne le
meilleur modèle tous algorithmes confondus (le champion servi par l'API).

Usage :
    python -m mlops.pipeline                       # qlearning + sarsa
    python -m mlops.pipeline --algorithms sarsa dqn --episodes 5000
    python -m mlops.pipeline --all
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlops import ALGORITHMS
from mlops import registry
from mlops import train as train_mod

# Par défaut : les algos tabulaires rapides et fiables (DQN exclu car lent).
DEFAULT_ALGOS = ["qlearning", "sarsa", "montecarlo"]


def run_pipeline(algorithms, overrides=None, tracking_dir="mlruns"):
    """Entraîne+enregistre chaque algo, retourne le résumé + le champion global."""
    results = []
    for algo in algorithms:
        print(f"\n{'#' * 60}\n#  PIPELINE — entraînement : {algo}\n{'#' * 60}")
        res = train_mod.run(algo, overrides=overrides,
                            tracking_dir=tracking_dir, register=True)
        results.append(res)

    _print_leaderboard(results)

    best_algo, best_meta = registry.best_algorithm()
    if best_algo is not None:
        print(f"\n>>> Champion global : {best_algo} v{best_meta['version']} "
              f"(mean_reward={best_meta['metrics']['mean_reward']:.2f}, "
              f"mean_steps={best_meta['metrics']['mean_steps']:.2f})")
    else:
        print("\n>>> Aucun modèle n'a passé les garde-fous de promotion.")

    return {"results": results, "champion": best_algo, "champion_meta": best_meta}


def _print_leaderboard(results):
    """Classe les runs de ce pipeline par mean_reward décroissant."""
    print(f"\n{'=' * 60}\n  LEADERBOARD (ce pipeline)\n{'=' * 60}")
    ranked = sorted(results, key=lambda r: r["metrics"]["mean_reward"], reverse=True)
    print(f"  {'algo':<12} {'reward':>9} {'steps':>8} {'penal':>7}  {'promu':>6}")
    for r in ranked:
        m = r["metrics"]
        promoted = r.get("registry", {}).get("promoted", False)
        print(f"  {r['algorithm']:<12} {m['mean_reward']:>9.2f} "
              f"{m['mean_steps']:>8.2f} {m['mean_penalties']:>7.2f}  "
              f"{'oui' if promoted else 'non':>6}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pipeline MLOps — Taxi Driver")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--algorithms", nargs="+", choices=sorted(ALGORITHMS),
                   help=f"Algos à entraîner (défaut: {DEFAULT_ALGOS})")
    g.add_argument("--all", action="store_true",
                   help="Entraîne tous les algorithmes (DQN inclus, lent)")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override du nombre d'épisodes pour tous les algos")
    p.add_argument("--test-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tracking-dir", default="mlruns")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.all:
        algorithms = sorted(ALGORITHMS)
    elif args.algorithms:
        algorithms = args.algorithms
    else:
        algorithms = DEFAULT_ALGOS

    overrides = {
        "train_episodes": args.episodes,
        "test_episodes": args.test_episodes,
        "seed": args.seed,
    }
    run_pipeline(algorithms, overrides=overrides, tracking_dir=args.tracking_dir)


if __name__ == "__main__":
    main()
