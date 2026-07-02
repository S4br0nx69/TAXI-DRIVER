"""Grid search pour optimiser les hyperparamètres du Q-Learning."""
import sys
sys.path.insert(0, 'Q_learning')  # import direct du module sans __init__.py
import q_learning as Taxi
import numpy as np
import argparse
from itertools import product


def grid_search(repeats=1):
    alphas = [0.05, 0.1, 0.2, 0.3]      # taux d'apprentissage à tester
    gammas = [0.6, 0.8, 0.9, 0.99]      # facteur de discount à tester
    decays = [0.999, 0.9995, 0.9999]    # vitesse de décroissance d'epsilon à tester

    results = []
    total = len(alphas) * len(gammas) * len(decays)
    print(f"=== Grid Search Q-Learning ({total} combinaisons, {repeats} run(s) chacune) ===\n")

    for i, (alpha, gamma, decay) in enumerate(product(alphas, gammas, decays)):  # produit cartésien = 48 combinaisons
        steps_runs, penalties_runs, reward_runs, completion_runs = [], [], [], []

        for r in range(repeats):  # plusieurs seeds pour moyenner le bruit d'une combinaison
            taxi = Taxi.Taxi('ansi')
            taxi.alpha = alpha
            taxi.gamma = gamma
            taxi.epsilon_decay = decay

            taxi.train(train_episodes=10000, training_graph=False, seed=r)
            test_result = taxi.test(test_episodes=50, fast_testing=True, seed=r)  # dict: steps/penalties/reward/completion_rate

            steps_runs.append(test_result['steps'])
            penalties_runs.append(test_result['penalties'])
            reward_runs.append(test_result['reward'])
            completion_runs.append(test_result['completion_rate'])

        steps = float(np.mean(steps_runs))
        penalties = float(np.mean(penalties_runs))
        reward = float(np.mean(reward_runs))
        completion_rate = float(np.mean(completion_runs))

        results.append({
            'alpha': alpha, 'gamma': gamma, 'decay': decay,
            'steps': steps, 'reward': reward, 'penalties': penalties,
            'completion_rate': completion_rate,
        })
        print(f"[{i+1}/{total}] α={alpha}, γ={gamma}, decay={decay} "
              f"→ steps={steps:.1f}, reward={reward:.2f}, completion={completion_rate:.1f}%")

    # Tri lexicographique : complétion d'abord, reward en 2e, steps (mini) en départage
    best = max(
        results,
        key=lambda x: (x['completion_rate'], x['reward'], -x['steps'])
    )
    print(f"\n{'='*60}")
    print(f"MEILLEUR : α={best['alpha']}, γ={best['gamma']}, decay={best['decay']}")
    print(f"           completion={best['completion_rate']:.1f}%, "
          f"reward={best['reward']:.2f}, steps={best['steps']:.1f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search Q-Learning')
    parser.add_argument('--repeats', type=int, default=1,  # 1 = rapide/bruité, >1 = plus fiable
                         help="Nombre de runs (seeds différents) par combinaison, "
                              "pour lisser le bruit d'une seule exécution.")
    args = parser.parse_args()
    grid_search(repeats=args.repeats)
