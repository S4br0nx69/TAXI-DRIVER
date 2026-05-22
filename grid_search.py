"""Grid search pour optimiser les hyperparamètres du Q-Learning."""
import sys
sys.path.insert(0, 'Q_learning')
import q_learning as Taxi
import numpy as np
import time
from itertools import product


def grid_search():
    alphas = [0.05, 0.1, 0.2, 0.3]
    gammas = [0.6, 0.8, 0.9, 0.99]
    decays = [0.999, 0.9995, 0.9999]

    results = []
    total = len(alphas) * len(gammas) * len(decays)
    print(f"=== Grid Search Q-Learning ({total} combinaisons) ===\n")

    for i, (alpha, gamma, decay) in enumerate(product(alphas, gammas, decays)):
        taxi = Taxi.Taxi('ansi')
        taxi.alpha = alpha
        taxi.gamma = gamma
        taxi.epsilon_decay = decay

        taxi.train(train_episodes=10000, training_graph=False)
        steps, penalties, reward = taxi.test(test_episodes=50, fast_testing=True)

        results.append({
            'alpha': alpha, 'gamma': gamma, 'decay': decay,
            'steps': steps, 'reward': reward, 'penalties': penalties
        })
        print(f"[{i+1}/{total}] α={alpha}, γ={gamma}, decay={decay} "
              f"→ steps={steps:.1f}, reward={reward:.2f}")

    # Meilleur résultat
    best = min(results, key=lambda x: x['steps'])
    print(f"\n{'='*60}")
    print(f"MEILLEUR : α={best['alpha']}, γ={best['gamma']}, decay={best['decay']}")
    print(f"           steps={best['steps']:.1f}, reward={best['reward']:.2f}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    grid_search()
