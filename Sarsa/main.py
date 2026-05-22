import sarsa as Agent
import numpy as np
import time


def cvar_metrics(reward_array):
    """Calcul du CVaR (Conditional Value at Risk) sur les rewards d'entraînement."""
    reward_array = np.array(reward_array)
    confidence_level = 0.95
    percentile_index = int((1 - confidence_level) * len(reward_array))
    sorted_rewards = np.sort(reward_array)
    cvar = sorted_rewards[:percentile_index + 1].mean()
    print(f"CVaR à {confidence_level} de niveau de confiance : {cvar}")


def user_mode():
    """Mode utilisateur : réglage manuel des hyperparamètres."""
    print("\n=== MODE UTILISATEUR ===\n")

    train_episodes = int(input("Nombre d'épisodes d'entraînement [25000] : ") or 25000)
    test_episodes = int(input("Nombre d'épisodes de test [25] : ") or 25)
    alpha = float(input("Learning rate α [0.2] : ") or 0.2)
    gamma = float(input("Discount factor γ [0.9] : ") or 0.9)
    epsilon = float(input("Epsilon initial [1.0] : ") or 1.0)
    epsilon_min = float(input("Epsilon minimum [0.01] : ") or 0.01)
    epsilon_decay = float(input("Epsilon decay [0.999] : ") or 0.999)
    display = input("Afficher les épisodes de test ? (o/n) [o] : ").strip().lower() or "o"

    agent = Agent.Sarsa('rgb_array')
    agent.alpha = alpha
    agent.gamma = gamma
    agent.epsilon = epsilon
    agent.epsilon_min = epsilon_min
    agent.epsilon_decay = epsilon_decay

    start = time.time()
    reward_array = agent.train(train_episodes=train_episodes, training_graph=True)
    train_time = round(time.time() - start, 2)

    fast = display != "o"
    agent.test(test_episodes=test_episodes,
               timestamp=0.3,
               fast_testing=fast,
               final_frame_pause=1)

    print(f"\nTrain execution time: {train_time}s")
    cvar_metrics(reward_array)


def time_limited_mode():
    """Mode time-limited : paramètres pré-optimisés."""
    print("\n=== MODE TIME-LIMITED ===\n")

    train_episodes = int(input("Nombre d'épisodes d'entraînement [10000] : ") or 10000)
    test_episodes = int(input("Nombre d'épisodes de test [25] : ") or 25)
    display = input("Afficher les épisodes de test ? (o/n) [o] : ").strip().lower() or "o"

    agent = Agent.Sarsa('rgb_array')
    agent.alpha = 0.2
    agent.gamma = 0.9
    agent.epsilon = 1.0
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.999

    print(f"\nParamètres optimisés : α={agent.alpha}, γ={agent.gamma}, "
          f"ε={agent.epsilon}→{agent.epsilon_min}, decay={agent.epsilon_decay}\n")

    start = time.time()
    reward_array = agent.train(train_episodes=train_episodes, training_graph=True)
    train_time = round(time.time() - start, 2)

    fast = display != "o"
    agent.test(test_episodes=test_episodes,
               timestamp=0.3,
               fast_testing=fast,
               final_frame_pause=1)

    print(f"\nTrain execution time: {train_time}s")
    cvar_metrics(reward_array)


if __name__ == "__main__":
    print("╔══════════════════════════════════╗")
    print("║      TAXI DRIVER — SARSA         ║")
    print("╚══════════════════════════════════╝")
    print("\n1. Mode utilisateur (réglage des hyperparamètres)")
    print("2. Mode time-limited (paramètres optimisés)\n")

    choice = input("Choix [1/2] : ").strip() or "2"

    if choice == "1":
        user_mode()
    else:
        time_limited_mode()