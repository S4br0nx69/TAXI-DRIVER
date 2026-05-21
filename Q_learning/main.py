import q_learning as Taxi
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')

def cvar_metrics(reward_array):
    """Calcul du CVaR (Conditional Value at Risk) sur les rewards d'entraînement."""
    reward_array = np.array(reward_array)
    confidence_level = 0.95
    percentile_index = int((1 - confidence_level) * len(reward_array))
    sorted_rewards = np.sort(reward_array)
    percentile_value = sorted_rewards[percentile_index]
    cvar = sorted_rewards[:percentile_index + 1].mean()
    print(f"CVaR à {confidence_level} de niveau de confiance : {cvar}")


start = time.time()
taxi = Taxi.Taxi('rgb_array')
reward_array = taxi.train(train_episodes=10000, training_graph=True)
train_time = round(time.time() - start, 2)

taxi.test(test_episodes=25,
          timestamp=0.1,
          fast_testing=True,
          final_frame_pause=1)

print(f"Train execution time: {train_time}s\n")
cvar_metrics(reward_array)