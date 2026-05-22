"""Tests unitaires pour les agents RL."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_bruteforce():
    """Le brute-force retourne des résultats cohérents."""
    from bruteforce import run_bruteforce
    results = run_bruteforce(test_episodes=10)
    assert results['mean_steps'] > 0
    assert results['mean_reward'] < 0
    assert results['mean_penalties'] > 0
    assert results['success_rate'] == 0.0


def test_qlearning_train():
    """Le Q-Learning s'entraîne et retourne un array de rewards."""
    sys.path.insert(0, 'Q_learning')
    import q_learning as Taxi
    taxi = Taxi.Taxi('ansi')
    rewards = taxi.train(train_episodes=100, training_graph=False)
    assert isinstance(rewards, np.ndarray)
    assert len(rewards) == 100


def test_qlearning_convergence():
    """Le Q-Learning converge après entraînement."""
    sys.path.insert(0, 'Q_learning')
    import q_learning as Taxi
    taxi = Taxi.Taxi('ansi')
    taxi.train(train_episodes=10000, training_graph=False)
    steps, penalties, reward = taxi.test(test_episodes=20, fast_testing=True)
    assert steps < 100, f"Steps trop élevés: {steps}"
    assert penalties == 0, f"Pénalités détectées: {penalties}"


def test_sarsa_train():
    """SARSA s'entraîne et retourne un array de rewards."""
    sys.path.insert(0, 'Sarsa')
    import sarsa as Agent
    agent = Agent.Sarsa('ansi')
    rewards = agent.train(train_episodes=100, training_graph=False)
    assert isinstance(rewards, np.ndarray)
    assert len(rewards) == 100


def test_montecarlo_train():
    """Monte Carlo s'entraîne et retourne un array de rewards."""
    sys.path.insert(0, 'MonteCarlo')
    import monte_carlo as Agent
    agent = Agent.MonteCarlo('ansi')
    rewards = agent.train(train_episodes=100, training_graph=False)
    assert isinstance(rewards, np.ndarray)
    assert len(rewards) == 100


def test_utils_safe_input_int(monkeypatch):
    """safe_input_int gère les entrées invalides."""
    from utils import safe_input_int
    monkeypatch.setattr('builtins.input', lambda _: 'abc')
    assert safe_input_int("test: ", 42, 1, 100) == 42

    monkeypatch.setattr('builtins.input', lambda _: '999999')
    assert safe_input_int("test: ", 42, 1, 100) == 42

    monkeypatch.setattr('builtins.input', lambda _: '50')
    assert safe_input_int("test: ", 42, 1, 100) == 50


def test_utils_safe_input_float(monkeypatch):
    """safe_input_float gère les entrées invalides."""
    from utils import safe_input_float
    monkeypatch.setattr('builtins.input', lambda _: 'abc')
    assert safe_input_float("test: ", 0.5, 0.0, 1.0) == 0.5

    monkeypatch.setattr('builtins.input', lambda _: '5.0')
    assert safe_input_float("test: ", 0.5, 0.0, 1.0) == 0.5

    monkeypatch.setattr('builtins.input', lambda _: '0.3')
    assert safe_input_float("test: ", 0.5, 0.0, 1.0) == 0.3
