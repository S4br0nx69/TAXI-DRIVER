"""Tests unitaires de la couche MLOps (config, seeding, registry, serving)."""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_resolve_merge():
    """resolve() fusionne defaults + hyperparams de l'algo."""
    from mlops.config import resolve
    cfg = resolve("qlearning")
    assert cfg["algorithm"] == "qlearning"
    assert cfg["alpha"] == 0.1            # spécifique qlearning
    assert cfg["seed"] == 42              # hérité des defaults
    assert "train_episodes" in cfg


def test_config_override_precedence():
    """Les overrides CLI priment, les None sont ignorés."""
    from mlops.config import resolve
    cfg = resolve("sarsa", overrides={"train_episodes": 123, "seed": None})
    assert cfg["train_episodes"] == 123
    assert cfg["seed"] == 42              # None ignoré -> défaut conservé


def test_config_unknown_algorithm():
    from mlops.config import resolve
    try:
        resolve("inexistant")
        assert False, "doit lever KeyError"
    except KeyError:
        pass


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def test_seeding_determinism():
    """Deux runs avec le même seed produisent la même séquence."""
    from mlops.seeding import set_global_seeds
    set_global_seeds(123)
    a = [random.random(), float(np.random.rand())]
    set_global_seeds(123)
    b = [random.random(), float(np.random.rand())]
    assert a == b


# --------------------------------------------------------------------------- #
# Registry (promotion)
# --------------------------------------------------------------------------- #
class _FakeAgent:
    """Agent minimal exposant une q_table + un env factice."""
    def __init__(self, q_table, n_actions=6):
        self.q_table = q_table

        class _Space:
            def __init__(self, n):
                self.n = n

        class _Env:
            def __init__(self, ns, na):
                self.observation_space = _Space(ns)
                self.action_space = _Space(na)

        self.env = _Env(q_table.shape[0], n_actions)


def test_registry_promotion_logic(tmp_path):
    """Promotion : premier modèle valide promu, meilleur reward le remplace."""
    from mlops import registry
    models_dir = str(tmp_path / "models")

    q = np.random.rand(500, 6)
    agent = _FakeAgent(q)

    good = {"mean_reward": 5.0, "mean_steps": 14.0, "mean_penalties": 0.0}
    r1 = registry.register_from_run(agent, "sarsa", good, {"p": 1}, "run1",
                                    models_dir=models_dir)
    assert r1["promoted"] is True
    assert r1["stage"] == "Production"

    # Pire reward -> pas de promotion.
    worse = {"mean_reward": 1.0, "mean_steps": 14.0, "mean_penalties": 0.0}
    r2 = registry.register_from_run(agent, "sarsa", worse, {"p": 2}, "run2",
                                    models_dir=models_dir)
    assert r2["promoted"] is False

    # Meilleur reward -> promotion + remplacement.
    better = {"mean_reward": 8.0, "mean_steps": 13.0, "mean_penalties": 0.0}
    r3 = registry.register_from_run(agent, "sarsa", better, {"p": 3}, "run3",
                                    models_dir=models_dir)
    assert r3["promoted"] is True

    idx = registry.list_versions("sarsa", models_dir=models_dir)
    assert idx["champion"] == 3


def test_registry_guardrails_block_promotion(tmp_path):
    """Un modèle qui dépasse max_steps n'est jamais promu."""
    from mlops import registry
    models_dir = str(tmp_path / "models")

    agent = _FakeAgent(np.random.rand(500, 6))
    bad = {"mean_reward": 100.0, "mean_steps": 999.0, "mean_penalties": 0.0}
    r = registry.register_from_run(agent, "qlearning", bad, {}, "run1",
                                   models_dir=models_dir)
    assert r["promoted"] is False
    assert registry.list_versions("qlearning", models_dir=models_dir)["champion"] is None


def test_extract_policy_from_qtable():
    """La policy greedy est l'argmax de la Q-table par état."""
    from mlops.registry import extract_policy
    q = np.zeros((500, 6))
    q[0, 3] = 1.0  # action 3 optimale en état 0
    q[1, 5] = 1.0
    agent = _FakeAgent(q)
    policy, raw, fmt = extract_policy(agent)
    assert policy.shape == (500,)
    assert policy[0] == 3 and policy[1] == 5
    assert fmt == "q_table"


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
def test_serving_predict(tmp_path, monkeypatch):
    """L'API renvoie l'action greedy du champion et valide les bornes."""
    from fastapi.testclient import TestClient
    from mlops import registry

    models_dir = str(tmp_path / "models")
    q = np.zeros((500, 6))
    q[42, 2] = 1.0  # action 2 (East) optimale en état 42
    registry.register_from_run(_FakeAgent(q), "sarsa",
                               {"mean_reward": 5.0, "mean_steps": 14.0,
                                "mean_penalties": 0.0},
                               {}, "run1", models_dir=models_dir)

    monkeypatch.setenv("TAXI_MODELS_DIR", models_dir)
    monkeypatch.setenv("TAXI_SERVE_ALGORITHM", "sarsa")
    # Recharger le module registry pour qu'il relise TAXI_MODELS_DIR.
    import importlib
    importlib.reload(registry)
    from mlops.serve import app as serve_app
    importlib.reload(serve_app)

    with TestClient(serve_app.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        pred = client.post("/predict", json={"state": 42}).json()
        assert pred["action"] == 2
        assert pred["action_name"] == "East"
        assert client.post("/predict", json={"state": 9999}).status_code == 422
