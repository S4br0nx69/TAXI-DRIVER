"""API d'inférence FastAPI — sert la policy greedy du modèle champion.

100% local : aucune dépendance réseau sortante, aucun appel à un service tiers.
Le serving ne charge que `policy.npy` (action greedy par état) + numpy : ni
torch ni l'architecture du réseau ne sont nécessaires, quel que soit l'algo.

Sélection du modèle servi (au démarrage) :
- `TAXI_SERVE_ALGORITHM=sarsa` : sert le champion d'un algo précis.
- non défini ou `auto` : sert le meilleur champion tous algos confondus.

Lancement :
    uvicorn mlops.serve.app:app --host 0.0.0.0 --port 8000

Endpoints :
    GET  /health        état du service + modèle chargé
    GET  /model/info    métadonnées du modèle servi
    POST /predict       {"state": int} -> action greedy
    POST /reload        recharge le champion courant (après retraining)
"""

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops import ACTION_NAMES
from mlops import registry


@asynccontextmanager
async def lifespan(_app):
    """Charge le champion au démarrage du service."""
    _load_champion()
    yield


app = FastAPI(
    title="Taxi Driver — Inference API",
    description="Sert la policy greedy du modèle RL champion (Taxi-v3/v4).",
    version="1.0.0",
    lifespan=lifespan,
)

# État du modèle chargé (rempli au démarrage et par /reload).
_STATE = {
    "algorithm": None,
    "policy": None,
    "metadata": None,
    "n_states": 0,
}


class PredictRequest(BaseModel):
    state: int = Field(..., ge=0, description="État discret Taxi (0..n_states-1)")


class PredictResponse(BaseModel):
    state: int
    action: int
    action_name: str
    algorithm: str
    model_version: int


def _load_champion():
    """(Re)charge le champion en mémoire selon TAXI_SERVE_ALGORITHM."""
    target = os.environ.get("TAXI_SERVE_ALGORITHM", "auto").strip().lower()

    if target in ("", "auto"):
        algorithm, meta = registry.best_algorithm()
        if algorithm is None:
            _reset_state()
            return False
        version_dir, _ = registry.get_champion(algorithm)
    else:
        algorithm = target
        version_dir, meta = registry.get_champion(algorithm)
        if meta is None:
            _reset_state()
            return False

    _STATE["algorithm"] = algorithm
    _STATE["metadata"] = meta
    _STATE["policy"] = registry.load_policy(version_dir)
    _STATE["n_states"] = int(meta["n_states"])
    return True


def _reset_state():
    _STATE.update({"algorithm": None, "policy": None, "metadata": None, "n_states": 0})


@app.get("/health")
def health():
    """Liveness/readiness : le service répond, modèle chargé ou non."""
    loaded = _STATE["policy"] is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "algorithm": _STATE["algorithm"],
    }


@app.get("/model/info")
def model_info():
    """Métadonnées du modèle servi (version, métriques, params, run MLflow)."""
    if _STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle champion disponible.")
    meta = _STATE["metadata"]
    return {
        "algorithm": meta["algorithm"],
        "version": meta["version"],
        "run_id": meta["run_id"],
        "created_at": meta["created_at"],
        "n_states": meta["n_states"],
        "n_actions": meta["n_actions"],
        "metrics": meta["metrics"],
        "params": meta["params"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Retourne l'action greedy recommandée pour un état donné."""
    if _STATE["policy"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle champion disponible.")
    if req.state >= _STATE["n_states"]:
        raise HTTPException(
            status_code=422,
            detail=f"state hors bornes : 0..{_STATE['n_states'] - 1}",
        )

    action = int(_STATE["policy"][req.state])
    action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
    return PredictResponse(
        state=req.state,
        action=action,
        action_name=action_name,
        algorithm=_STATE["algorithm"],
        model_version=_STATE["metadata"]["version"],
    )


@app.post("/reload")
def reload_model():
    """Recharge le champion courant (hot reload après un retraining)."""
    ok = _load_champion()
    if not ok:
        raise HTTPException(status_code=503, detail="Aucun modèle champion à charger.")
    return {
        "status": "reloaded",
        "algorithm": _STATE["algorithm"],
        "version": _STATE["metadata"]["version"],
    }
