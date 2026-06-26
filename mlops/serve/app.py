"""API d'inférence FastAPI — sert la policy greedy des modèles champions.

100% local : aucune dépendance réseau sortante, aucun appel à un service tiers.
Le serving ne charge que `policy.npy` (action greedy par état) + numpy : ni
torch ni l'architecture du réseau ne sont nécessaires, quel que soit l'algo.

Tous les champions disponibles (un par algorithme) sont chargés en mémoire au
démarrage : la console peut donc commuter d'un algorithme à l'autre à la volée.

Modèle servi par défaut (quand l'appelant ne précise pas d'algorithme) :
- `TAXI_SERVE_ALGORITHM=sarsa` : champion d'un algo précis.
- non défini ou `auto` : meilleur champion tous algos confondus.

Lancement :
    uvicorn mlops.serve.app:app --host 0.0.0.0 --port 8000

Endpoints :
    GET  /health        état du service + modèle par défaut
    GET  /algorithms    liste des champions disponibles (un par algo)
    GET  /model/info    métadonnées d'un modèle (?algorithm=, défaut = courant)
    POST /predict       {"state": int, "algorithm"?: str} -> action greedy
    POST /reload        recharge tous les champions (après retraining)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mlops import ACTION_NAMES, ALGORITHMS
from mlops import registry

# Mini interface web (statique, 100% local) servie à la racine.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(_app):
    """Charge tous les champions disponibles au démarrage du service."""
    _load_all()
    yield


app = FastAPI(
    title="Taxi Driver — Inference API",
    description="Sert la policy greedy des modèles RL champions (Taxi-v3/v4).",
    version="2.0.0",
    lifespan=lifespan,
)

# Champions chargés en mémoire : algorithme -> {policy, metadata, n_states}.
_MODELS = {}
# Algorithme servi par défaut (résolu depuis TAXI_SERVE_ALGORITHM au chargement).
_DEFAULT = None


class PredictRequest(BaseModel):
    state: int = Field(..., ge=0, description="État discret Taxi (0..n_states-1)")
    algorithm: Optional[str] = Field(
        None, description="Algorithme à utiliser ; défaut = modèle courant."
    )


class PredictResponse(BaseModel):
    state: int
    action: int
    action_name: str
    algorithm: str
    model_version: int


def _load_all():
    """(Re)charge en mémoire un modèle par algorithme connu.

    On sert le champion validé s'il existe ; sinon, la dernière version
    enregistrée (marquée `validated=False`) pour permettre de tester en console
    un algorithme qui n'a pas (encore) passé les garde-fous de promotion.
    """
    _MODELS.clear()
    for algorithm in ALGORITHMS:
        version_dir, meta = registry.get_champion(algorithm)
        validated = meta is not None
        if meta is None:
            version_dir, meta = registry.get_latest(algorithm)
        if meta is None:
            continue
        _MODELS[algorithm] = {
            "policy": registry.load_policy(version_dir),
            "metadata": meta,
            "n_states": int(meta["n_states"]),
            "validated": validated,
        }

    # Résolution du modèle par défaut (algo précis, ou 'auto' = meilleur champion).
    global _DEFAULT
    target = os.environ.get("TAXI_SERVE_ALGORITHM", "auto").strip().lower()
    if target in ("", "auto"):
        best, _ = registry.best_algorithm()
        _DEFAULT = best if best in _MODELS else (next(iter(_MODELS), None))
    else:
        _DEFAULT = target if target in _MODELS else None
    return bool(_MODELS)


def _resolve(algorithm):
    """Nom de l'algo effectif pour une requête (None si indisponible)."""
    if algorithm and algorithm.strip().lower() not in ("", "auto"):
        algorithm = algorithm.strip().lower()
        return algorithm if algorithm in _MODELS else None
    return _DEFAULT if _DEFAULT in _MODELS else None


def _info(model):
    """Vue publique des métadonnées d'un modèle chargé."""
    meta = model["metadata"]
    return {
        "algorithm": meta["algorithm"],
        "version": meta["version"],
        "validated": model["validated"],
        "run_id": meta["run_id"],
        "created_at": meta["created_at"],
        "n_states": meta["n_states"],
        "n_actions": meta["n_actions"],
        "metrics": meta["metrics"],
        "params": meta["params"],
    }


@app.get("/", include_in_schema=False)
def ui():
    """Sert la mini interface web (visualisation + prédictions)."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    """Liveness/readiness : le service répond, modèle par défaut chargé ou non."""
    return {
        "status": "ok",
        "model_loaded": bool(_MODELS),
        "algorithm": _DEFAULT,
        "available": sorted(_MODELS.keys()),
    }


@app.get("/algorithms")
def algorithms():
    """Liste les champions disponibles (un par algorithme) + le défaut."""
    best, _ = registry.best_algorithm()
    items = []
    for algo, m in _MODELS.items():
        meta = m["metadata"]
        items.append({
            "algorithm": algo,
            "version": meta["version"],
            "validated": m["validated"],
            "metrics": meta["metrics"],
            "n_actions": meta["n_actions"],
            "is_best": algo == best,
        })
    items.sort(key=lambda x: x["metrics"].get("mean_reward", float("-inf")), reverse=True)
    return {"algorithms": items, "default": _DEFAULT, "best": best}


@app.get("/model/info")
def model_info(algorithm: Optional[str] = None):
    """Métadonnées d'un modèle (?algorithm= ; défaut = modèle courant)."""
    algo = _resolve(algorithm)
    if algo is None:
        raise HTTPException(status_code=503, detail="Aucun modèle champion disponible.")
    return _info(_MODELS[algo])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Retourne l'action greedy recommandée pour un état donné."""
    algo = _resolve(req.algorithm)
    if algo is None:
        if req.algorithm:
            raise HTTPException(
                status_code=404,
                detail=f"Algorithme '{req.algorithm}' indisponible. "
                       f"Disponibles : {sorted(_MODELS.keys())}",
            )
        raise HTTPException(status_code=503, detail="Aucun modèle champion disponible.")

    model = _MODELS[algo]
    if req.state >= model["n_states"]:
        raise HTTPException(
            status_code=422,
            detail=f"state hors bornes : 0..{model['n_states'] - 1}",
        )

    action = int(model["policy"][req.state])
    action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
    return PredictResponse(
        state=req.state,
        action=action,
        action_name=action_name,
        algorithm=algo,
        model_version=model["metadata"]["version"],
    )


@app.post("/reload")
def reload_model():
    """Recharge tous les champions (hot reload après un retraining)."""
    ok = _load_all()
    if not ok:
        raise HTTPException(status_code=503, detail="Aucun modèle champion à charger.")
    return {
        "status": "reloaded",
        "default": _DEFAULT,
        "available": sorted(_MODELS.keys()),
    }
