# Makefile — orchestration MLOps Taxi Driver
# Tout passe par Docker : aucune dépendance Python à installer sur l'hôte.

ALGO    ?= qlearning      # algo ciblé par `make train`
EPISODES ?= 10000         # nb d'épisodes (override: make train EPISODES=5000)

.PHONY: help build train pipeline serve mlflow-ui test clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Build les images Docker (entraînement + serving)
	docker compose build train serve

train:  ## Entraîne + enregistre un algo (ALGO=sarsa EPISODES=8000)
	docker compose run --rm train $(ALGO) --episodes $(EPISODES) --register

pipeline:  ## Entraîne+enregistre qlearning+sarsa+montecarlo et désigne le champion
	docker compose run --rm --entrypoint python3 train -m mlops.pipeline --episodes $(EPISODES)

serve:  ## Lance l'API d'inférence (http://localhost:8000)
	docker compose up serve

mlflow-ui:  ## Lance l'UI MLflow (http://localhost:5000)
	docker compose up mlflow-ui

test:  ## Exécute la suite de tests (pytest) dans l'image de dev
	docker build -f Dockerfile.dev -t taxi-dev .
	docker run --rm -v $(PWD):/app -w /app taxi-dev python3 -m pytest tests/ -q

clean:  ## Supprime les artefacts de runs (mlruns/, models/)
	rm -rf mlruns models
