# Déploiement Azure — Taxi Driver

Déploie deux applications sur **Azure Container Apps** :

| App | Rôle | Port |
|-----|------|------|
| `taxi-serve` | API d'inférence FastAPI + interface web de benchmark | 8000 |
| `taxi-mlflow` | UI MLflow (snapshot des runs `mlruns/`) | 5000 |

Les images sont buildées **dans le cloud** (`az acr build`) : aucun démon Docker
local requis. Les modèles (`models/`) et le snapshot `mlruns/` sont **embarqués**
dans les images (le `.dockerignore` du dépôt les exclut du build local, donc
`deploy.sh` passe par un contexte de staging `azure/.build/`).

## Prérequis

- Azure CLI connecté : `az login`
- Un abonnement Azure actif (les crédits étudiants suffisent — Container Apps a
  une offre gratuite mensuelle et scale-to-zero).

## Déployer

```bash
./azure/deploy.sh
```

Le script est idempotent sur l'infra (groupe, registre, environnement) et
affiche à la fin les URLs publiques HTTPS des deux apps.

### Paramétrer (optionnel)

```bash
LOCATION=westeurope \
RESOURCE_GROUP=taxi-rg \
SERVE_ALGO=sarsa \
./azure/deploy.sh
```

| Variable | Défaut | Rôle |
|----------|--------|------|
| `LOCATION` | `francecentral` | région Azure |
| `RESOURCE_GROUP` | `taxi-driver-rg` | groupe de ressources |
| `ACR_NAME` | `taxidriveracr<random>` | nom du registre (globalement unique) |
| `SERVE_ALGO` | `auto` | modèle servi par défaut (`auto`/`sarsa`/`qlearning`/`dqn`) |

## Vérifier

```bash
curl https://<serve-url>/health
curl -X POST https://<serve-url>/predict \
     -H 'Content-Type: application/json' -d '{"state": 328}'
```

## Mettre à jour après un réentraînement

Les modèles/runs étant figés dans l'image, un nouveau champion nécessite un
rebuild + redeploy :

```bash
./azure/deploy.sh          # rebuild les images et met à jour les révisions
```

## Nettoyer (éviter les coûts)

```bash
az group delete -n taxi-driver-rg --yes --no-wait
```

## Limites connues

- **MLflow UI = snapshot en lecture.** Les runs affichés sont ceux présents au
  build. Les liens d'artefacts peuvent pointer vers des chemins absolus
  d'origine ; params/métriques restent corrects. Pour du live, monter un
  **Azure File Share** sur `/app/mlruns` (évolution possible).
- **Pas d'entraînement dans le cloud** ici (choix : périmètre « site + MLflow »).
  L'entraînement reste local via `make train` / `make pipeline`.
