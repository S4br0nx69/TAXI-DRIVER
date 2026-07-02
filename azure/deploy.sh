#!/usr/bin/env bash
# =============================================================================
# Déploiement Taxi Driver sur Azure Container Apps.
#   - serve      : API d'inférence FastAPI + interface web  (port 8000)
#   - mlflow-ui  : UI MLflow des runs d'expériences         (port 5000)
#
# Les images sont buildées EN LOCAL avec Docker puis poussées vers l'ACR
# (les ACR Tasks/build cloud sont bloqués sur les abonnements gratuits). Les
# modèles et le snapshot mlruns/ sont embarqués dans les images (contexte de
# staging dédié pour contourner le .dockerignore du dépôt).
#
# Usage :  ./azure/deploy.sh
# Nettoyage : az group delete -n $RESOURCE_GROUP --yes --no-wait
# =============================================================================
set -euo pipefail

# --- Paramètres (surchargables par variables d'environnement) ----------------
LOCATION="${LOCATION:-francecentral}"
RESOURCE_GROUP="${RESOURCE_GROUP:-taxi-driver-rg}"
ACR_NAME="${ACR_NAME:-taxidriveracr$RANDOM}"   # doit être globalement unique
ENV_NAME="${ENV_NAME:-taxi-env}"
SERVE_APP="${SERVE_APP:-taxi-serve}"
MLFLOW_APP="${MLFLOW_APP:-taxi-mlflow}"
SERVE_ALGO="${SERVE_ALGO:-auto}"               # auto | qlearning | sarsa | dqn

# Racine du dépôt (le script vit dans azure/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/azure/.build"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- 0. Prérequis ------------------------------------------------------------
step "Vérification de la connexion Azure"
az account show --query "{sub:name, user:user.name}" -o table

step "Enregistrement des resource providers (idempotent)"
az provider register -n Microsoft.App --wait
az provider register -n Microsoft.OperationalInsights --wait
az provider register -n Microsoft.ContainerRegistry --wait
az extension add --name containerapp --upgrade --only-show-errors

# --- 1. Groupe de ressources + registre --------------------------------------
step "Groupe de ressources : $RESOURCE_GROUP ($LOCATION)"
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o none

step "Azure Container Registry : $ACR_NAME"
az acr create -g "$RESOURCE_GROUP" -n "$ACR_NAME" --sku Basic --admin-enabled true -o none

# --- 2. Contexte de staging (contourne le .dockerignore du dépôt) ------------
step "Préparation du contexte de build (staging)"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp -r "$REPO_ROOT/mlops"                    "$BUILD_DIR/mlops"
cp -r "$REPO_ROOT/models"                   "$BUILD_DIR/models"
cp -r "$REPO_ROOT/mlruns"                   "$BUILD_DIR/mlruns"
cp    "$REPO_ROOT/requirements-serve.txt"   "$BUILD_DIR/"
cp    "$REPO_ROOT/azure/Dockerfile.serve"   "$BUILD_DIR/"
cp    "$REPO_ROOT/azure/Dockerfile.mlflow"  "$BUILD_DIR/"
# Pas de .dockerignore ici -> models/ et mlruns/ sont bien inclus.

# --- 3. Build local (Docker) + push vers l'ACR -------------------------------
# NB : on ne peut pas utiliser `az acr build` (ACR Tasks) sur les abonnements
# gratuits/sponsorisés (erreur TasksOperationsNotAllowed). On build donc en
# local avec Docker puis on pousse vers le registre.
ACR_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv)
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

step "Connexion Docker au registre $ACR_SERVER"
echo "$ACR_PASS" | docker login "$ACR_SERVER" -u "$ACR_USER" --password-stdin

step "Build + push de l'image serve"
docker build -t "$ACR_SERVER/$SERVE_APP:latest" -f "$BUILD_DIR/Dockerfile.serve" "$BUILD_DIR"
docker push "$ACR_SERVER/$SERVE_APP:latest"

step "Build + push de l'image mlflow-ui"
docker build -t "$ACR_SERVER/$MLFLOW_APP:latest" -f "$BUILD_DIR/Dockerfile.mlflow" "$BUILD_DIR"
docker push "$ACR_SERVER/$MLFLOW_APP:latest"

# --- 4. Environnement Container Apps -----------------------------------------
step "Environnement Container Apps : $ENV_NAME"
az containerapp env create -g "$RESOURCE_GROUP" -n "$ENV_NAME" -l "$LOCATION" -o none

# --- 5. Déploiement des deux applications (create-or-update) ------------------
# Suffixe de révision unique : force Container Apps à retirer le nouveau :latest
# (sinon, tag identique => pas de nouvelle révision, ancienne image conservée).
REV_SUFFIX="r$(date +%Y%m%d%H%M%S)"
app_exists(){ az containerapp show -g "$RESOURCE_GROUP" -n "$1" -o none 2>/dev/null; }

if app_exists "$SERVE_APP"; then
  step "Mise à jour du serving ($SERVE_APP) — révision $REV_SUFFIX"
  az containerapp update -g "$RESOURCE_GROUP" -n "$SERVE_APP" \
    --image "$ACR_SERVER/$SERVE_APP:latest" \
    --revision-suffix "$REV_SUFFIX" \
    --set-env-vars "TAXI_SERVE_ALGORITHM=$SERVE_ALGO" -o none
else
  step "Déploiement du serving ($SERVE_APP) — ingress public port 8000"
  az containerapp create -g "$RESOURCE_GROUP" -n "$SERVE_APP" \
    --environment "$ENV_NAME" \
    --image "$ACR_SERVER/$SERVE_APP:latest" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --target-port 8000 --ingress external \
    --min-replicas 0 --max-replicas 2 \
    --cpu 0.5 --memory 1.0Gi \
    --env-vars "TAXI_SERVE_ALGORITHM=$SERVE_ALGO" \
    -o none
fi

if app_exists "$MLFLOW_APP"; then
  step "Mise à jour de MLflow UI ($MLFLOW_APP) — révision $REV_SUFFIX"
  az containerapp update -g "$RESOURCE_GROUP" -n "$MLFLOW_APP" \
    --image "$ACR_SERVER/$MLFLOW_APP:latest" \
    --revision-suffix "$REV_SUFFIX" -o none
else
  step "Déploiement de MLflow UI ($MLFLOW_APP) — ingress public port 5000"
  az containerapp create -g "$RESOURCE_GROUP" -n "$MLFLOW_APP" \
    --environment "$ENV_NAME" \
    --image "$ACR_SERVER/$MLFLOW_APP:latest" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --target-port 5000 --ingress external \
    --min-replicas 0 --max-replicas 1 \
    --cpu 0.5 --memory 1.0Gi \
    -o none
fi

# --- 6. Récapitulatif --------------------------------------------------------
SERVE_URL=$(az containerapp show -g "$RESOURCE_GROUP" -n "$SERVE_APP" \
  --query properties.configuration.ingress.fqdn -o tsv)
MLFLOW_URL=$(az containerapp show -g "$RESOURCE_GROUP" -n "$MLFLOW_APP" \
  --query properties.configuration.ingress.fqdn -o tsv)

step "Déploiement terminé 🎉"
echo "  Site (serve)  : https://$SERVE_URL"
echo "  MLflow UI     : https://$MLFLOW_URL"
echo "  Registre ACR  : $ACR_SERVER"
echo
echo "  Test rapide :"
echo "    curl https://$SERVE_URL/health"
echo "    curl -X POST https://$SERVE_URL/predict -H 'Content-Type: application/json' -d '{\"state\": 328}'"
echo
echo "  Nettoyer (tout supprimer) :"
echo "    az group delete -n $RESOURCE_GROUP --yes --no-wait"
