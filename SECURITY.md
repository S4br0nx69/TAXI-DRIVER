# Politique de sécurité

## Versions supportées

| Version | Supportée |
|---------|-----------|
| 1.4.x   | ✅         |
| < 1.4   | ❌         |

## Signaler une vulnérabilité

Si vous découvrez une vulnérabilité de sécurité dans ce projet, merci de la signaler de manière responsable.

**Ne créez pas d'issue publique.** Envoyez un email à :

📧 **sabri.hammi@epitech.eu**

### Informations à fournir

- Description détaillée de la vulnérabilité
- Étapes de reproduction
- Impact potentiel
- Version concernée
- Correctif proposé (si applicable)

### Délai de réponse

| Étape | Délai |
|-------|-------|
| Accusé de réception | 48h |
| Évaluation initiale | 5 jours |
| Correctif déployé | 15 jours |

## Périmètre

### Inclus

- Code source Python (agents RL, modules d'entraînement)
- Dépendances directes (`requirements.txt`)
- Scripts d'exécution (`main.py`, `bruteforce.py`)
- Sérialisation des modèles (`.pth`)

### Exclus

- Environnement Gymnasium (maintenu par Farama Foundation)
- Bibliothèques tierces (PyTorch, NumPy, Matplotlib)
- Environnements locaux (`.venv/`)

## Bonnes pratiques appliquées

- **Isolation des dépendances** : exécution dans un environnement virtuel Python (venv), aucune installation système requise
- **Pas de secrets en dur** : aucune clé API, token ou credential dans le code source
- **Gitignore strict** : `.venv/`, `__pycache__/`, `*.pth` et fichiers générés exclus du versioning
- **Désérialisation contrôlée** : les modèles PyTorch (`.pth`) ne sont chargés que localement via `torch.load()` avec `map_location` explicite
- **Pas d'exécution distante** : aucun appel réseau, aucune API externe, exécution 100% locale
- **Entrées utilisateur validées** : les saisies dans les `main.py` sont castées (`int()`, `float()`) avec des valeurs par défaut sécurisées

## Dépendances

Les versions des dépendances sont fixées pour éviter les régressions :

| Package | Version | CVE connues |
|---------|---------|-------------|
| gymnasium | 1.3.0 | Aucune |
| numpy | 2.4.6 | Aucune |
| matplotlib | 3.10.9 | Aucune |
| torch | 2.12.0 | Aucune |

> Dernière vérification : mai 2026