# Audit de sécurité et corrections du code

## Fix 1 — Désérialisation non sécurisée (torch.load)

La méthode `load()` de la classe `DQNAgent` utilisait `torch.load()` sans restriction. Par défaut, `torch.load()` utilise le module `pickle` de Python pour désérialiser les fichiers `.pth`, ce qui permet l'exécution de code arbitraire si un fichier malveillant est fourni. Un attaquant pourrait embarquer du code dans un faux modèle pré-entraîné et compromettre la machine de l'utilisateur au moment du chargement.

La correction consiste à ajouter le paramètre `weights_only=True` à l'appel `torch.load()`. Ce paramètre restreint la désérialisation aux seuls tenseurs et types primitifs Python, bloquant toute tentative d'exécution de code embarqué. Le chargement des poids du réseau (state_dict) fonctionne normalement puisqu'il ne contient que des tenseurs.

Avant :
```python
checkpoint = torch.load(path, map_location=self.device)
```

Après :
```python
checkpoint = torch.load(path, map_location=self.device, weights_only=True)
```

## Fix 2 — Validation des entrées utilisateur

Les fichiers `main.py` utilisaient des conversions directes (`int(input(...))`, `float(input(...))`) sans aucune vérification. Ce manque de validation exposait le programme à trois risques : un crash immédiat si l'utilisateur saisit du texte au lieu d'un nombre (`ValueError`), un comportement imprévisible avec des valeurs négatives ou nulles (épisodes négatifs, learning rate à 0), et un déni de service involontaire si l'utilisateur entre un nombre d'épisodes excessif qui bloque la machine pendant des heures.

Un fichier `utils.py` a été créé à la racine du projet pour centraliser deux fonctions de validation : `safe_input_int()` et `safe_input_float()`. Chaque fonction encapsule la saisie dans un `try/except`, vérifie que la valeur est comprise dans des bornes définies, et applique la valeur par défaut en cas d'erreur ou de dépassement. La centralisation dans un fichier unique évite la duplication de code dans les 4 fichiers `main.py` et garantit un comportement cohérent sur tous les algorithmes.

Bornes appliquées :

| Paramètre | Type | Min | Max |
|-----------|------|-----|-----|
| Épisodes d'entraînement | int | 1 | 500 000 |
| Épisodes de test | int | 1 | 1 000 |
| Learning rate (α) | float | 0.001 | 1.0 |
| Discount factor (γ) | float | 0.0 | 1.0 |
| Epsilon initial | float | 0.0 | 1.0 |
| Epsilon minimum | float | 0.0 | 1.0 |
| Epsilon decay | float | 0.9 | 1.0 |

## Fix 3 — Application de la validation dans les main.py

Les 4 fichiers `main.py` (Q_learning, Sarsa, MonteCarlo, deep_Q_learning) ont été modifiés pour importer les fonctions de validation depuis `utils.py` via `sys.path.append('..')`. Chaque appel `int(input(...))` a été remplacé par `safe_input_int()` et chaque `float(input(...))` par `safe_input_float()`, avec les bornes adaptées à chaque algorithme.

Cette modification est transparente pour l'utilisateur, le comportement par défaut (Entrée = valeur par défaut) reste identique. La différence se manifeste uniquement en cas de saisie invalide, où le programme affiche un message explicatif et applique la valeur par défaut au lieu de crasher.