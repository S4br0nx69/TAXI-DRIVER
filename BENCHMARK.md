# Rapport de Benchmark — Taxi Driver RL

**Date :** 26 juin 2026  
**Branche :** `features/fine-tunning`  
**Environnement :** Gymnasium `Taxi-v3`

---

## 1. Contexte et objectif

Ce rapport compare quatre algorithmes d'apprentissage par renforcement appliqués au problème du Taxi Driver : le taxi doit récupérer un passager et le déposer à destination dans une grille 5×5. L'objectif du fine-tuning est d'identifier, pour chaque algorithme, les hyperparamètres qui maximisent la récompense et minimisent le nombre de steps, à partir d'une recherche par grille (grid search).

Les quatre algorithmes évalués sont :
- **Q-Learning** — méthode off-policy par différence temporelle (TD)
- **SARSA** — méthode on-policy TD, avec variantes Expected SARSA et n-step
- **Monte Carlo** — apprentissage par retours complets d'épisodes
- **DQN** (Deep Q-Network) — Q-Learning avec réseau de neurones et replay buffer

---

## 2. Méthodologie

### 2.1 Protocole en trois phases

| Phase | Description |
|-------|-------------|
| **Baseline** | Évaluation avec les hyperparamètres par défaut de chaque modèle |
| **Grid Search** | Exploration par produit cartésien de l'espace des hyperparamètres |
| **Final** | Ré-entraînement avec les meilleurs paramètres identifiés |

Chaque phase entraîne un agent vierge (Q-table ou réseau réinitialisé) sur **10 000 épisodes**, puis l'évalue sur **100 épisodes de test** en politique greedy (ε = 0).

### 2.2 Métriques

| Métrique | Description |
|----------|-------------|
| **Reward moyen** | Récompense cumulée moyenne par épisode de test. Plus élevé = meilleur. |
| **Steps moyen** | Nombre d'actions moyen par épisode. Plus bas = plus efficace. |
| **Penalties moyennes** | Nombre d'actions illégales (pickup/dropoff hors case valide, reward = -10). Idéalement 0. |
| **Taux de complétion** | Pourcentage d'épisodes où le taxi livre le passager à destination (`terminated = True`). |

Le taux de complétion est distinct du reward : un agent peut terminer chaque épisode avec succès mais en prenant un chemin long (reward faible), ou inversement se faire tronquer par la limite de steps avant d'arriver à destination (pas de complétion malgré un reward négatif accumulé).

### 2.3 Espaces de recherche

**Q-Learning** (27 combinaisons)

| Paramètre | Valeurs testées |
|-----------|----------------|
| `alpha` (learning rate) | 0.05, 0.1, 0.2 |
| `gamma` (discount factor) | 0.6, 0.8, 0.99 |
| `epsilon_decay` | 0.999, 0.9995, 0.9999 |

**SARSA** (32 combinaisons)

| Paramètre | Valeurs testées |
|-----------|----------------|
| `alpha` | 0.1, 0.2 |
| `gamma` | 0.9, 0.99 |
| `epsilon_decay` | 0.999, 0.9995 |
| `policy_type` | `epsilon_greedy`, `expected` |
| `n_steps` | 1, 3 |

**Monte Carlo** (36 combinaisons)

| Paramètre | Valeurs testées |
|-----------|----------------|
| `alpha` | 0.05, 0.1 |
| `gamma` | 0.9, 0.95, 0.99 |
| `epsilon_decay` | 0.9995, 0.9997, 0.9999 |
| `visit_mode` | `first_visit`, `every_visit` |

**DQN** (16 combinaisons)

| Paramètre | Valeurs testées |
|-----------|----------------|
| `lr` | 0.0005, 0.001 |
| `gamma` | 0.95, 0.99 |
| `batch_size` | 32, 64 |
| `optimizer_type` | `adam`, `rmsprop` |

---

## 3. Résultats baseline

Évaluation avec les paramètres par défaut de chaque modèle, avant tout fine-tuning.

| Modèle | Reward | Steps | Penalties | Complétion |
|--------|-------:|------:|----------:|-----------:|
| Q-Learning | -18.91 | 37.18 | 0.00 | 87 % |
| SARSA | -8.76 | 28.08 | 0.00 | 92 % |
| Monte Carlo | -110.33 | 119.36 | 0.00 | 43 % |
| DQN | **+7.54** | **13.46** | 0.00 | **100 %** |

**Paramètres par défaut utilisés**

| Modèle | Paramètres |
|--------|-----------|
| Q-Learning | α=0.1, γ=0.6, ε-decay=0.9995 |
| SARSA | α=0.2, γ=0.9, ε-decay=0.999, epsilon_greedy, n_steps=1 |
| Monte Carlo | α=0.05, γ=0.95, ε-decay=0.9997, first_visit |
| DQN | lr=0.001, γ=0.99, batch=64, adam |

**Observations :**

- Le **DQN** est le seul modèle performant dès la baseline (reward positif, 100 % de complétion), grâce à la capacité de généralisation de son réseau de neurones.
- **Q-Learning** et **SARSA** complètent la majorité des épisodes (87–92 %) mais accumulent des récompenses négatives : ils livrent le passager, mais empruntent des chemins trop longs (chaque step non optimal coûte -1).
- **Monte Carlo** est le plus en difficulté : seulement 43 % de complétion. Le mode `first_visit` avec γ=0.95 converge lentement car les mises à jour n'interviennent qu'en fin d'épisode, et la décroissance d'epsilon maintient une exploration excessive.
- **Zéro penalties** sur tous les modèles dès la baseline : les algorithmes apprennent rapidement à éviter les actions illégales, même sans fine-tuning.

---

## 4. Analyse du Grid Search

### 4.1 Principaux enseignements par modèle

**Q-Learning**  
Le facteur le plus déterminant est `gamma`. Avec γ=0.6 (défaut), l'agent valorise trop peu les récompenses futures et se contente d'actions localement acceptables. Passer à γ ≥ 0.99 permet de planifier sur le long terme et d'atteindre systématiquement des rewards positifs. Le learning rate `alpha` a un impact secondaire : des valeurs faibles (0.05) associées à un gamma élevé restent stables et performantes.

**SARSA**  
La variante **Expected SARSA** (`policy_type=expected`) se distingue positivement sur plusieurs configurations : elle produit des mises à jour plus stables en moyennant sur toutes les actions possibles plutôt qu'en suivant aveuglément la prochaine action choisie. Le paramètre `n_steps` a un impact contrasté : n_steps=1 est plus robuste, le retour 3-step introduisant trop de variance sur des épisodes courts.

**Monte Carlo**  
Le résultat le plus marquant du grid search : le mode `first_visit` est systématiquement sous-performant, et devient catastrophique avec γ=0.99 (rewards inférieurs à -300, avec penalties > 0). Le mode **`every_visit`** corrige ce problème en mettant à jour la Q-table à chaque passage par une paire (état, action), accélérant significativement la convergence. Les configurations `first_visit` + γ=0.99 sont les seules du benchmark à générer des penalties > 0, révélant une divergence complète de l'apprentissage.

**DQN**  
Déjà performant en baseline, les gains du grid search sont plus marginaux. L'optimiseur **Adam** avec un faible learning rate (lr=0.0005) et un petit batch (batch=32) offre la meilleure stabilité. RMSProp produit des résultats plus variables selon les configurations.

### 4.2 Meilleurs paramètres identifiés

| Modèle | Meilleurs paramètres |
|--------|---------------------|
| Q-Learning | α=0.05, γ=0.99, ε-decay=0.999 |
| SARSA | α=0.1, γ=0.9, ε-decay=0.999, expected, n_steps=1 |
| Monte Carlo | α=0.1, γ=0.99, ε-decay=0.9995, every_visit |
| DQN | lr=0.0005, γ=0.99, batch=32, adam |

---

## 5. Résultats après fine-tuning

Évaluation après ré-entraînement avec les meilleurs hyperparamètres du grid search.

| Modèle | Reward | Steps | Penalties | Complétion |
|--------|-------:|------:|----------:|-----------:|
| Q-Learning | **+8.12** | **12.88** | 0.00 | **100 %** |
| SARSA | +5.62 | 15.17 | 0.00 | 99 % |
| Monte Carlo | +7.39 | 13.61 | 0.00 | **100 %** |
| DQN | +8.07 | 12.93 | 0.00 | **100 %** |

---

## 6. Analyse comparative : avant / après fine-tuning

| Modèle | Δ Reward | Δ Steps | Δ Complétion |
|--------|:--------:|:-------:|:------------:|
| Q-Learning | +27.0 | -24.3 | +13 pp |
| SARSA | +14.4 | -12.9 | +7 pp |
| Monte Carlo | **+117.7** | **-105.8** | **+57 pp** |
| DQN | +0.5 | -0.5 | 0 pp |

**Q-Learning** enregistre la progression la plus nette parmi les méthodes tabulaires : le passage de γ=0.6 à γ=0.99 est seul responsable de l'essentiel du gain, en forçant l'agent à planifier sur l'horizon complet de l'épisode.

**SARSA** s'améliore mais reste légèrement en retrait sur le reward final. La variante Expected SARSA produit des mises à jour plus conservatrices, ce qui se traduit par une politique un peu moins agressive (plus de steps pour les mêmes livraisons réussies).

**Monte Carlo** est le grand bénéficiaire du fine-tuning : +117 points de reward, -106 steps, taux de complétion multiplié par 2.3. Le seul changement structurel — passer de `first_visit` à `every_visit` — corrige un problème fondamental de convergence sur des épisodes de longueur variable.

**DQN** était déjà proche de l'optimum en baseline ; le fine-tuning apporte une amélioration marginale. Il confirme sa robustesse mais montre aussi ses limites sur un environnement à espace d'états discret et fini : les méthodes tabulaires, une fois bien paramétrées, l'atteignent ou le dépassent.

---

## 7. Conclusion

### 7.1 Classement final

| Rang | Modèle | Reward | Steps | Complétion |
|------|--------|-------:|------:|-----------:|
| 1 | **Q-Learning** | 8.12 | 12.88 | 100 % |
| 2 | **DQN** | 8.07 | 12.93 | 100 % |
| 3 | **Monte Carlo** | 7.39 | 13.61 | 100 % |
| 4 | **SARSA** | 5.62 | 15.17 | 99 % |

Q-Learning et DQN sont pratiquement ex-æquo en performance finale. Q-Learning atteint ce niveau avec une architecture bien plus simple (table d'états discrets), ce qui illustre que sur un environnement à espace d'états fini et relativement petit comme Taxi-v3, les méthodes tabulaires correctement configurées peuvent égaler un réseau de neurones.

### 7.2 Impact du fine-tuning

Le fine-tuning a un impact décisif sur les trois méthodes tabulaires. Tous atteignent 99–100 % de complétion et des rewards positifs après grid search, contre des rewards négatifs et des taux de complétion partiels en baseline. Le DQN, grâce à ses capacités de généralisation inhérentes, bénéficie peu du fine-tuning mais confirme sa robustesse dès le départ.

Le paramètre le plus influent varie selon l'algorithme :
- **Q-Learning / Monte Carlo** : `gamma` — la portée temporelle de la récompense est critique
- **SARSA** : `policy_type` — la stratégie de mise à jour (expected vs greedy) prime
- **Monte Carlo** : `visit_mode` — le mode `every_visit` est indispensable à la convergence

### 7.3 Zéro penalties

Tous les modèles maintiennent un taux de penalties nul sur les épisodes de test, y compris en baseline. Ce résultat confirme que les algorithmes apprennent rapidement à ne jamais tenter de pickup ou dropoff invalides. Le taux de complétion est donc l'indicateur le plus discriminant pour comparer les modèles sur cet environnement, bien plus que les penalties qui convergent toutes à zéro dès la phase d'entraînement.
