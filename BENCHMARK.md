# Rapport de Benchmark — Taxi Driver RL

**Date :** 30 juin 2026  
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

## 3. Résultats baseline _(26 juin 2026)_

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

### 4.2 Meilleurs paramètres identifiés _(26 juin 2026 — révisés en section 8.2)_

| Modèle | Meilleurs paramètres |
|--------|---------------------|
| Q-Learning | α=0.05, γ=0.99, ε-decay=0.999 |
| SARSA | α=0.1, γ=0.9, ε-decay=0.999, expected, n_steps=1 |
| Monte Carlo | α=0.1, γ=0.99, ε-decay=0.9995, every_visit |
| DQN | lr=0.0005, γ=0.99, batch=32, adam |

---

## 5. Résultats après fine-tuning _(26 juin 2026)_

Évaluation après ré-entraînement avec les meilleurs hyperparamètres du grid search.

| Modèle | Reward | Steps | Penalties | Complétion |
|--------|-------:|------:|----------:|-----------:|
| Q-Learning | **+8.12** | **12.88** | 0.00 | **100 %** |
| SARSA | +5.62 | 15.17 | 0.00 | 99 % |
| Monte Carlo | +7.39 | 13.61 | 0.00 | **100 %** |
| DQN | +8.07 | 12.93 | 0.00 | **100 %** |

---

## 6. Analyse comparative : avant / après fine-tuning _(26 juin 2026)_

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

---

## 8. Mise à jour — 30 juin 2026

Re-run complet du benchmark avec les implémentations finalisées. Les paramètres par défaut n'ont pas changé, mais les métriques baseline reflètent les implémentations stabilisées. Le grid search a été validé sur plusieurs seeds (3 à 4), ce qui a conduit à réviser certains meilleurs paramètres.

### 8.1 Baseline ré-évalué (30 juin 2026)

*(Mêmes paramètres par défaut qu'en section 3)*

| Modèle | Reward | Steps | Penalties | Complétion | Δ vs 26 juin |
|--------|-------:|------:|----------:|-----------:|:------------|
| Q-Learning | -11.05 | 30.09 | 0.00 | 91 % | reward +7.86, steps -7.09, complétion +4 pp |
| SARSA | +2.21 | 18.30 | 0.00 | 98 % | reward +10.97, steps -9.78, complétion +6 pp |
| Monte Carlo | +4.48 | 16.24 | 0.00 | 99 % | reward **+114.81**, steps **-103.12**, complétion **+56 pp** |
| DQN | **+8.11** | **12.89** | 0.00 | **100 %** | reward +0.57, steps -0.57, complétion 0 pp |

**Observations :** Monte Carlo et SARSA bénéficient le plus des implémentations finalisées, même à paramètres constants. DQN reste stable et dominant en baseline. Le réseau de neurones du DQN conserve son avantage de généralisation sans fine-tuning.

### 8.2 Meilleurs paramètres corrigés (validation multi-seeds)

La validation sur 3–4 seeds des top candidats du grid search a conduit à réviser les paramètres de SARSA, Monte Carlo et DQN.

| Modèle     | Paramètres (26 juin) | Paramètres corrigés (30 juin) | Ce qui a changé |
|--------    |---------------------|-------------------------------|----------------|
| Q-Learning | α=0.05, γ=0.99, ε-decay=0.999 | α=0.05, γ=0.99, ε-decay=0.999 | — Inchangé |
| SARSA      | α=0.1, γ=0.9, ε-decay=0.999, expected, n_steps=1 | α=0.1, **γ=0.99**, ε-decay=**0.9995**, **epsilon_greedy, n_steps=3** | γ, policy_type, n_steps |
| Monte Carlo | α=0.1, γ=0.99, ε-decay=0.9995, **every_visit** | α=0.1, **γ=0.95**, ε-decay=**0.9997**, **first_visit** | γ, ε-decay, visit_mode |
| DQN | lr=0.0005, **γ=0.99**, batch=32, adam | lr=0.0005, **γ=0.95**, batch=32, adam | γ |

**Explication des révisions :**

- **SARSA** : la validation multi-seeds montre que `epsilon_greedy` + `n_steps=3` + `gamma=0.99` surpasse Expected SARSA. À seed unique, Expected SARSA semblait plus stable, mais sa variance inter-seeds est en réalité plus élevée.
- **Monte Carlo** : `first_visit` gagne sur 4 seeds (reward 7.38 vs 6.43 pour `every_visit`). Le résultat à seed unique était trompeur. `gamma=0.99` reste instable pour cet algorithme ; `gamma=0.95` offre un meilleur compromis biais/variance.
- **DQN** : `gamma=0.95` (au lieu de 0.99) est confirmé plus stable sur plusieurs seeds ; les métriques finales restent néanmoins identiques.

### 8.3 Résultats après fine-tuning (30 juin 2026)

*(3 seeds : 0, 1, 2 — paramètres corrigés de la section 8.2)*

| Modèle | Reward | ±σ | Steps | ±σ | Penalties | Complétion | Δ vs 26 juin |
|--------|-------:|---:|------:|---:|----------:|-----------:|:------------|
| Q-Learning | +8.11 | 0.23 | 12.89 | 0.23 | 0.00 | **100 %** | -0.01 reward, stable |
| **SARSA** | **+8.21** | 0.25 | **12.79** | 0.25 | 0.00 | **100 %** | **+2.59 reward, -2.38 steps, +1 pp** |
| Monte Carlo | +7.61 | 0.23 | 13.39 | 0.23 | 0.00 | **100 %** | +0.22 reward, -0.22 steps |
| DQN | +8.11 | 0.23 | 12.89 | 0.23 | 0.00 | **100 %** | +0.04 reward, stable |

SARSA est le seul modèle à progresser significativement entre les deux runs de fine-tuning, grâce aux paramètres corrigés (gamma=0.99, n_steps=3).

### 8.4 Analyse comparative baseline → fine-tuning (30 juin 2026)

| Modèle | Δ Reward | Δ Steps | Δ Complétion |
|--------|:--------:|:-------:|:------------:|
| **Q-Learning** | **+19.2** | **-17.2** | +9 pp |
| SARSA | +6.0 | -5.5 | +2 pp |
| Monte Carlo | +3.1 | -2.9 | +1 pp |
| DQN | 0.0 | 0.0 | 0 pp |

**Q-Learning** reste le plus grand bénéficiaire relatif du fine-tuning (+19.2 reward), porté par le seul passage de γ=0.6 à γ=0.99. SARSA et Monte Carlo montrent des gains plus modestes car leurs baselines sont désormais bien meilleures. **DQN** est invariant — il atteignait déjà l'optimum sans fine-tuning.

### 8.5 Classement final révisé (30 juin 2026)

| Rang | Modèle | Reward | Steps | Complétion |
|------|--------|-------:|------:|-----------:|
| 1 | **SARSA** | 8.21 | 12.79 | 100 % |
| 2 | **Q-Learning** | 8.11 | 12.89 | 100 % |
| 2 | **DQN** | 8.11 | 12.89 | 100 % |
| 4 | **Monte Carlo** | 7.61 | 13.39 | 100 % |

**SARSA** prend la première place grâce aux paramètres corrigés, détrônant Q-Learning. Q-Learning et DQN sont désormais strictement ex-æquo (métriques identiques). Tous les modèles atteignent 100 % de complétion : le reward et les steps sont les seuls critères discriminants.

Le paramètre le plus influent par algorithme, revisité :
- **Q-Learning** : `gamma` — passage 0.6 → 0.99, critique
- **SARSA** : `gamma` + `n_steps` — γ=0.99 et retour 3-step sont décisifs
- **Monte Carlo** : `gamma` — 0.99 trop instable, 0.95 est l'optimum confirmé multi-seeds
- **DQN** : robuste par nature, insensible aux ajustements de cette amplitude

---

## 9. Annexe technique — pipeline de fine-tuning

Le code source garde des commentaires courts (une ligne par point) ; cette section rassemble le détail : comment tourne le pipeline, ce que fait chaque hyperparamètre, et l'historique des bugs corrigés.

### 9.1 Deux outils, deux portées

- **`grid_search.py`** : script autonome, Q-Learning uniquement, grille fixe codée en dur (`alpha` × `gamma` × `epsilon_decay`, 48 combinaisons). Premier outil de fine-tuning du projet, conservé pour un balayage rapide et isolé.
- **`benchmark.py`** : orchestrateur général des 4 modèles, en 4 phases indépendantes et rejouables (chaque phase lit/écrit son propre JSON dans `results/`) :

  | Phase | Rôle | Dépend de |
  |-------|------|-----------|
  | `baseline` | Évalue chaque modèle avec ses params par défaut (`DEFAULT_PARAMS`) — le point de référence t=0 | rien |
  | `grid` | Explore `GRID_PARAMS` par produit cartésien, par modèle | rien (mais comparé à `baseline` en phase 4) |
  | `final` | Ré-entraîne avec les meilleurs params trouvés en `grid` | `results/grid_search.json` |
  | `compare` | Charge `baseline.json` + `final.json`, produit le graphe et le tableau avant/après | `results/baseline.json` + `results/final.json` |

### 9.2 Pourquoi multi-seeds (`run_model`, `--repeats`)

Un seul run d'entraînement dépend de l'initialisation aléatoire (Q-table, ordre des transitions explorées) : la « meilleure » combinaison d'un balayage à 1 seed peut n'être meilleure que par chance de tirage, pas par qualité réelle des hyperparamètres. `run_model()` (dans `benchmark.py`) et l'option `--repeats` (dans `grid_search.py`) répètent l'entraînement sur plusieurs seeds et moyennent les métriques, avec un écart-type (`*_std`) qui permet de juger si un écart entre deux configurations est significatif ou n'est que du bruit.

`phase_grid()` applique cette idée en deux temps pour limiter le coût : un balayage large à 1 seed sert de présélection (`--grid-seed`), puis seules les `--refine-top-k` meilleures combinaisons sont ré-évaluées sur plusieurs seeds (`--refine-seeds`) pour départager le vrai meilleur. La section 8.2 documente un cas concret où ce raffinement a changé le classement (SARSA `expected` semblait meilleur à seed unique, `epsilon_greedy` + n_steps=3 gagne en réalité sur 4 seeds).

### 9.3 Glossaire des hyperparamètres de fine-tuning

**Communs aux 4 modèles**
- `alpha` : taux d'apprentissage — vitesse à laquelle la Q-table intègre une nouvelle estimation.
- `gamma` : facteur de discount — poids donné aux récompenses futures vs immédiates.
- `epsilon_decay` : vitesse de décroissance de l'exploration (epsilon-greedy) au fil des épisodes.

**Q-Learning**
- `optimistic_init` : valeur initiale de la Q-table (0 = neutre, >0 = optimiste, force l'exploration initiale).
- `double_q` : Double Q-Learning — deux Q-tables pour réduire le biais de surestimation du max.

**SARSA**
- `policy_type` : `epsilon_greedy` (standard), `softmax` (sélection par distribution Boltzmann via `temperature`), `expected` (bootstrap sur la valeur espérée sous la politique plutôt que sur l'action suivante réellement choisie).
- `n_steps` : 1 = SARSA standard, n>1 = retour sur n pas avant bootstrap (horizon plus long, plus de variance).
- `lambda_` : SARSA(λ), traces d'éligibilité — généralise SARSA (λ=0) et Monte Carlo (λ→1), propage la récompense sur tout le chemin de l'épisode.

**Monte Carlo**
- `visit_mode` : `first_visit` (met à jour Q seulement à la première occurrence d'une paire état-action dans l'épisode) vs `every_visit` (à chaque occurrence).
- `exploring_starts` : démarre chaque épisode sur un couple (état, action) aléatoire, pour garantir la couverture de l'espace sans dépendre uniquement d'epsilon.

**DQN**
- `optimizer_type` : `adam`, `rmsprop`, `sgd` — algorithme d'optimisation du réseau.
- `hidden_sizes` : architecture des couches cachées du réseau.
- `double_dqn` : le réseau policy choisit l'action, le réseau target l'évalue — réduit la surestimation.
- `dueling` : sépare l'estimation en deux flux V(s) (valeur de l'état) et A(s,a) (avantage de l'action) — utile quand plusieurs actions ont une valeur proche.
- `tau` : soft update du target network (0 = copie périodique complète, >0 = mélange continu τ·θ_policy + (1-τ)·θ_target).
- `use_factored_encoding` (flag, désactivé par défaut) : remplace l'encodage one-hot de l'état (qui rend chaque état orthogonal aux autres, sans généralisation possible) par un vecteur factorisé (position taxi / passager / destination), pour vérifier si l'égalité de performance DQN ≈ Q-Learning observée sur Taxi-v3 est un artefact de l'encodage.

### 9.4 Historique des bugs corrigés

- **SARSA — `policy_type` ignoré en n-step et en λ** : `_train_nstep()` et `_train_lambda()` bootstrapaient toujours sur `Q[s', a']` (SARSA standard) même quand `policy_type='expected'` était configuré. Le paramètre n'avait donc aucun effet dès que `n_steps>1`. Corrigé en appelant `_expected_value()` dans le bootstrap final quand `policy_type='expected'`.
- **Monte Carlo — `first_visit` était en réalité un `last-visit`** : l'implémentation parcourait l'épisode à l'envers en marquant les paires (état, action) comme « déjà vues » au fur et à mesure, ce qui faisait déclencher la mise à jour sur la *dernière* occurrence chronologique plutôt que la première. Corrigé par un calcul en deux passes : une passe avant identifie l'indice de la première occurrence de chaque paire, puis la passe arrière habituelle (nécessaire pour calculer le retour cumulé G) ne met à jour Q qu'à cet indice en mode `first_visit`. Ce bug explique une bonne partie de la sous-performance spectaculaire de `first_visit` observée avant correction.
- **Crash post-fusion avec `main` (ajout de `completion_rate`)** : le passage de `test()` d'un tuple `(steps, penalties, reward)` à un dict (`{'steps', 'penalties', 'reward', 'completion_rate'}`) sur les 4 agents n'avait pas été répercuté partout. `grid_search.py` et `tests/test_agents.py` continuaient à déballer 3 valeurs (`ValueError: too many values to unpack`), cassant l'exécution et la CI. `benchmark.py` était déjà correct. Corrigé en passant ces deux fichiers à l'accès par clés.
- **Couverture de tests DQN manquante** : `tests/test_agents.py` avait un test d'entraînement pour Q-Learning, SARSA et Monte Carlo, mais aucun pour DQN — un bug dans `deep_q_learning.py` (import, constructeur, contrat de retour de `test()`) serait passé inaperçu en CI. Ajout de `test_dqn_train()` et `test_dqn_test_returns_dict()` sur le même modèle que les 3 autres.

### 9.5 Limites connues du grid search (non corrigées, choix de scope)

- `epsilon_decay=0.9999` combiné à un budget de 10 000 épisodes laisse ~37 % d'exploration résiduelle en fin d'entraînement (`epsilon_min` jamais atteint) : une combinaison avec cette valeur peut sembler sous-performante simplement parce qu'elle n'a pas fini de converger.
- Le grid SARSA ne couvre pas la même plage de `gamma` que Q-Learning/Monte Carlo (0.9/0.99 contre 0.6/0.8/0.99) : la comparaison inter-algorithmes sur l'effet de gamma n'est donc pas totalement équitable.
- Le grid DQN n'explore pas `epsilon_decay`, `target_update`, ni la taille du replay buffer ou l'architecture du réseau — tous fixés à leur valeur par défaut.
