"""Benchmark des 4 modèles RL — capture de métriques avant/pendant/après fine-tuning.

Usage:
    python benchmark.py baseline [--train-episodes N] [--test-episodes N] [--seeds 0,1,2] [--no-interactive]
    python benchmark.py grid     [--train-episodes N] [--test-episodes N] [--grid-seed 0] [--refine-top-k 3] [--refine-seeds 0,1,2,3] [--no-interactive]
    python benchmark.py final    [--train-episodes N] [--test-episodes N] [--seeds 0,1,2] [--no-interactive]
    python benchmark.py compare

Corrections apportées (cf. revue de code) :
- run_model() accepte maintenant une liste de seeds et agrège (moyenne +
  écart-type INTER-seed) sur chacune des métriques. Avant, chaque run était
  unique et non reproductible (aucun seed nulle part) : le classement final
  entre modèles ne reposait sur aucune mesure de dispersion, ce qui rendait
  impossible de distinguer un vrai écart de performance d'un simple bruit
  d'initialisation.
- phase_baseline() et phase_final() utilisent plusieurs seeds par défaut
  (--seeds, ex: "0,1,2") : les JSON produits contiennent maintenant
  metric_std pour chaque métrique, et les graphes affichent des barres
  d'erreur.
- phase_grid() corrige le problème d'argmax sur un run unique bruité : le
  balayage initial reste à 1 seed par combinaison (coût), mais les
  --refine-top-k meilleures combinaisons (par reward) sont ensuite
  ré-évaluées sur --refine-seeds seeds avant de désigner le "meilleur" —
  on choisit donc le meilleur en espérance, pas par chance de tirage.
  Le JSON conserve maintenant 'refined_top_candidates' en plus de
  'all_results', pour audit.
- Les fonctions de plotting (_plot_snapshot, phase_compare) affichent des
  barres d'erreur (yerr) quand les champs *_std sont présents, avec un
  fallback à 0 pour rester compatible avec d'anciens fichiers JSON sans std.
"""

import sys
import os
import json
import curses
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product

# Chaque modèle vit dans son propre dossier sans __init__.py ;
# sys.path.insert permet l'import direct sans restructurer le projet.
sys.path.insert(0, 'Q_learning')
sys.path.insert(0, 'Sarsa')
sys.path.insert(0, 'MonteCarlo')
sys.path.insert(0, 'deep_Q_learning')

import q_learning as QL
import sarsa as SRS
import monte_carlo as MC
import deep_q_learning as DQN

# Dossiers de sortie : results/ pour les JSON, results/plots/ pour les PNG.
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Un fichier JSON par phase — chaque phase est indépendante et rejouable.
# baseline  → métriques avec params par défaut (point de départ)
# grid      → toutes les combinaisons testées + meilleurs params par modèle
# final     → métriques après application des meilleurs params (point d'arrivée)
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline.json")
GRID_FILE     = os.path.join(RESULTS_DIR, "grid_search.json")
FINAL_FILE    = os.path.join(RESULTS_DIR, "final.json")

MODEL_COLORS = {
    'q_learning':  '#2196F3',
    'sarsa':       '#4CAF50',
    'monte_carlo': '#FF9800',
    'dqn':         '#E91E63',
}

MODEL_LABELS = {
    'q_learning':  'Q-Learning',
    'sarsa':       'SARSA',
    'monte_carlo': 'Monte Carlo',
    'dqn':         'DQN',
}

# Défauts d'épisodes calibrés par algorithme sur Taxi-v3.
# Q-Learning/SARSA : convergence rapide (~5k-8k épisodes), test 200 pour réduire le bruit.
# Monte Carlo : mises à jour uniquement en fin d'épisode, convergence plus lente → plus d'épisodes.
# DQN : replay buffer réutilise chaque expérience plusieurs fois → converge vite,
#        mais chaque épisode coûte ~100x plus cher (forward+backward pass réseau).
MODEL_EPISODE_DEFAULTS = {
    'q_learning':  {'train': 8000,  'test': 200},
    'sarsa':       {'train': 8000,  'test': 200},
    'monte_carlo': {'train': 15000, 'test': 200},
    'dqn':         {'train': 4000,  'test': 100},
}

# Params par défaut de chaque modèle tels qu'ils sont définis dans leurs __init__.
# Sert de référence pour la phase baseline ET de base pour le grid search
# (les clés absentes de GRID_PARAMS sont remplies depuis ici).
DEFAULT_PARAMS = {
    'q_learning': {
        'alpha': 0.1, 'gamma': 0.6, 'epsilon_decay': 0.9995,
        'optimistic_init': 0.0, 'double_q': False,
    },
    'sarsa': {
        'alpha': 0.2, 'gamma': 0.9, 'epsilon_decay': 0.999,
        'policy_type': 'epsilon_greedy', 'n_steps': 1, 'temperature': 1.0,
        'lambda_': 0.0,
    },
    'monte_carlo': {
        'alpha': 0.05, 'gamma': 0.95, 'epsilon_decay': 0.9997,
        'visit_mode': 'first_visit', 'exploring_starts': False,
    },
    'dqn': {
        'gamma': 0.99, 'epsilon_decay': 0.9995,
        'lr': 0.001, 'batch_size': 64, 'target_update': 10,
        'optimizer_type': 'adam',
        'hidden_sizes': (128, 64), 'double_dqn': False, 'dueling': False, 'tau': 0.0,
    },
}

# Espace de recherche par modèle — seuls les hyperparamètres qui ont un vrai
# impact sur Taxi sont inclus. Le produit cartésien de chaque liste génère
# toutes les combinaisons à tester (voir phase_grid).
# Pour réduire le temps de run : raccourcir les listes ou baisser --train-episodes.
#
# LIMITES CONNUES (documentées, non corrigées ici car ce sont des choix de
# scope plutôt que des bugs) :
# - epsilon_decay=0.9999 combiné à 10 000 épisodes laisse ~37% d'exploration
#   résiduelle en fin d'entraînement (epsilon_min=0.01 jamais atteint) : une
#   combinaison avec cette valeur peut sembler sous-performante simplement
#   parce qu'elle n'a pas fini de converger, pas parce que le reste de ses
#   hyperparamètres est mauvais. train() affiche maintenant un avertissement
#   explicite dans ce cas (voir les fichiers des agents).
# - Le grid SARSA ne couvre pas la même plage de gamma que Q-Learning/MC
#   (0.9/0.99 contre 0.6/0.8/0.99) : une comparaison inter-algorithmes sur
#   "l'effet de gamma" n'est donc pas totalement équitable en l'état.
# - Le grid DQN n'explore pas epsilon_decay, target_update, ni la taille du
#   replay buffer ou l'architecture du réseau — tous fixés à leur défaut.
GRID_PARAMS = {
    # Q-Learning : 3×3×3×2×2 = 108 combinaisons de base
    # optimistic_init : valeur initiale Q-table (0=neutre, 1=optimiste)
    # double_q : Double Q-Learning (réduit l'overestimation bias)
    'q_learning': {
        'alpha':           [0.05, 0.1, 0.2],
        'gamma':           [0.6, 0.8, 0.99],
        'epsilon_decay':   [0.999, 0.9995, 0.9999],
        'optimistic_init': [0.0, 1.0],
        'double_q':        [False, True],
    },
    # SARSA : 2×2×2×2×2×2 = 64 combinaisons
    # policy_type 'expected' = Expected SARSA (bootstraps sur valeur espérée)
    # n_steps > 1 = n-step SARSA (retours à plus long horizon)
    # lambda_ > 0 = SARSA(λ) avec traces d'éligibilité (prioritaire sur n_steps)
    'sarsa': {
        'alpha':         [0.1, 0.2],
        'gamma':         [0.9, 0.99],
        'epsilon_decay': [0.999, 0.9995],
        'policy_type':   ['epsilon_greedy', 'expected'],
        'n_steps':       [1, 3],
        'lambda_':       [0.0, 0.8],
    },
    # Monte Carlo : 2×3×3×2×2 = 72 combinaisons
    # visit_mode 'every_visit' met à jour la Q-table à chaque passage
    # par la même paire (s, a) dans l'épisode, pas seulement la première
    # exploring_starts : départ (s,a) aléatoire pour couvrir tout l'espace
    'monte_carlo': {
        'alpha':             [0.05, 0.1],
        'gamma':             [0.9, 0.95, 0.99],
        'epsilon_decay':     [0.9995, 0.9997, 0.9999],
        'visit_mode':        ['first_visit', 'every_visit'],
        'exploring_starts':  [False, True],
    },
    # DQN : 2×2×2×2×2×2 = 64 combinaisons
    # buffer_size exclu : changer la capacité du replay buffer après __init__
    # n'a pas d'effet (le deque est déjà créé avec l'ancienne valeur).
    # double_dqn : policy_net sélectionne l'action, target_net évalue → réduit l'overestimation
    # dueling : streams V(s) et A(s,a) séparés → meilleure estimation des états non-décisifs
    # tau : soft update du target network (0=hard copy, 0.01=update doux continu)
    'dqn': {
        'lr':             [0.0005, 0.001],
        'gamma':          [0.95, 0.99],
        'batch_size':     [32, 64],
        'optimizer_type': ['adam', 'rmsprop'],
        'double_dqn':     [False, True],
        'dueling':        [False, True],
    },
}


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _parse_seeds(seeds_str):
    """Parse une chaîne 'a,b,c' en tuple d'entiers."""
    return tuple(int(s.strip()) for s in seeds_str.split(',') if s.strip() != '')


# ---------------------------------------------------------------------------
# Sélection interactive
# ---------------------------------------------------------------------------

def interactive_model_selection():
    """Menu curses : navigation ↑↓, Espace pour cocher/décocher, Entrée pour confirmer."""
    models   = list(DEFAULT_PARAMS.keys())
    selected = {m: True for m in models}  # tous cochés par défaut
    cursor   = [0]

    def _draw(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # coché
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # décoché
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)  # ligne active

        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, "Modèles à exécuter", curses.A_BOLD)
            stdscr.addstr(1, 0, "  ↑↓ naviguer   Espace cocher/décocher   Entrée confirmer")

            for i, m in enumerate(models):
                check = "[x]" if selected[m] else "[ ]"
                if i == cursor[0]:
                    attr = curses.color_pair(3) | curses.A_BOLD
                elif selected[m]:
                    attr = curses.color_pair(1)
                else:
                    attr = curses.color_pair(2)
                stdscr.addstr(3 + i, 4, f"{check}  {MODEL_LABELS[m]}", attr)

            stdscr.refresh()
            key = stdscr.getch()

            if key == curses.KEY_UP:
                cursor[0] = max(0, cursor[0] - 1)
            elif key == curses.KEY_DOWN:
                cursor[0] = min(len(models) - 1, cursor[0] + 1)
            elif key == ord(' '):
                selected[models[cursor[0]]] = not selected[models[cursor[0]]]
            elif key in (ord('\n'), ord('\r'), 10, 13, curses.KEY_ENTER):
                break

    curses.wrapper(_draw)

    chosen = [m for m in models if selected[m]]
    if not chosen:
        print("Aucun modèle sélectionné, abandon.")
        sys.exit(0)

    print("Modèles sélectionnés : " + ", ".join(MODEL_LABELS[m] for m in chosen))
    return chosen


def interactive_episode_config(selected_models, default_train, default_test):
    """Demande les épisodes d'entraînement et de test séparément pour chaque modèle.
    Les valeurs par défaut sont calibrées par algorithme (MODEL_EPISODE_DEFAULTS).
    default_train/default_test servent de fallback pour un modèle absent du dict.
    """
    print("\n--- Configuration des épisodes par modèle ---\n")
    config = {}
    for model in selected_models:
        per_model = MODEL_EPISODE_DEFAULTS.get(model, {'train': default_train, 'test': default_test})
        print(f"  {MODEL_LABELS[model]} :")
        train = _ask_int(f"    Épisodes d'entraînement [{per_model['train']}] : ", per_model['train'], 1, 1_000_000)
        test  = _ask_int(f"    Épisodes de test        [{per_model['test']}] : ",  per_model['test'],  1, 10_000)
        config[model] = {'train': train, 'test': test}
        print()
    return config


def _ask_int(prompt, default, min_val, max_val):
    """Saisie sécurisée d'un entier avec bornes et valeur par défaut."""
    try:
        raw = input(prompt).strip()
        val = int(raw) if raw else default
        if not (min_val <= val <= max_val):
            print(f"    Hors bornes [{min_val}–{max_val}], défaut appliqué : {default}")
            return default
        return val
    except ValueError:
        print(f"    Entrée invalide, défaut appliqué : {default}")
        return default


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def _build_agent(model_name, params):
    """Instancie un agent vierge et injecte les hyperparamètres avant l'entraînement.

    Les constructeurs des agents n'acceptent pas de params en argument,
    donc on les écrase sur l'instance après création.
    """
    if model_name == 'q_learning':
        agent = QL.Taxi('ansi')
        agent.alpha           = params.get('alpha', 0.1)
        agent.gamma           = params.get('gamma', 0.6)
        agent.epsilon_decay   = params.get('epsilon_decay', 0.9995)
        agent.optimistic_init = params.get('optimistic_init', 0.0)
        agent.double_q        = params.get('double_q', False)

    elif model_name == 'sarsa':
        agent = SRS.Sarsa('ansi')
        agent.alpha         = params.get('alpha', 0.2)
        agent.gamma         = params.get('gamma', 0.9)
        agent.epsilon_decay = params.get('epsilon_decay', 0.999)
        agent.policy_type   = params.get('policy_type', 'epsilon_greedy')
        agent.n_steps       = params.get('n_steps', 1)
        agent.temperature   = params.get('temperature', 1.0)
        agent.lambda_       = params.get('lambda_', 0.0)

    elif model_name == 'monte_carlo':
        agent = MC.MonteCarlo('ansi')
        agent.alpha             = params.get('alpha', 0.05)
        agent.gamma             = params.get('gamma', 0.95)
        agent.epsilon_decay     = params.get('epsilon_decay', 0.9997)
        agent.visit_mode        = params.get('visit_mode', 'first_visit')
        agent.exploring_starts  = params.get('exploring_starts', False)

    elif model_name == 'dqn':
        agent = DQN.DQNAgent('ansi')
        agent.gamma          = params.get('gamma', 0.99)
        agent.epsilon_decay  = params.get('epsilon_decay', 0.9995)
        agent.lr             = params.get('lr', 0.001)
        agent.batch_size     = params.get('batch_size', 64)
        agent.target_update  = params.get('target_update', 10)
        agent.optimizer_type = params.get('optimizer_type', 'adam')
        agent.hidden_sizes   = params.get('hidden_sizes', (128, 64))
        agent.double_dqn     = params.get('double_dqn', False)
        agent.dueling        = params.get('dueling', False)
        agent.tau            = params.get('tau', 0.0)
        # Reconstruire les réseaux et l'optimiseur après injection des params
        # (dueling peut changer l'architecture, lr/optimizer_type l'optimiseur)
        agent._build_networks()
        agent.optimizer = agent._build_optimizer()

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return agent


def run_model(model_name, params, train_episodes, test_episodes, seeds=(None,)):
    """Unité de travail atomique : pour chaque seed donné, crée un agent
    vierge, l'entraîne, l'évalue. Agrège ensuite les métriques sur l'ensemble
    des seeds (moyenne + écart-type INTER-seed).

    FIX : avant, un seul run non reproductible servait de base à toute
    décision (sélection du "meilleur" en grid search, classement final entre
    modèles). L'écart-type inter-seed retourné ici (`*_std`) permet de juger
    si un écart entre deux configurations est significatif ou relève du
    bruit d'initialisation — ce qui était impossible à évaluer auparavant.

    seeds : tuple de seeds. (None,) reproduit l'ancien comportement (un seul
    run, non reproductible, pas de std calculable). Pour une comparaison
    fiable, utiliser plusieurs seeds, ex: (0, 1, 2).
    """
    per_seed_results = []
    for sd in seeds:
        agent = _build_agent(model_name, params)
        agent.train(train_episodes=train_episodes, training_graph=False, seed=sd)
        test_result = agent.test(test_episodes=test_episodes, fast_testing=True, seed=sd)
        per_seed_results.append(test_result)

    agg = {'n_seeds': len(seeds), 'seeds': [s for s in seeds]}
    for key in ('reward', 'steps', 'penalties', 'completion_rate'):
        values = [r[key] for r in per_seed_results]
        agg[key] = float(np.mean(values))
        # Écart-type INTER-seed : variabilité due aux différentes
        # initialisations/seeds d'entraînement — distinct de reward_std
        # retourné par agent.test() qui mesure la variabilité INTRA-seed
        # (entre les épisodes de test d'un même run).
        agg[f'{key}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
    return agg


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_baseline(selected_models, model_config, seeds=(0,)):
    """Capture le niveau de performance de chaque modèle avec ses params par défaut.
    C'est le point de référence (t=0) qui permettra de mesurer le gain du fine-tuning.
    """
    ensure_dirs()
    print("\n=== PHASE BASELINE ===")
    print(f"Seeds : {seeds} (n={len(seeds)})\n")

    snapshot = {'timestamp': datetime.now().isoformat(), 'seeds': list(seeds), 'models': {}}

    for model_name in selected_models:
        cfg = model_config[model_name]
        print(f"\n--- {MODEL_LABELS[model_name]} "
              f"({cfg['train']} épisodes train, {cfg['test']} test, {len(seeds)} seed(s)) ---")
        metrics = run_model(model_name, DEFAULT_PARAMS[model_name], cfg['train'], cfg['test'], seeds=seeds)
        snapshot['models'][model_name] = {
            'params':  DEFAULT_PARAMS[model_name],
            'metrics': metrics,
        }
        print(f"  reward={metrics['reward']:.2f} (±{metrics['reward_std']:.2f})  "
              f"steps={metrics['steps']:.2f}  penalties={metrics['penalties']:.2f}")

    with open(BASELINE_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nBaseline sauvegardée : {BASELINE_FILE}")

    _plot_snapshot(snapshot, 'baseline')


def phase_grid(selected_models, model_config, grid_seed=0, refine_top_k=3, refine_seeds=(0, 1, 2, 3)):
    """Explore l'espace des hyperparamètres de chaque modèle par produit cartésien.

    FIX : le balayage initial (1 seed, pour limiter le coût) sert seulement
    de présélection. Les `refine_top_k` meilleures combinaisons (par reward)
    sont ensuite ré-évaluées sur `refine_seeds` seeds, et c'est cette
    évaluation multi-seed qui désigne le "meilleur" final — évite de choisir
    une combinaison qui n'a l'air bonne que par chance de tirage sur un
    unique run (argmax sur run bruité).

    Charge les résultats existants si le fichier existe, pour permettre un run partiel.
    """
    ensure_dirs()

    # Charge les résultats existants pour compléter un run interrompu sans tout relancer
    if os.path.exists(GRID_FILE):
        with open(GRID_FILE) as f:
            grid_results = json.load(f)
    else:
        grid_results = {'timestamp': datetime.now().isoformat(), 'models': {}}

    for model_name in selected_models:
        param_grid = GRID_PARAMS[model_name]
        cfg        = model_config[model_name]
        print(f"\n=== Grid Search — {MODEL_LABELS[model_name]} "
              f"({cfg['train']} épisodes train, {cfg['test']} test, "
              f"balayage seed={grid_seed}) ===")
        keys   = list(param_grid.keys())
        # product() génère toutes les combinaisons possibles des valeurs listées
        combos = list(product(*param_grid.values()))
        print(f"{len(combos)} combinaisons\n")

        results = []
        for i, combo in enumerate(combos):
            # Fusion avec DEFAULT_PARAMS : les hyperparamètres absents de la grille
            # (ex. temperature pour SARSA) gardent leur valeur par défaut.
            params = {**DEFAULT_PARAMS[model_name], **dict(zip(keys, combo))}
            try:
                metrics = run_model(model_name, params, cfg['train'], cfg['test'], seeds=(grid_seed,))
                # On ne sauvegarde que les clés de la grille (pas les params fixes)
                # pour garder all_results lisible
                results.append({'params': dict(zip(keys, combo)), 'metrics': metrics})
                print(f"  [{i+1}/{len(combos)}] {dict(zip(keys, combo))} "
                      f"→ reward={metrics['reward']:.2f}  steps={metrics['steps']:.2f}")
            except Exception as e:
                print(f"  [{i+1}/{len(combos)}] ERREUR {dict(zip(keys, combo))}: {e}")

        if not results:
            continue

        # FIX : ne pas désigner le "meilleur" sur l'argmax d'un balayage à
        # un seul seed. On présélectionne les `refine_top_k` meilleures
        # combinaisons sur ce critère, puis on les ré-évalue chacune sur
        # plusieurs seeds pour départager le vrai meilleur en espérance.
        results_sorted = sorted(results, key=lambda x: x['metrics']['reward'], reverse=True)
        top_candidates = results_sorted[:refine_top_k]
        print(f"\n  Affinage des {len(top_candidates)} meilleures combinaisons "
              f"sur {len(refine_seeds)} seeds {refine_seeds}...")

        refined = []
        for cand in top_candidates:
            full_params = {**DEFAULT_PARAMS[model_name], **cand['params']}
            refined_metrics = run_model(model_name, full_params, cfg['train'], cfg['test'], seeds=refine_seeds)
            refined.append({'params': cand['params'], 'metrics': refined_metrics})
            print(f"    {cand['params']} → reward={refined_metrics['reward']:.2f} "
                  f"(±{refined_metrics['reward_std']:.2f})")

        best = max(refined, key=lambda x: x['metrics']['reward'])
        print(f"\n  MEILLEUR (après affinage multi-seeds) : {best['params']}")
        print(f"  reward={best['metrics']['reward']:.2f} (±{best['metrics']['reward_std']:.2f})  "
              f"steps={best['metrics']['steps']:.2f}")

        grid_results['models'][model_name] = {
            'all_results':            results,
            'refined_top_candidates': refined,
            # best_params = params complets (grille + fixes) prêts à être injectés en phase final
            'best_params':  {**DEFAULT_PARAMS[model_name], **best['params']},
            'best_metrics': best['metrics'],
        }

    with open(GRID_FILE, 'w') as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nGrid search sauvegardé : {GRID_FILE}")


def phase_final(selected_models, model_config, seeds=(0,)):
    """Ré-entraîne chaque modèle avec les meilleurs params trouvés par le grid search.
    Produit le snapshot final qui sera comparé à la baseline dans phase_compare.
    """
    ensure_dirs()

    # Dépendance explicite : la phase final ne peut pas tourner sans le grid search
    if not os.path.exists(GRID_FILE):
        sys.exit(f"ERREUR : {GRID_FILE} introuvable. Lancez d'abord : python benchmark.py grid")

    with open(GRID_FILE) as f:
        grid_data = json.load(f)

    print("\n=== PHASE FINAL ===")
    print(f"Seeds : {seeds} (n={len(seeds)})\n")

    snapshot = {'timestamp': datetime.now().isoformat(), 'seeds': list(seeds), 'models': {}}

    for model_name in selected_models:
        if model_name not in grid_data['models']:
            print(f"  {model_name} absent du grid search, ignoré.")
            continue

        cfg = model_config[model_name]
        print(f"\n--- {MODEL_LABELS[model_name]} "
              f"({cfg['train']} épisodes train, {cfg['test']} test, {len(seeds)} seed(s)) ---")
        best_params = grid_data['models'][model_name]['best_params']
        print(f"  Params : {best_params}")

        metrics = run_model(model_name, best_params, cfg['train'], cfg['test'], seeds=seeds)
        snapshot['models'][model_name] = {'params': best_params, 'metrics': metrics}
        print(f"  reward={metrics['reward']:.2f} (±{metrics['reward_std']:.2f})  "
              f"steps={metrics['steps']:.2f}  penalties={metrics['penalties']:.2f}")

    with open(FINAL_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nFinal sauvegardé : {FINAL_FILE}")

    _plot_snapshot(snapshot, 'final')


def phase_compare():
    """Charge baseline.json et final.json pour générer le comparatif avant/après.
    Produit un PNG persistant, affiche le tableau terminal, puis ouvre la fenêtre interactive.
    """
    missing = [
        name for name, path in [('baseline', BASELINE_FILE), ('final', FINAL_FILE)]
        if not os.path.exists(path)
    ]
    if missing:
        sys.exit(f"ERREUR : fichiers manquants : {', '.join(missing)}.\n"
                 "Lancez d'abord les phases baseline et final.")

    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    with open(FINAL_FILE) as f:
        final = json.load(f)

    # Seuls les modèles présents dans les deux fichiers sont comparables
    models       = [m for m in DEFAULT_PARAMS if m in baseline['models'] and m in final['models']]
    metric_names = ['reward', 'steps', 'penalties', 'completion_rate']
    metric_labels = {
        'reward': 'Reward', 'steps': 'Steps',
        'penalties': 'Penalties', 'completion_rate': 'Completion rate (%)',
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    n_seeds_b = len(baseline.get('seeds', [])) or 1
    n_seeds_f = len(final.get('seeds', [])) or 1
    fig.suptitle(
        f"Comparaison avant / après fine-tuning  "
        f"(baseline n={n_seeds_b} seed(s), final n={n_seeds_f} seed(s))",
        fontsize=13, fontweight='bold')

    # Positions des groupes sur l'axe X : un groupe par modèle, deux barres par groupe
    x     = np.arange(len(models))
    width = 0.35  # largeur d'une barre ; les deux barres tiennent dans un espace de 1

    for ax, metric in zip(axes, metric_names):
        before = [baseline['models'].get(m, {}).get('metrics', {}).get(metric, 0) for m in models]
        after  = [final['models'].get(m, {}).get('metrics', {}).get(metric, 0) for m in models]
        before_err = [baseline['models'].get(m, {}).get('metrics', {}).get(f'{metric}_std', 0) for m in models]
        after_err  = [final['models'].get(m, {}).get('metrics', {}).get(f'{metric}_std', 0) for m in models]

        # Barres "Avant" en gris neutre, "Après" avec la couleur propre à chaque modèle
        # FIX : ajout de barres d'erreur (yerr) reflétant l'écart-type inter-seed,
        # quand disponible (fallback à 0 pour rester compatible avec les anciens
        # fichiers JSON sans std).
        bars_b = ax.bar(x - width / 2, before, width, yerr=before_err, capsize=4,
                        label='Avant', color='#90A4AE', alpha=0.8)
        bars_a = ax.bar(x + width / 2, after, width, yerr=after_err, capsize=4,
                        label='Après', color=[MODEL_COLORS[m] for m in models], alpha=0.9)

        ax.set_title(metric_labels[metric], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Annotation de la valeur numérique au-dessus de chaque barre
        for bars in (bars_b, bars_a):
            for bar in bars:
                ax.annotate(
                    f'{bar.get_height():.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8,
                )

    plt.tight_layout()
    ensure_dirs()

    # savefig avant show() : garantit que le PNG est écrit même si la fenêtre
    # interactive est fermée immédiatement
    path = os.path.join(PLOTS_DIR, 'comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé : {path}")

    _print_summary_table(baseline, final, models, metric_names)

    plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_snapshot(snapshot, phase_name):
    """Bar chart des 4 métriques pour une seule phase (baseline ou final).
    Appelé automatiquement à la fin de chaque phase pour avoir une trace visuelle immédiate.
    Affiche des barres d'erreur (écart-type inter-seed) quand disponibles.
    """
    models       = list(snapshot['models'].keys())
    metric_names = ['reward', 'steps', 'penalties', 'completion_rate']
    metric_labels = {
        'reward': 'Reward', 'steps': 'Steps',
        'penalties': 'Penalties', 'completion_rate': 'Completion rate (%)',
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    n_seeds = len(snapshot.get('seeds', [])) or 1
    fig.suptitle(f"Phase : {phase_name.capitalize()}  (n={n_seeds} seed(s))",
                 fontsize=13, fontweight='bold')

    x      = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]

    for ax, metric in zip(axes, metric_names):
        vals = [snapshot['models'][m]['metrics'].get(metric, 0) for m in models]
        errs = [snapshot['models'][m]['metrics'].get(f'{metric}_std', 0) for m in models]
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_title(metric_labels[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            ax.annotate(
                f'{bar.get_height():.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9,
            )

    plt.tight_layout()
    # Nom du fichier = nom de la phase → baseline.png ou final.png
    path = os.path.join(PLOTS_DIR, f'{phase_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Graphique sauvegardé : {path}")
    plt.show()


def _print_summary_table(baseline, final, models, metrics):
    """Tableau terminal : pour chaque modèle et chaque métrique, affiche
    Avant (±std) / Après (±std) / Δ. Le delta permet de voir en un coup
    d'œil si le fine-tuning a apporté un gain réel — et le ± permet de voir
    si ce gain dépasse le bruit inter-seed.
    """
    col    = 20
    line_w = 14 + len(metrics) * (col * 2 + 12)

    print("\n" + "=" * line_w)
    print(f"{'':14}" + "  ".join(
        f"{m.capitalize():^{col * 2 + 10}}" for m in metrics
    ))
    print(f"{'Modèle':<14}" + "  ".join(
        f"{'Avant (±std)':>{col}}  {'Après (±std)':>{col}}  {'Δ':>8}" for _ in metrics
    ))
    print("=" * line_w)

    for model in models:
        row = f"{MODEL_LABELS[model]:<14}"
        for metric in metrics:
            b      = baseline['models'].get(model, {}).get('metrics', {}).get(metric, 0)
            b_std  = baseline['models'].get(model, {}).get('metrics', {}).get(f'{metric}_std', 0)
            a      = final['models'].get(model, {}).get('metrics', {}).get(metric, 0)
            a_std  = final['models'].get(model, {}).get('metrics', {}).get(f'{metric}_std', 0)
            d = a - b
            sign = '+' if d > 0 else ''
            b_str = f"{b:.1f}±{b_std:.1f}"
            a_str = f"{a:.1f}±{a_std:.1f}"
            row += f"{b_str:>{col}}  {a_str:>{col}}  {sign}{d:>7.1f}  "
        print(row)

    print("=" * line_w)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark RL — Taxi Driver')
    parser.add_argument('phase', choices=['baseline', 'grid', 'final', 'compare'],
                        help='Phase à exécuter')
    parser.add_argument('--train-episodes', type=int, default=None,
                        help='Force un nombre d\'épisodes d\'entraînement identique pour tous les modèles. '
                             'Si absent, utilise les défauts par modèle de MODEL_EPISODE_DEFAULTS.')
    parser.add_argument('--test-episodes', type=int, default=None,
                        help='Force un nombre d\'épisodes de test identique pour tous les modèles. '
                             'Si absent, utilise les défauts par modèle de MODEL_EPISODE_DEFAULTS.')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Désactive les prompts — utilise tous les modèles avec les valeurs passées en argument')
    parser.add_argument('--seeds', type=str, default='0,1,2',
                        help="Seeds (séparés par des virgules) utilisés pour baseline/final. "
                             "Plusieurs seeds permettent de calculer un écart-type inter-seed "
                             "et de juger si un écart entre modèles est significatif plutôt "
                             "que du bruit d'initialisation. Ex: '0,1,2,3,4'.")
    parser.add_argument('--grid-seed', type=int, default=0,
                        help="Seed unique utilisé pour le balayage initial du grid search "
                             "(rapide mais bruité sur un seul run — voir --refine-top-k).")
    parser.add_argument('--refine-top-k', type=int, default=3,
                        help="Nombre de meilleures combinaisons (sur le balayage initial à "
                             "1 seed) ré-évaluées sur plusieurs seeds avant de désigner le "
                             "'meilleur' final du grid search.")
    parser.add_argument('--refine-seeds', type=str, default='0,1,2,3',
                        help="Seeds utilisés pour départager les top candidates du grid search.")
    args = parser.parse_args()

    # compare ne nécessite pas de sélection ni de configuration
    if args.phase == 'compare':
        phase_compare()
        return

    if args.no_interactive:
        selected = list(DEFAULT_PARAMS.keys())
        model_config = {}
        for m in selected:
            per_model = MODEL_EPISODE_DEFAULTS.get(m, {'train': 10000, 'test': 100})
            model_config[m] = {
                'train': args.train_episodes if args.train_episodes is not None else per_model['train'],
                'test':  args.test_episodes  if args.test_episodes  is not None else per_model['test'],
            }
    else:
        selected     = interactive_model_selection()
        model_config = interactive_episode_config(selected, 10000, 100)

    if args.phase == 'baseline':
        phase_baseline(selected, model_config, seeds=_parse_seeds(args.seeds))
    elif args.phase == 'grid':
        phase_grid(selected, model_config,
                  grid_seed=args.grid_seed,
                  refine_top_k=args.refine_top_k,
                  refine_seeds=_parse_seeds(args.refine_seeds))
    elif args.phase == 'final':
        phase_final(selected, model_config, seeds=_parse_seeds(args.seeds))


if __name__ == '__main__':
    main()