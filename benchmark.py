"""Benchmark des 4 modèles RL — capture de métriques avant/pendant/après fine-tuning.

Usage:
    python benchmark.py baseline [--train-episodes N] [--test-episodes N] [--no-interactive]
    python benchmark.py grid     [--train-episodes N] [--test-episodes N] [--no-interactive]
    python benchmark.py final    [--train-episodes N] [--test-episodes N] [--no-interactive]
    python benchmark.py compare
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

# Params par défaut de chaque modèle tels qu'ils sont définis dans leurs __init__.
# Sert de référence pour la phase baseline ET de base pour le grid search
# (les clés absentes de GRID_PARAMS sont remplies depuis ici).
DEFAULT_PARAMS = {
    'q_learning': {
        'alpha': 0.1, 'gamma': 0.6, 'epsilon_decay': 0.9995,
    },
    'sarsa': {
        'alpha': 0.2, 'gamma': 0.9, 'epsilon_decay': 0.999,
        'policy_type': 'epsilon_greedy', 'n_steps': 1, 'temperature': 1.0,
    },
    'monte_carlo': {
        'alpha': 0.05, 'gamma': 0.95, 'epsilon_decay': 0.9997,
        'visit_mode': 'first_visit',
    },
    'dqn': {
        'gamma': 0.99, 'epsilon_decay': 0.9995,
        'lr': 0.001, 'batch_size': 64, 'target_update': 10,
        'optimizer_type': 'adam',
    },
}

# Espace de recherche par modèle — seuls les hyperparamètres qui ont un vrai
# impact sur Taxi sont inclus. Le produit cartésien de chaque liste génère
# toutes les combinaisons à tester (voir phase_grid).
# Pour réduire le temps de run : raccourcir les listes ou baisser --train-episodes.
GRID_PARAMS = {
    # Q-Learning : 3×3×3 = 27 combinaisons
    'q_learning': {
        'alpha':         [0.05, 0.1, 0.2],
        'gamma':         [0.6, 0.8, 0.99],
        'epsilon_decay': [0.999, 0.9995, 0.9999],
    },
    # SARSA : 2×2×2×2×2 = 32 combinaisons
    # policy_type 'expected' = Expected SARSA (bootstraps sur valeur espérée)
    # n_steps > 1 = n-step SARSA (retours à plus long horizon)
    'sarsa': {
        'alpha':         [0.1, 0.2],
        'gamma':         [0.9, 0.99],
        'epsilon_decay': [0.999, 0.9995],
        'policy_type':   ['epsilon_greedy', 'expected'],
        'n_steps':       [1, 3],
    },
    # Monte Carlo : 2×3×3×2 = 36 combinaisons
    # visit_mode 'every_visit' met à jour la Q-table à chaque passage
    # par la même paire (s, a) dans l'épisode, pas seulement la première
    'monte_carlo': {
        'alpha':         [0.05, 0.1],
        'gamma':         [0.9, 0.95, 0.99],
        'epsilon_decay': [0.9995, 0.9997, 0.9999],
        'visit_mode':    ['first_visit', 'every_visit'],
    },
    # DQN : 2×2×2×2 = 16 combinaisons
    # buffer_size exclu : changer la capacité du replay buffer après __init__
    # n'a pas d'effet (le deque est déjà créé avec l'ancienne valeur)
    'dqn': {
        'lr':             [0.0005, 0.001],
        'gamma':          [0.95, 0.99],
        'batch_size':     [32, 64],
        'optimizer_type': ['adam', 'rmsprop'],
    },
}


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


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
    """Demande une seule fois le nombre d'épisodes — appliqué à tous les modèles sélectionnés."""
    print("\n--- Configuration des épisodes (tous les modèles) ---\n")
    train = _ask_int(f"  Épisodes d'entraînement [{default_train}] : ", default_train, 1, 1_000_000)
    test  = _ask_int(f"  Épisodes de test        [{default_test}] : ", default_test,  1, 10_000)
    print()
    return {m: {'train': train, 'test': test} for m in selected_models}


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
        agent.alpha         = params.get('alpha', 0.1)
        agent.gamma         = params.get('gamma', 0.6)
        agent.epsilon_decay = params.get('epsilon_decay', 0.9995)

    elif model_name == 'sarsa':
        agent = SRS.Sarsa('ansi')
        agent.alpha         = params.get('alpha', 0.2)
        agent.gamma         = params.get('gamma', 0.9)
        agent.epsilon_decay = params.get('epsilon_decay', 0.999)
        agent.policy_type   = params.get('policy_type', 'epsilon_greedy')
        agent.n_steps       = params.get('n_steps', 1)
        agent.temperature   = params.get('temperature', 1.0)

    elif model_name == 'monte_carlo':
        agent = MC.MonteCarlo('ansi')
        agent.alpha         = params.get('alpha', 0.05)
        agent.gamma         = params.get('gamma', 0.95)
        agent.epsilon_decay = params.get('epsilon_decay', 0.9997)
        agent.visit_mode    = params.get('visit_mode', 'first_visit')

    elif model_name == 'dqn':
        agent = DQN.DQNAgent('ansi')
        agent.gamma          = params.get('gamma', 0.99)
        agent.epsilon_decay  = params.get('epsilon_decay', 0.9995)
        agent.lr             = params.get('lr', 0.001)
        agent.batch_size     = params.get('batch_size', 64)
        agent.target_update  = params.get('target_update', 10)
        agent.optimizer_type = params.get('optimizer_type', 'adam')
        # L'optimiseur est créé dans __init__ avant qu'on change lr/optimizer_type,
        # il faut donc le reconstruire explicitement après injection des params.
        agent.optimizer      = agent._build_optimizer()

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    return agent


def run_model(model_name, params, train_episodes, test_episodes):
    """Unité de travail atomique : crée un agent vierge, l'entraîne, l'évalue.

    Chaque appel repart d'une Q-table/réseau vide — pas de réutilisation entre
    combinaisons du grid search. fast_testing=True désactive le rendu visuel
    pour ne pas bloquer le terminal pendant les séries de tests.
    """
    agent = _build_agent(model_name, params)
    agent.train(train_episodes=train_episodes, training_graph=False)
    steps, penalties, reward, completion_rate = agent.test(test_episodes=test_episodes, fast_testing=True)
    # Conversion explicite en float : numpy scalars ne sont pas sérialisables en JSON.
    return {
        'reward': float(reward),
        'steps': float(steps),
        'penalties': float(penalties),
        'completion_rate': float(completion_rate),
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_baseline(selected_models, model_config):
    """Capture le niveau de performance de chaque modèle avec ses params par défaut.
    C'est le point de référence (t=0) qui permettra de mesurer le gain du fine-tuning.
    """
    ensure_dirs()
    print("\n=== PHASE BASELINE ===\n")

    # timestamp inclus dans le JSON pour retracer quand la baseline a été générée
    snapshot = {'timestamp': datetime.now().isoformat(), 'models': {}}

    for model_name in selected_models:
        cfg = model_config[model_name]
        print(f"\n--- {MODEL_LABELS[model_name]} "
              f"({cfg['train']} épisodes train, {cfg['test']} test) ---")
        metrics = run_model(model_name, DEFAULT_PARAMS[model_name], cfg['train'], cfg['test'])
        snapshot['models'][model_name] = {
            'params':  DEFAULT_PARAMS[model_name],
            'metrics': metrics,
        }
        print(f"  reward={metrics['reward']:.2f}  steps={metrics['steps']:.2f}  penalties={metrics['penalties']:.2f}")

    with open(BASELINE_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nBaseline sauvegardée : {BASELINE_FILE}")

    _plot_snapshot(snapshot, 'baseline')


def phase_grid(selected_models, model_config):
    """Explore l'espace des hyperparamètres de chaque modèle par produit cartésien.
    Sauvegarde toutes les combinaisons testées et les meilleurs params dans grid_search.json.
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
              f"({cfg['train']} épisodes train, {cfg['test']} test) ===")
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
                metrics = run_model(model_name, params, cfg['train'], cfg['test'])
                # On ne sauvegarde que les clés de la grille (pas les params fixes)
                # pour garder all_results lisible
                results.append({'params': dict(zip(keys, combo)), 'metrics': metrics})
                print(f"  [{i+1}/{len(combos)}] {dict(zip(keys, combo))} "
                      f"→ reward={metrics['reward']:.2f}  steps={metrics['steps']:.2f}")
            except Exception as e:
                print(f"  [{i+1}/{len(combos)}] ERREUR {dict(zip(keys, combo))}: {e}")

        if not results:
            continue

        # Critère de sélection : reward maximum (meilleure performance au test)
        best = max(results, key=lambda x: x['metrics']['reward'])
        print(f"\n  MEILLEUR : {best['params']}")
        print(f"  reward={best['metrics']['reward']:.2f}  steps={best['metrics']['steps']:.2f}")

        grid_results['models'][model_name] = {
            'all_results':  results,
            # best_params = params complets (grille + fixes) prêts à être injectés en phase final
            'best_params':  {**DEFAULT_PARAMS[model_name], **best['params']},
            'best_metrics': best['metrics'],
        }

    with open(GRID_FILE, 'w') as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nGrid search sauvegardé : {GRID_FILE}")


def phase_final(selected_models, model_config):
    """Ré-entraîne chaque modèle avec les meilleurs params trouvés par le grid search.
    Produit le snapshot final qui sera comparé à la baseline dans phase_compare.
    """
    ensure_dirs()

    # Dépendance explicite : la phase final ne peut pas tourner sans le grid search
    if not os.path.exists(GRID_FILE):
        sys.exit(f"ERREUR : {GRID_FILE} introuvable. Lancez d'abord : python benchmark.py grid")

    with open(GRID_FILE) as f:
        grid_data = json.load(f)

    print("\n=== PHASE FINAL ===\n")

    snapshot = {'timestamp': datetime.now().isoformat(), 'models': {}}

    for model_name in selected_models:
        if model_name not in grid_data['models']:
            print(f"  {model_name} absent du grid search, ignoré.")
            continue

        cfg = model_config[model_name]
        print(f"\n--- {MODEL_LABELS[model_name]} "
              f"({cfg['train']} épisodes train, {cfg['test']} test) ---")
        best_params = grid_data['models'][model_name]['best_params']
        print(f"  Params : {best_params}")

        metrics = run_model(model_name, best_params, cfg['train'], cfg['test'])
        snapshot['models'][model_name] = {'params': best_params, 'metrics': metrics}
        print(f"  reward={metrics['reward']:.2f}  steps={metrics['steps']:.2f}  penalties={metrics['penalties']:.2f}")

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
    fig.suptitle("Comparaison avant / après fine-tuning", fontsize=14, fontweight='bold')

    # Positions des groupes sur l'axe X : un groupe par modèle, deux barres par groupe
    x     = np.arange(len(models))
    width = 0.35  # largeur d'une barre ; les deux barres tiennent dans un espace de 1

    for ax, metric in zip(axes, metric_names):
        before = [baseline['models'].get(m, {}).get('metrics', {}).get(metric, 0) for m in models]
        after  = [final['models'].get(m, {}).get('metrics', {}).get(metric, 0) for m in models]

        # Barres "Avant" en gris neutre, "Après" avec la couleur propre à chaque modèle
        bars_b = ax.bar(x - width / 2, before, width, label='Avant',  color='#90A4AE', alpha=0.8)
        bars_a = ax.bar(x + width / 2, after,  width, label='Après',
                        color=[MODEL_COLORS[m] for m in models], alpha=0.9)

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
    """Bar chart des 3 métriques pour une seule phase (baseline ou final).
    Appelé automatiquement à la fin de chaque phase pour avoir une trace visuelle immédiate.
    """
    models       = list(snapshot['models'].keys())
    metric_names = ['reward', 'steps', 'penalties', 'completion_rate']
    metric_labels = {
        'reward': 'Reward', 'steps': 'Steps',
        'penalties': 'Penalties', 'completion_rate': 'Completion rate (%)',
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Phase : {phase_name.capitalize()}", fontsize=13, fontweight='bold')

    x      = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]

    for ax, metric in zip(axes, metric_names):
        vals = [snapshot['models'][m]['metrics'].get(metric, 0) for m in models]
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
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
    """Tableau terminal : pour chaque modèle et chaque métrique, affiche Avant / Après / Δ.
    Le delta permet de voir en un coup d'œil si le fine-tuning a apporté un gain réel.
    """
    col    = 11
    line_w = 14 + len(metrics) * (col * 3 + 6)

    print("\n" + "=" * line_w)
    print(f"{'':14}" + "  ".join(
        f"{m.capitalize():^{col * 3 + 4}}" for m in metrics
    ))
    print(f"{'Modèle':<14}" + "  ".join(
        f"{'Avant':>{col}}  {'Après':>{col}}  {'Δ':>{col}}" for _ in metrics
    ))
    print("=" * line_w)

    for model in models:
        row = f"{MODEL_LABELS[model]:<14}"
        for metric in metrics:
            b = baseline['models'].get(model, {}).get('metrics', {}).get(metric, 0)
            a = final['models'].get(model, {}).get('metrics', {}).get(metric, 0)
            d = a - b
            sign = '+' if d > 0 else ''
            row += f"{b:>{col}.1f}  {a:>{col}.1f}  {sign}{d:>{col - 1}.1f}  "
        print(row)

    print("=" * line_w)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark RL — Taxi Driver')
    parser.add_argument('phase', choices=['baseline', 'grid', 'final', 'compare'],
                        help='Phase à exécuter')
    parser.add_argument('--train-episodes', type=int, default=10000,
                        help='Défaut épisodes d\'entraînement (pré-remplit le prompt interactif)')
    parser.add_argument('--test-episodes', type=int, default=100,
                        help='Défaut épisodes de test (pré-remplit le prompt interactif)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Désactive les prompts — utilise tous les modèles avec les valeurs passées en argument')
    args = parser.parse_args()

    # compare ne nécessite pas de sélection ni de configuration
    if args.phase == 'compare':
        phase_compare()
        return

    if args.no_interactive:
        selected     = list(DEFAULT_PARAMS.keys())
        model_config = {m: {'train': args.train_episodes, 'test': args.test_episodes}
                        for m in selected}
    else:
        selected     = interactive_model_selection()
        model_config = interactive_episode_config(selected, args.train_episodes, args.test_episodes)

    if args.phase == 'baseline':
        phase_baseline(selected, model_config)
    elif args.phase == 'grid':
        phase_grid(selected, model_config)
    elif args.phase == 'final':
        phase_final(selected, model_config)


if __name__ == '__main__':
    main()
