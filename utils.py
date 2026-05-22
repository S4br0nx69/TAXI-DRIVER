"""Fonctions utilitaires partagées entre les main.py."""


def safe_input_int(prompt, default, min_val=1, max_val=1000000):
    """Saisie sécurisée d'un entier avec bornes."""
    try:
        val = int(input(prompt) or default)
        if val < min_val or val > max_val:
            print(f"  Valeur hors bornes [{min_val}-{max_val}], défaut appliqué : {default}")
            return default
        return val
    except ValueError:
        print(f"  Entrée invalide, défaut appliqué : {default}")
        return default


def safe_input_float(prompt, default, min_val=0.0, max_val=1.0):
    """Saisie sécurisée d'un float avec bornes."""
    try:
        val = float(input(prompt) or default)
        if val < min_val or val > max_val:
            print(f"  Valeur hors bornes [{min_val}-{max_val}], défaut appliqué : {default}")
            return default
        return val
    except ValueError:
        print(f"  Entrée invalide, défaut appliqué : {default}")
        return default