"""Sauvegarde des tableaux (resultats/) et des figures (figures/) produits par les scripts."""

import matplotlib

matplotlib.use("Agg")  # les scripts enregistrent les figures sans les afficher

import matplotlib.pyplot as plt

from src.config import DOSSIER_FIGURES, DOSSIER_RESULTATS


def sauvegarder_tableau(tableau, nom, afficher=True):
    """Enregistre un tableau en CSV dans resultats/ et l'affiche dans la console."""
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    tableau.to_csv(DOSSIER_RESULTATS / f"{nom}.csv", encoding="utf-8")
    if afficher:
        print(f"\n{nom}\n{tableau.round(3).to_string()}")


def sauvegarder_figure(nom, figure=None):
    """Enregistre la figure courante (ou celle donnée) en PNG dans figures/, puis la ferme."""
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    figure = figure or plt.gcf()
    figure.savefig(DOSSIER_FIGURES / f"{nom}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
