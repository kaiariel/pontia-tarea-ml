"""Métricas y visualización de resultados.

Es el ÚNICO módulo que abre el conjunto de test, y lo abre una sola vez, al final.
Cada vez que el test influye en una decisión —qué modelo, qué hiperparámetro, qué
umbral— deja de ser una estimación honesta.
"""

from . import config
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve


def metricas(y_true, y_pred, y_proba=None) -> dict:
    """Calcula las métricas de clasificación para la clase 1: cancela."""
    resultado = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None:
        resultado["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return resultado


def matriz_confusion(y_true, y_pred, ruta=None):
    """Guarda la matriz con recuentos y porcentajes por fila."""
    destino = Path(config.FIG_CONFUSION if ruta is None else ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)

    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    totales_fila = matriz.sum(axis=1, keepdims=True)
    porcentajes = np.divide(
        matriz,
        totales_fila,
        out=np.zeros_like(matriz, dtype=float),
        where=totales_fila != 0,
    ) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagen, ax=ax, label="Número de reservas")

    etiquetas = ["No cancela", "Cancela"]
    ax.set_xticks([0, 1], labels=etiquetas)
    ax.set_yticks([0, 1], labels=etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión")

    for fila in range(2):
        for columna in range(2):
            color = "white" if matriz[fila, columna] > matriz.max() / 2 else "black"
            ax.text(
                columna,
                fila,
                f"{matriz[fila, columna]:,}\n{porcentajes[fila, columna]:.1f} %",
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def curva_roc(modelos_proba: dict, y_true, ruta=None):
    """Figura obligatoria del enunciado. Guarda en config.FIG_ROC.

    LAS CINCO CURVAS EN LOS MISMOS EJES, con su AUC en la leyenda y la diagonal
    del azar. Cinco gráficos sueltos no demuestran nada.
    """
    raise NotImplementedError("TODO")


def importancias(pipeline, X, y, ruta=None):
    """Guarda las variables originales más importantes del ganador."""
    destino = Path(config.FIG_IMPORTANCIAS if ruta is None else ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)

    resultado = permutation_importance(
        pipeline,
        X,
        y,
        scoring=config.METRICA_PRINCIPAL,
        n_repeats=config.IMPORTANCIA_REPETICIONES,
        random_state=config.SEMILLA,
        n_jobs=1,
        max_samples=min(config.IMPORTANCIA_MUESTRA, len(X)),
    )

    cantidad = min(config.IMPORTANCIA_TOP, len(X.columns))
    indices = np.argsort(resultado.importances_mean)[-cantidad:]
    nombres = [str(X.columns[i]) for i in indices]
    valores = resultado.importances_mean[indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(nombres, valores)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel(f"Caída de {config.METRICA_PRINCIPAL} al permutar")
    ax.set_title("Importancia de variables originales del modelo ganador")
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)
    return destino


def informe(tabla_comparativa, metricas_test: dict, ruta=None):
    """Guarda la tabla comparativa y las métricas de test."""
    carpeta = Path(config.OUTPUTS if ruta is None else ruta)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_csv = carpeta / config.TABLA_COMPARATIVA.name
    ruta_json = carpeta / config.METRICAS_TEST.name

    tabla_comparativa.to_csv(ruta_csv, index=False)

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(metricas_test, archivo, ensure_ascii=False, indent=2)

    return ruta_csv, ruta_json