"""Carga el modelo guardado y predice reservas sin reentrenar."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config


def cargar(ruta=None):
    """Devuelve el Pipeline guardado y sus metadatos."""
    ruta_modelo = Path(config.MODELO_PKL if ruta is None else ruta)
    ruta_metadatos = ruta_modelo.with_name(config.METADATOS.name)

    for archivo in (ruta_modelo, ruta_metadatos):
        if not archivo.exists():
            raise FileNotFoundError(
                f"Falta {archivo}. Ejecuta primero python main.py."
            )

    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))

    ruta_keras = ruta_modelo.with_name(config.MODELO_KERAS.name)
    if metadatos["ganador"] == "red_keras" and not ruta_keras.exists():
        raise FileNotFoundError(
            f"Falta {ruta_keras}. Vuelve a generar el modelo."
        )

    pipeline = joblib.load(ruta_modelo)

    if metadatos["ganador"] == "red_keras":
        pipeline.steps[-1][1].cargar_aparte(ruta_keras)

    # Mantiene las firmas previstas y conserva el umbral del artefacto.
    pipeline.metadatos_ = metadatos

    return pipeline, metadatos


def predecir_proba(pipeline, reservas):
    """Devuelve la probabilidad de cancelación de cada reserva."""
    if not hasattr(pipeline, "metadatos_"):
        raise ValueError("Carga primero el modelo con cargar().")

    columnas = pipeline.metadatos_["columnas"]
    faltantes = [columna for columna in columnas if columna not in reservas]

    if faltantes:
        raise ValueError(f"Faltan columnas de entrada: {faltantes}")

    entradas = reservas.loc[:, columnas]
    probabilidades = pipeline.predict_proba(entradas)

    clases = np.asarray(pipeline.classes_)
    posiciones = np.flatnonzero(clases == 1)

    if len(posiciones) != 1:
        raise ValueError("El modelo no identifica correctamente la clase 1.")

    return np.asarray(
        probabilidades[:, posiciones[0]], dtype=float
    )


def predecir(pipeline, reservas):
    """Devuelve 0/1 usando el umbral guardado al entrenar."""
    probabilidades = predecir_proba(pipeline, reservas)
    umbral = float(pipeline.metadatos_["umbral"])
    return (probabilidades >= umbral).astype(int)


if __name__ == "__main__":
    # Ejemplo de uso sobre cinco registros; no mide rendimiento.
    from . import data_loader

    pipeline, metadatos = cargar()

    crudo = data_loader.cargar_crudo().head(5)
    reservas = crudo.loc[:, metadatos["columnas"]]

    resultado = pd.DataFrame({
        "probabilidad_cancelacion": predecir_proba(pipeline, reservas),
        "prediccion": predecir(pipeline, reservas),
    })

    print("Modelo:", metadatos["ganador"])
    print("Umbral:", metadatos["umbral"])
    print("Artefacto demo:", metadatos["demo"])
    print(resultado.to_string(index=False))