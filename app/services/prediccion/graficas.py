import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# EXTENSIÓN PROPIA — gráfico de la clasificación ABC/variabilidad
# (segmentacion.py), que tampoco viene de ningún ejemplo de referencia.
def graficar_segmentacion(producto_comportamiento, carpeta_resultados):
    ruta = os.path.join(carpeta_resultados, "segmentacion_abc_variabilidad.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    producto_comportamiento["categoria_abc"].value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set_title("Productos por categoría ABC")
    producto_comportamiento["variabilidad"].value_counts().plot.bar(ax=axes[1])
    axes[1].set_title("Productos por variabilidad")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


# Ejemplo 1 (SKU Demand Forecasting): mismo gráfico de importancia de
# variables de XGBoost que el notebook de referencia
# (model_sample.feature_importances_ -> barh horizontal).
def graficar_importancia_features(model, feature_cols, carpeta_resultados):
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    ruta = os.path.join(carpeta_resultados, "xgb_feature_importance.png")
    plt.figure(figsize=(10, 6))
    importance.head(15).sort_values().plot.barh()
    plt.title("XGBoost - Importancia de Características")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta


# Ejemplo 1: gráfico de barras comparando modelos (equivalente al
# "Benchmark bar chart" de la sección "Model Benchmarking" del notebook).
def graficar_comparacion_metricas(resultados, carpeta_resultados):
    graficas_metricas = {}
    metricas = ["MAE", "RMSE", "MAPE (%)", "WAPE (%)"]
    for metrica in metricas:
        ruta_grafica = os.path.join(
            carpeta_resultados,
            f"comparacion_{metrica.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'porcentaje')}.png"
        )
        plt.figure(figsize=(8, 5))
        plt.bar(resultados["Modelo"], resultados[metrica])
        plt.title(f"Comparación de modelos - {metrica}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(ruta_grafica, bbox_inches="tight")
        plt.close()
        graficas_metricas[metrica] = ruta_grafica
    return graficas_metricas


# EXTENSIÓN PROPIA — para el capítulo de juicio de expertos: qué tan de
# acuerdo están los tres métodos ENTRE SÍ (no contra la demanda real,
# que es lo que miden las métricas de arriba, sino unos contra otros).
# Cada punto es un producto-mes del período de backtest; cuánto más
# pegado a la línea roja (acuerdo perfecto), más cerca coinciden esos
# dos modelos en ese caso. Nubes de puntos muy separadas de la línea
# indican que los modelos discrepan bastante para ese producto — señal
# de que conviene mirar con más cuidado cuál se usó (columna "Método
# campeón" de la tabla de predicciones, elegido por MAE más bajo).
def graficar_dispersion_modelos(resultados_prediccion, carpeta_resultados):
    ruta = os.path.join(carpeta_resultados, "dispersion_modelos.png")
    pares = [
        ("prediccion_xgboost", "prediccion_regresion_lineal", "XGBoost", "Regresión Lineal"),
        ("prediccion_xgboost", "prediccion_media_movil", "XGBoost", "Media Móvil"),
        ("prediccion_regresion_lineal", "prediccion_media_movil", "Regresión Lineal", "Media Móvil"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col_x, col_y, nombre_x, nombre_y) in zip(axes, pares):
        x = resultados_prediccion[col_x]
        y = resultados_prediccion[col_y]
        ax.scatter(x, y, alpha=0.4, s=18)
        limite = max(float(x.max()), float(y.max()), 1.0)
        ax.plot([0, limite], [0, limite], "r--", linewidth=1, label="Acuerdo perfecto (y = x)")
        ax.set_xlabel(f"Predicción {nombre_x}")
        ax.set_ylabel(f"Predicción {nombre_y}")
        ax.set_title(f"{nombre_x} vs. {nombre_y}")
        ax.legend(fontsize=8)
    plt.suptitle("Dispersión entre modelos — más cerca de la línea roja = más de acuerdo")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta
