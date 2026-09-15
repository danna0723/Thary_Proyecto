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
