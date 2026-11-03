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
# EXTENSIÓN PROPIA — el Ejemplo 1 solo compara con MAE/RMSE/MAPE; WAPE,
# Mediana Error y R² (bondad de ajuste por mínimos cuadrados) no vienen
# de ahí, se agregaron para el capítulo de juicio de expertos.
def graficar_comparacion_metricas(resultados, carpeta_resultados):
    graficas_metricas = {}
    metricas = ["MAE", "RMSE", "Mediana Error", "MAPE (%)", "WAPE (%)", "R2"]
    for metrica in metricas:
        ruta_grafica = os.path.join(
            carpeta_resultados,
            f"comparacion_{metrica.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'porcentaje')}.png"
        )
        titulo_metrica = "R²" if metrica == "R2" else metrica
        plt.figure(figsize=(8, 5))
        plt.bar(resultados["Modelo"], resultados[metrica])
        plt.title(f"Comparación de modelos - {titulo_metrica}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(ruta_grafica, bbox_inches="tight")
        plt.close()
        graficas_metricas[metrica] = ruta_grafica
    return graficas_metricas


# EXTENSIÓN PROPIA - gráfico combinado de WAPE y MAE por modelo, para
# comparar de un vistazo el error de cada método. Menor es mejor en
# ambas métricas. Incluye el Sistema final (selección por producto)
# calculado a partir de resultados_prediccion.
def graficar_comparacion_wape_mae(resultados, carpeta_resultados, resultados_prediccion=None):
    ruta = os.path.join(carpeta_resultados, 'comparacion_wape_mae.png')
    modelos = resultados['Modelo'].tolist()
    wape = resultados['WAPE (%)'].tolist()
    mae = resultados['MAE'].tolist()

    # Agregar el Sistema final (selección por producto) si hay datos de backtest.
    if resultados_prediccion is not None and not resultados_prediccion.empty:
        real = resultados_prediccion['demanda'].values
        sistema = resultados_prediccion['prediccion_final'].values
        mae_sistema = float((abs(real - sistema)).mean())
        wape_sistema = float(abs(real - sistema).sum() / abs(real).sum() * 100)
        modelos = modelos + ['Sistema final']
        wape = wape + [round(wape_sistema, 2)]
        mae = mae + [round(mae_sistema, 2)]

    import numpy as np
    x = np.arange(len(modelos))
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    barras_wape = ax.bar(x - ancho / 2, wape, ancho, label='WAPE (%)', color='#4C72B0')
    barras_mae = ax.bar(x + ancho / 2, mae, ancho, label='MAE', color='#DD8452')
    ax.set_ylabel('Error (menor es mejor)')
    ax.set_title('Comparacion de error por modelo - WAPE y MAE')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, rotation=15, ha='right')
    ax.legend()
    for b in barras_wape:
        ax.annotate(f'{b.get_height():.2f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for b in barras_mae:
        ax.annotate(f'{b.get_height():.2f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    return ruta


# EXTENSIÓN PROPIA — para el capítulo de juicio de expertos: qué tan de
# acuerdo están los tres métodos ENTRE SÍ (no contra la demanda real,
# que es lo que miden las métricas de arriba, sino unos contra otros).
# Cada punto es un producto-mes del período de backtest; cuánto más
# pegado a la línea roja (acuerdo perfecto), más cerca coinciden esos
# dos modelos en ese caso. Nubes de puntos muy separadas de la línea
# indican que los modelos discrepan bastante para ese producto — señal
# de que conviene mirar con más cuidado cuál se usó (columna "Método
# campeón" de la tabla de predicciones, elegido por MAE más bajo).
def graficar_wape_por_segmento(resultados_prediccion, carpeta_resultados):
    ruta = os.path.join(carpeta_resultados, 'wape_por_segmento.png')
    df = pd.DataFrame(resultados_prediccion)
    if df.empty or 'categoria_abc' not in df.columns or 'variabilidad' not in df.columns:
        return None

    df['segmento'] = df['categoria_abc'].astype(str) + ' / ' + df['variabilidad'].astype(str)

    modelos = {
        'Media Movil': 'prediccion_media_movil',
        'XGBoost': 'prediccion_xgboost',
        'Sistema final': 'prediccion_final',
    }

    segmentos = sorted(df['segmento'].unique())
    wape_por_modelo = {m: [] for m in modelos}
    for seg in segmentos:
        sub = df[df['segmento'] == seg]
        real = sub['demanda'].values
        for nombre, col in modelos.items():
            pred = sub[col].values
            wape = float(abs(real - pred).sum() / abs(real).sum() * 100) if abs(real).sum() > 0 else 0.0
            wape_por_modelo[nombre].append(round(wape, 2))

    import numpy as np
    x = np.arange(len(segmentos))
    ancho = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(segmentos) * 1.5), 6))
    colores = {'Media Movil': '#4C72B0', 'XGBoost': '#DD8452', 'Sistema final': '#55A868'}
    for i, (nombre, _) in enumerate(modelos.items()):
        offset = (i - 1) * ancho
        barras = ax.bar(x + offset, wape_por_modelo[nombre], ancho, label=nombre, color=colores[nombre])
        for b in barras:
            ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('WAPE (%) - menor es mejor')
    ax.set_title('WAPE por segmento de producto (clase ABC / variabilidad)')
    ax.set_xticks(x)
    ax.set_xticklabels(segmentos, rotation=30, ha='right', fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    return ruta


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


# EXTENSION PROPIA - grafico de lineas que compara la demanda real del
# periodo de test con las predicciones de los tres modelos (Media Movil,
# Regresion Lineal y XGBoost), agregadas por mes. Permite ver de un
# vistazo que tan cerca queda cada modelo de la demanda real.
def graficar_predicciones_vs_real(resultados_prediccion, carpeta_resultados):
    ruta = os.path.join(carpeta_resultados, "predicciones_vs_real.png")
    df = resultados_prediccion.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    agregado = df.groupby("fecha").agg(
        demanda_real=("demanda", "sum"),
        media_movil=("prediccion_media_movil", "sum"),
        regresion_lineal=("prediccion_regresion_lineal", "sum"),
        xgboost=("prediccion_xgboost", "sum"),
    ).sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(agregado.index, agregado["demanda_real"], label="Demanda real", color="black", linewidth=2, marker="o")
    plt.plot(agregado.index, agregado["media_movil"], label="Media Movil", linestyle="--", marker="s")
    plt.plot(agregado.index, agregado["regresion_lineal"], label="Regresion Lineal", linestyle="--", marker="^")
    plt.plot(agregado.index, agregado["xgboost"], label="XGBoost", linestyle="--", marker="d")
    plt.title("Predicciones vs. demanda real (periodo de prueba, agregado por mes)")
    plt.xlabel("Mes")
    plt.ylabel("Demanda (unidades)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()
    return ruta
