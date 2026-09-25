import numpy as np
import pandas as pd

from app.services.prediccion.modelos import evaluar_modelo


# EXTENSIÓN PROPIA — para el capítulo de juicio de expertos: ¿aporta valor
# combinar dos métodos en vez de usar uno solo? Ninguno de los 4 ejemplos
# de referencia combina modelos; la idea viene de la literatura de
# combinación de pronósticos:
#   - Bates, J. M., & Granger, C. W. J. (1969). "The combination of
#     forecasts." Operational Research Quarterly, 20(4), 451-468 — los
#     pesos de la combinación se derivan de los errores pasados de cada
#     pronóstico, y la combinación puede tener menor error cuadrático
#     medio que cualquiera de los dos por separado.
#   - Granger, C. W. J., & Ramanathan, R. (1984). "Improved methods of
#     combining forecasts." Journal of Forecasting, 3(2), 197-204 —
#     plantean el cálculo de los pesos como un problema de regresión
#     (mínimos cuadrados) sobre la demanda real. Acá se usa la versión
#     con la restricción clásica de Bates y Granger (los pesos suman 1,
#     sin constante), que ellos después relajaron: así la combinación es
#     un promedio ponderado (w * XGBoost + (1 - w) * Media móvil), fácil
#     de interpretar y que nunca sale del rango de las dos predicciones.
def _peso_minimos_cuadrados(y, pred_a, pred_b):
    """
    Peso w en [0, 1] que le toca a pred_a (1 - w le toca a pred_b) para
    minimizar la suma de errores al cuadrado de la combinación contra y:

        minimizar  sum( (y - (w * pred_a + (1 - w) * pred_b)) ** 2 )

    Reordenando, es una regresión por mínimos cuadrados sin constante de
    (y - pred_b) sobre (pred_a - pred_b), con solución cerrada:

        w = sum((y - pred_b) * (pred_a - pred_b)) / sum((pred_a - pred_b) ** 2)

    Se recorta a [0, 1] para que sea un promedio ponderado de verdad. Si
    las dos predicciones son idénticas, no hay nada que decidir: 0.5.
    """
    diferencia = pred_a - pred_b
    denominador = float(np.sum(diferencia ** 2))
    if denominador == 0:
        return 0.5
    w = float(np.sum((y - pred_b) * diferencia) / denominador)
    return min(max(w, 0.0), 1.0)


def evaluar_combinacion(predicciones):
    """
    Compara XGBoost solo, Media móvil sola y la combinación de ambos
    (promedio ponderado con pesos por mínimos cuadrados) sobre las mismas
    filas del backtest — la tabla de predicciones que ya devuelve
    construir_tabla_predicciones (una fila por producto-mes, con la demanda
    real y la predicción de cada método).

    Para que la comparación sea justa, el peso de cada mes se estima SIN
    ese mes (con los demás meses del backtest, todos los productos juntos):
    si se estimara con los mismos datos contra los que después se mide el
    error, la combinación saldría artificialmente buena porque el peso
    "vio" las respuestas de antemano. Como los productos se juntan en un
    solo cálculo, hay miles de filas por peso aunque cada producto tenga
    pocos meses.

    Devuelve (filas, detalle):
      - filas: lista de dicts con las mismas métricas de evaluar_modelo
        (MAE, RMSE, Mediana Error, R2, MAPE, WAPE), una por cada opción.
      - detalle: {"peso_xgboost": peso estimado con todos los meses (el
        que se usaría de acá en adelante), "peso_media_movil": 1 - eso,
        "pesos_por_mes": el peso usado al evaluar cada mes,
        "mejora_mae_pct"/"mejora_rmse_pct": cuánto mejora (positivo) o
        empeora (negativo) la combinación respecto al mejor de los dos
        métodos solos, en porcentaje}.
    Devuelve (None, None) si no hay predicciones para evaluar.
    """
    df = pd.DataFrame(predicciones)
    if df.empty:
        return None, None

    y = df["demanda"].to_numpy(dtype=float)
    pred_xgb = df["prediccion_xgboost"].to_numpy(dtype=float)
    pred_mm = df["prediccion_media_movil"].to_numpy(dtype=float)
    meses = df["fecha"].astype(str).to_numpy()

    combinada = np.empty_like(y)
    pesos_por_mes = {}
    for mes in np.unique(meses):
        es_mes = meses == mes
        resto = ~es_mes
        if resto.any():
            w = _peso_minimos_cuadrados(y[resto], pred_xgb[resto], pred_mm[resto])
        else:
            w = 0.5
        pesos_por_mes[str(mes)] = round(w, 4)
        combinada[es_mes] = w * pred_xgb[es_mes] + (1 - w) * pred_mm[es_mes]

    filas = [
        evaluar_modelo(y, pred_mm, "Media móvil sola"),
        evaluar_modelo(y, pred_xgb, "XGBoost solo"),
        evaluar_modelo(y, combinada, "Combinación XGBoost + Media móvil"),
    ]

    peso_final = _peso_minimos_cuadrados(y, pred_xgb, pred_mm)
    mejor_mae_solo = min(filas[0]["MAE"], filas[1]["MAE"])
    mejor_rmse_solo = min(filas[0]["RMSE"], filas[1]["RMSE"])
    detalle = {
        "peso_xgboost": round(peso_final, 4),
        "peso_media_movil": round(1 - peso_final, 4),
        "pesos_por_mes": pesos_por_mes,
        "mejora_mae_pct": _mejora_pct(mejor_mae_solo, filas[2]["MAE"]),
        "mejora_rmse_pct": _mejora_pct(mejor_rmse_solo, filas[2]["RMSE"]),
    }
    return filas, detalle


def _mejora_pct(referencia, valor):
    """Cuánto menor (positivo) o mayor (negativo) es `valor` que
    `referencia`, en porcentaje de la referencia."""
    if referencia == 0:
        return None
    return round((referencia - valor) / referencia * 100, 2)
