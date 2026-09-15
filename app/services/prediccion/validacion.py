import numpy as np
import pandas as pd

from app.services.prediccion.modelos import (
    COLUMNA_PREDICCION_POR_METODO,
    elegir_campeon_por_producto,
    entrenar_xgboost,
    evaluar_modelo,
    prediccion_baseline_movil,
)
from app.services.prediccion.regresion_lineal import entrenar_regresion_lineal

# Cuántos meses de test tiene cada fold — el mismo tamaño que usa el
# split único de producción (sistema_prediccion.py), para que el fold
# más reciente sea directamente comparable con esos números.
TEST_SIZE_MESES = 3
# Mínimo de meses de entrenamiento para que un fold sea válido. Con muy
# poca historia, un modelo entrenado con menos que esto no dice nada.
MIN_TRAIN_MESES = 3
# Techo de folds — no tiene sentido pedir más folds de los que el
# historial disponible permite generar sin pisar MIN_TRAIN_MESES, y
# entrenar XGBoost + Regresión Lineal de más no aporta a partir de ahí.
MAX_FOLDS = 3


def _armar_folds(meses_ordenados):
    """
    Genera los splits de validación cruzada TEMPORAL (walk-forward /
    rolling origin) — nunca un k-fold al azar, que mezclaría meses
    futuros al entrenamiento de un fold que evalúa el pasado. Cada fold
    respeta el mismo principio que el split único de producción: solo
    entrena con meses anteriores a su propio test.

    El fold más reciente (el último de la lista devuelta) usa
    exactamente los últimos TEST_SIZE_MESES meses como test — el mismo
    holdout que ya reporta la tabla comparativa de producción. Los
    folds anteriores retroceden de a un mes en el tiempo, para
    confirmar que ese resultado no depende de que ese período haya sido
    particularmente fácil o difícil de pronosticar.

    Devuelve una lista de (meses_train, meses_test), ordenada del fold
    más viejo al más nuevo.
    """
    n = len(meses_ordenados)
    folds = []
    fin_test = n
    while len(folds) < MAX_FOLDS:
        inicio_test = fin_test - TEST_SIZE_MESES
        if inicio_test < MIN_TRAIN_MESES:
            break
        folds.append((meses_ordenados[:inicio_test], meses_ordenados[inicio_test:fin_test]))
        fin_test -= 1
    folds.reverse()
    return folds


# EXTENSIÓN PROPIA — no viene de ningún ejemplo de referencia. Ninguno
# valida con más de un split; acá se repite el pipeline completo (los
# tres modelos MÁS la selección de campeón por producto de
# modelos.py::elegir_campeon_por_producto, no solo un modelo aislado)
# contra varios períodos de tiempo distintos, para confirmar que las
# métricas de la tabla comparativa de producción no son un golpe de
# suerte de ese único período de prueba.
def validacion_cruzada_temporal(df_model, feature_cols):
    """
    Devuelve (detalle_folds, resumen):
      - detalle_folds: una fila por (fold, modelo) con sus métricas —
        lista de dicts lista para iterar en un template.
      - resumen: una fila por (modelo, métrica) con el promedio y
        desvío estándar a través de los folds — mismo formato.

    Si no hay suficiente historial para al menos un fold, devuelve dos
    listas vacías en vez de fallar (un CSV chico puede no alcanzar).
    """
    meses_ordenados = sorted(df_model["fecha"].unique())
    folds = _armar_folds(meses_ordenados)

    detalle_folds = []
    metricas_acumuladas = {"Media móvil": [], "Regresión Lineal": [], "XGBoost": [], "Sistema final": []}

    for numero_fold, (meses_train, meses_test) in enumerate(folds, start=1):
        train_fold = df_model[df_model["fecha"].isin(meses_train)]
        test_fold = df_model[df_model["fecha"].isin(meses_test)]
        if train_fold.empty or test_fold.empty:
            continue

        X_train = train_fold[feature_cols]
        X_test = test_fold[feature_cols]
        y_train = train_fold["demanda"]
        y_test = test_fold["demanda"]
        y_train_log = np.log1p(y_train)

        _, pred_lr, _ = entrenar_regresion_lineal(X_train, y_train, X_test, feature_cols)
        _, pred_xgb = entrenar_xgboost(X_train, y_train_log, X_test)
        pred_baseline = prediccion_baseline_movil(test_fold)

        metricas_fold = {
            "Media móvil": evaluar_modelo(y_test.values, pred_baseline, "Media móvil"),
            "Regresión Lineal": evaluar_modelo(y_test.values, pred_lr, "Regresión Lineal"),
            "XGBoost": evaluar_modelo(y_test.values, pred_xgb, "XGBoost"),
        }

        # "Sistema final" de este fold: mismo mecanismo de campeón por
        # producto que en producción, aplicado a los datos de este fold.
        tabla_fold = test_fold[["producto_id", "demanda"]].copy()
        tabla_fold["prediccion_media_movil"] = pred_baseline
        tabla_fold["prediccion_regresion_lineal"] = pred_lr
        tabla_fold["prediccion_xgboost"] = pred_xgb
        campeon_fold = elegir_campeon_por_producto(tabla_fold)
        tabla_fold["metodo_usado"] = tabla_fold["producto_id"].map(campeon_fold)
        tabla_fold["prediccion_final"] = tabla_fold.apply(
            lambda fila: fila[COLUMNA_PREDICCION_POR_METODO[fila["metodo_usado"]]], axis=1
        )
        metricas_fold["Sistema final"] = evaluar_modelo(
            tabla_fold["demanda"].values, tabla_fold["prediccion_final"].values, "Sistema final"
        )

        periodo_test = f"{meses_test[0].strftime('%Y-%m')} a {meses_test[-1].strftime('%Y-%m')}"
        for modelo, metricas in metricas_fold.items():
            metricas_acumuladas[modelo].append(metricas)
            detalle_folds.append({
                "fold": numero_fold,
                "meses_entrenamiento": len(meses_train),
                "periodo_test": periodo_test,
                "modelo": modelo,
                "mae": round(float(metricas["MAE"]), 2),
                "rmse": round(float(metricas["RMSE"]), 2),
                "mape": None if pd.isna(metricas["MAPE (%)"]) else round(float(metricas["MAPE (%)"]), 2),
                "wape": None if pd.isna(metricas["WAPE (%)"]) else round(float(metricas["WAPE (%)"]), 2),
            })

    resumen = _resumir_folds(metricas_acumuladas)
    return detalle_folds, resumen


def _resumir_folds(metricas_acumuladas):
    """Promedio y desvío estándar de cada métrica, por modelo, a través
    de todos los folds válidos."""
    resumen = []
    for modelo, corridas in metricas_acumuladas.items():
        if not corridas:
            continue
        for etiqueta, clave in [("MAE", "MAE"), ("RMSE", "RMSE"), ("MAPE (%)", "MAPE (%)"), ("WAPE (%)", "WAPE (%)")]:
            valores = [c[clave] for c in corridas if not pd.isna(c[clave])]
            if not valores:
                continue
            resumen.append({
                "modelo": modelo,
                "metrica": etiqueta,
                "promedio": round(float(np.mean(valores)), 2),
                "desvio": round(float(np.std(valores)), 2) if len(valores) > 1 else 0.0,
                "folds": len(valores),
            })
    return resumen
