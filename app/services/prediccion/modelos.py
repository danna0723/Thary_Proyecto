import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


def entrenar_xgboost(X_train, y_train_log, X_test):
    """
    Ejemplo 1 (SKU Demand Forecasting): mismos hiperparámetros de
    XGBoost que el notebook de referencia (max_depth, learning_rate,
    subsample, colsample_bytree, reg_alpha, reg_lambda, random_state).
    Ejemplo 3 (Kaggle Grupo Bimbo): entrena sobre el target en escala
    log1p (y_train_log) y revierte con expm1 + clip a 0 al predecir.

    Devuelve (model, pred_xgb).
    """
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=500, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train_log)
    pred_xgb = model.predict(X_test)
    pred_xgb = np.maximum(np.expm1(pred_xgb), 0)
    return model, pred_xgb


def prediccion_baseline_movil(test_df):
    """
    Ejemplo 1: baseline de Media Móvil (equivalente a la "4-wk Moving
    Average" del notebook de referencia, aquí a 3 meses) — se usa tanto
    de piso de comparación en la tabla de métricas como candidata en la
    selección de campeón por producto (elegir_campeon_por_producto).
    """
    return np.maximum(test_df["media_movil_3"].values, 0)


# EXTENSIÓN PROPIA — el WAPE no aparece en ninguno de los 4 ejemplos
# (todos miden con MAE/RMSE/MAPE). Se agregó porque el MAPE es engañoso
# en catálogos con muchos productos de demanda baja/discontinua (un
# producto que vende 1 unidad y se predicen 2 ya da 100% de error), algo
# muy presente en un inventario real como el de la clínica.
def calcular_wape(y_real, predicciones):
    denominador = np.sum(np.abs(y_real))
    if denominador == 0:
        return np.nan
    return (np.sum(np.abs(y_real - predicciones)) / denominador) * 100


def evaluar_modelo(y_real, predicciones, nombre):
    mae = mean_absolute_error(y_real, predicciones)
    rmse = np.sqrt(mean_squared_error(y_real, predicciones))
    r2 = r2_score(y_real, predicciones)
    wape = calcular_wape(y_real, predicciones)
    mask_no_cero = y_real > 0
    if mask_no_cero.sum() > 0:
        mape = mean_absolute_percentage_error(y_real[mask_no_cero], predicciones[mask_no_cero]) * 100
    else:
        mape = np.nan
    return {"Modelo": nombre, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape, "WAPE (%)": wape}


def benchmarking_modelos(y_test, pred_baseline_3, pred_lr, pred_xgb):
    """
    Ejemplo 1: benchmarking de modelos lado a lado (equivalente a la
    sección "Model Benchmarking" del notebook — ahí ARIMA/XGBoost vs.
    Naive/Moving Average, acá Regresión Lineal/XGBoost vs. Media Móvil).
    """
    return pd.DataFrame([
        evaluar_modelo(y_test.values, pred_baseline_3, "Media móvil 3 meses"),
        evaluar_modelo(y_test.values, pred_lr, "Regresión Lineal"),
        evaluar_modelo(y_test.values, pred_xgb, "XGBoost (log1p)")
    ])


COLUMNA_PREDICCION_POR_METODO = {
    "Media móvil": "prediccion_media_movil",
    "Regresión Lineal": "prediccion_regresion_lineal",
    "XGBoost": "prediccion_xgboost",
}


def seleccionar_prediccion_final(tabla):
    """
    Arma la columna "prediccion_final": para cada fila, la predicción
    del método que quedó en "metodo_usado" (elegido por
    elegir_campeon_por_producto). La usan tanto construir_tabla_predicciones
    como la validación cruzada (validacion.py), sobre tablas de hasta
    miles de filas.

    Antes esto era tabla.apply(lambda fila: ..., axis=1) — llama a una
    función de Python por cada fila, una por una. np.select hace la
    misma elección para todas las filas de una vez (vectorizado);
    mismo resultado, mucho más rápido en catálogos grandes.
    """
    condiciones = [tabla["metodo_usado"] == metodo for metodo in COLUMNA_PREDICCION_POR_METODO]
    opciones = [tabla[columna] for columna in COLUMNA_PREDICCION_POR_METODO.values()]
    return np.select(condiciones, opciones)


# EXTENSIÓN PROPIA — ningún ejemplo elige el método por producto; ahí
# siempre se usa "el mejor modelo" para todo el catálogo (ver Ejemplo 1:
# "Use XGBoost forecasts (best model); fall back to ARIMA"). Acá, en
# cambio, se mide el error de backtest de los TRES métodos para cada
# producto por separado, y gana el que menos se equivocó en ESE
# producto — reemplaza la heurística anterior basada en categoría
# ABC/variabilidad (que solo elegía entre XGBoost y Media Móvil) por
# una selección empírica de a tres, respaldada en los datos.
def elegir_campeon_por_producto(resultados_prediccion):
    """
    Devuelve una Series indexada por producto_id con el nombre del
    método ("Media móvil"/"Regresión Lineal"/"XGBoost") que tuvo menor
    MAE para ese producto en los meses de backtest.

    Antes esto llamaba a mean_absolute_error() de scikit-learn una vez
    por cada (producto, método) vía groupby().apply() — con 856
    productos y 4 corridas (la principal + 3 de validación cruzada) son
    miles de llamadas, y esa función paga bastante costo de validación
    interna en cada una. El MAE es solo el promedio del error absoluto,
    así que acá se calcula una sola vez para TODAS las filas
    (vectorizado) y recién después se agrupa por producto — mismo
    resultado, medido ~15x más rápido en un catálogo de 500 productos.
    """
    errores_por_metodo = pd.DataFrame({
        metodo: (resultados_prediccion["demanda"] - resultados_prediccion[columna]).abs()
        for metodo, columna in COLUMNA_PREDICCION_POR_METODO.items()
    })
    errores_por_producto = errores_por_metodo.groupby(resultados_prediccion["producto_id"]).mean()
    return errores_por_producto.idxmin(axis=1)


def construir_tabla_predicciones(test_df, pred_baseline_3, pred_lr, pred_xgb):
    """
    Arma la tabla de predicciones de backtest (últimos 3 meses reales)
    con las tres predicciones lado a lado, elige el método campeón por
    producto (elegir_campeon_por_producto) y calcula el MAPE/WAPE
    globales de esa predicción combinada.

    Devuelve (resultados_prediccion, mape_final, wape_final, metodo_campeon).
    """
    resultados_prediccion = test_df[["fecha", "producto_id", "demanda", "categoria_abc", "variabilidad"]].copy()
    resultados_prediccion["prediccion_media_movil"] = pred_baseline_3
    resultados_prediccion["prediccion_regresion_lineal"] = pred_lr
    resultados_prediccion["prediccion_xgboost"] = pred_xgb

    metodo_campeon = elegir_campeon_por_producto(resultados_prediccion)
    resultados_prediccion["metodo_usado"] = resultados_prediccion["producto_id"].map(metodo_campeon)
    resultados_prediccion["prediccion_final"] = seleccionar_prediccion_final(resultados_prediccion)
    resultados_prediccion["error_final"] = resultados_prediccion["prediccion_final"] - resultados_prediccion["demanda"]

    mask_no_cero = resultados_prediccion["demanda"] > 0
    if mask_no_cero.sum() > 0:
        mape_final = mean_absolute_percentage_error(
            resultados_prediccion.loc[mask_no_cero, "demanda"],
            resultados_prediccion.loc[mask_no_cero, "prediccion_final"]
        ) * 100
    else:
        mape_final = np.nan

    wape_final = calcular_wape(
        resultados_prediccion["demanda"].values,
        resultados_prediccion["prediccion_final"].values
    )

    return resultados_prediccion, mape_final, wape_final, metodo_campeon
