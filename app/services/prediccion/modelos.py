import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

# Umbral de significancia estándar (Diebold & Mariano, 1995) para la
# prueba usada en elegir_campeon_por_producto.
NIVEL_SIGNIFICANCIA_DM = 0.05

# EXTENSIÓN PROPIA — tamaño de la ventana de backtest (meses de "test"),
# usada tanto por el split único de producción (sistema_prediccion.py)
# como por la validación cruzada temporal (validacion.py). Con un test
# fijo de 3 meses, la prueba de Diebold-Mariano de arriba siempre queda
# con solo 2 grados de libertad, sin importar cuánta historia tenga el
# archivo — con archivos largos conviene un test más grande, para darle
# más poder estadístico a esa prueba (más meses = comparación más
# confiable), sin sacrificar tanto entrenamiento como sacrificaría un
# archivo corto con el mismo test grande.
#
# El corte de 18 meses de archivo (no 18 "usables" — ver
# tamano_test_backtest) es una heurística práctica definida para este
# proyecto, no un valor tomado de una fuente bibliográfica externa (a
# diferencia de NIVEL_SIGNIFICANCIA_DM o los umbrales de CV² de
# Syntetos & Boylan, 2005).
MESES_ARCHIVO_UMBRAL_TEST_GRANDE = 18
TEST_SIZE_MESES_DEFAULT = 3
TEST_SIZE_MESES_GRANDE = 6

# Meses que siempre se pierden por no tener suficiente historial previo
# para calcular lag_1/lag_2/lag_3 (ver features.py) — necesario para
# traducir "meses usables" (después de ese descarte) a "meses del
# archivo original" (lo que el usuario subió).
MESES_PERDIDOS_POR_REZAGOS = 3


def tamano_test_backtest(n_meses_usables):
    """
    Cuántos meses usar como período de prueba (backtest), dado cuántos
    meses quedan disponibles después de perder los primeros
    MESES_PERDIDOS_POR_REZAGOS por rezagos (n_meses_usables =
    len(meses_ordenados), tanto en sistema_prediccion.py como en
    validacion.py).

    Con archivos largos (18 meses del archivo original o más) usa un
    test de TEST_SIZE_MESES_GRANDE en vez de TEST_SIZE_MESES_DEFAULT,
    para que la prueba de Diebold-Mariano en elegir_campeon_por_producto
    tenga más grados de libertad y más poder estadístico. Con archivos
    cortos se mantiene el test chico, para no dejar muy poca historia
    para entrenar.
    """
    n_meses_archivo_original = n_meses_usables + MESES_PERDIDOS_POR_REZAGOS
    if n_meses_archivo_original >= MESES_ARCHIVO_UMBRAL_TEST_GRANDE:
        return TEST_SIZE_MESES_GRANDE
    return TEST_SIZE_MESES_DEFAULT


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
    # EXTENSIÓN PROPIA — la mediana del error absoluto, a diferencia del
    # MAE (que es el PROMEDIO del error absoluto), no se deja arrastrar
    # por unos pocos productos con error muy grande — sirve para ver si
    # el MAE de un modelo está siendo inflado por pocos casos atípicos o
    # si de verdad representa el error "típico" en el catálogo.
    mediana_error = np.median(np.abs(np.asarray(y_real) - np.asarray(predicciones)))
    mask_no_cero = y_real > 0
    if mask_no_cero.sum() > 0:
        mape = mean_absolute_percentage_error(y_real[mask_no_cero], predicciones[mask_no_cero]) * 100
    else:
        mape = np.nan
    return {
        "Modelo": nombre, "MAE": mae, "RMSE": rmse, "R2": r2,
        "Mediana Error": mediana_error, "MAPE (%)": mape, "WAPE (%)": wape,
    }


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
#
# El backtest tiene muy pocos meses (3 en la corrida principal, hasta 5
# si el archivo trae más historial) — con tan pocos datos, que un
# método tenga el MAE más bajo no significa que sea realmente mejor:
# puede ser azar. Por eso, antes de declarar ganador al método de menor
# MAE, se compara contra el segundo mejor con la prueba de
# Diebold-Mariano (Diebold, F. X., & Mariano, R. S. (1995). "Comparing
# predictive accuracy." Journal of Business & Economic Statistics,
# 13(3), 253-263), con la corrección para muestras chicas de Harvey,
# Leybourne y Newbold (1997) — sin esta corrección, la prueba clásica
# asume muestras grandes y no es válida con solo 3-5 meses. Si la
# diferencia de error no es estadísticamente significativa (p >=
# NIVEL_SIGNIFICANCIA_DM), se prefiere "Media móvil" en vez del que
# ganó por una diferencia que podría ser puro ruido.
def elegir_campeon_por_producto(resultados_prediccion):
    """
    Devuelve una Series indexada por producto_id con el nombre del
    método ("Media móvil"/"Regresión Lineal"/"XGBoost") elegido para ese
    producto: el de menor MAE en los meses de backtest, siempre que esa
    diferencia contra el segundo mejor sea estadísticamente significativa
    (prueba de Diebold-Mariano con corrección de Harvey-Leybourne-Newbold,
    ver comentario arriba); si no lo es, se usa "Media móvil" por ser el
    método más simple y robusto.

    El cálculo del MAE por producto sigue vectorizado (ver comentario
    original: antes esto usaba groupby().apply(), ~15x más lento en un
    catálogo de 500 productos) — la comparación estadística se arma con
    la misma lógica: primero se identifica el mejor y segundo mejor
    método por producto (np.argsort sobre la tabla de MAE), y recién
    después se calcula la prueba, también vectorizada sobre todos los
    productos a la vez.
    """
    errores_por_metodo = pd.DataFrame({
        metodo: (resultados_prediccion["demanda"] - resultados_prediccion[columna]).abs()
        for metodo, columna in COLUMNA_PREDICCION_POR_METODO.items()
    })
    productos = resultados_prediccion["producto_id"]
    errores_por_producto = errores_por_metodo.groupby(productos).mean()

    metodos = errores_por_producto.columns.to_numpy()
    orden = np.argsort(errores_por_producto.to_numpy(), axis=1)
    mejor_metodo = pd.Series(metodos[orden[:, 0]], index=errores_por_producto.index)
    segundo_metodo = pd.Series(metodos[orden[:, 1]], index=errores_por_producto.index)

    # Error del método mejor y del segundo mejor, mes a mes (una fila
    # por producto-mes), para poder comparar la diferencia a lo largo
    # del período de backtest — no solo su promedio.
    mejor_por_fila = productos.map(mejor_metodo)
    segundo_por_fila = productos.map(segundo_metodo)
    error_mejor = np.select(
        [mejor_por_fila == metodo for metodo in errores_por_metodo.columns],
        [errores_por_metodo[metodo] for metodo in errores_por_metodo.columns],
    )
    error_segundo = np.select(
        [segundo_por_fila == metodo for metodo in errores_por_metodo.columns],
        [errores_por_metodo[metodo] for metodo in errores_por_metodo.columns],
    )
    diferencia = pd.Series(error_mejor - error_segundo, index=resultados_prediccion.index)

    stats_diferencia = diferencia.groupby(productos).agg(d_media="mean", d_std="std", n="count")

    with np.errstate(invalid="ignore", divide="ignore"):
        error_estandar = stats_diferencia["d_std"] / np.sqrt(stats_diferencia["n"])
        dm = stats_diferencia["d_media"] / error_estandar
        # Corrección de Harvey, Leybourne y Newbold (1997) para muestras
        # chicas, con h=1 (backtest de un paso): DM* = DM * sqrt((n-1)/n),
        # comparado contra t de Student con n-1 grados de libertad en vez
        # de la normal estándar que usa la prueba clásica.
        correccion_hln = np.sqrt((stats_diferencia["n"] - 1) / stats_diferencia["n"])
        dm_ajustado = dm * correccion_hln
        grados_libertad = (stats_diferencia["n"] - 1).clip(lower=1)
        p_valor = pd.Series(
            2 * (1 - stats.t.cdf(dm_ajustado.abs(), df=grados_libertad)),
            index=stats_diferencia.index,
        )

    # Sin al menos 2 meses no hay forma de medir varianza de la
    # diferencia; y si los dos métodos se equivocaron exactamente igual
    # mes a mes (d_std == 0), no hay evidencia de que uno sea mejor —
    # ambos casos se tratan como "no significativo".
    suficientes_datos = stats_diferencia["n"] >= 2
    sin_varianza = stats_diferencia["d_std"].fillna(0) == 0
    significativo = suficientes_datos & (p_valor < NIVEL_SIGNIFICANCIA_DM) & ~sin_varianza

    campeon = mejor_metodo.copy()
    campeon[~significativo] = "Media móvil"
    return campeon


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
