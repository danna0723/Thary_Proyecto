import numpy as np


# EXTENSIÓN PROPIA (corrección de fuga temporal) — media y desvío de la
# demanda mensual de cada producto. Son predictoras, pero NO se calculan
# acá: dependen del corte temporal (ver incorporar_estadisticas_producto).
# Antes se tomaban de todo el historial del CSV, incluidos los meses de
# prueba y de validación, así que el modelo "conocía" parte del período
# que después se usaba para medirlo.
COLUMNAS_ESTADISTICAS_PRODUCTO = ["media_producto", "desviacion_producto"]


def incorporar_estadisticas_producto(df, df_mensual, fecha_corte=None):
    """
    Agrega a "df" las columnas media_producto y desviacion_producto,
    calculadas SOLO con los meses de df_mensual hasta fecha_corte
    (inclusive). Con fecha_corte=None usa todo el historial; eso solo
    corresponde al pronóstico futuro, donde todo el historial ya es
    conocido — nunca a filas que se van a usar para evaluar.

    Devuelve una copia de df. Si df ya traía esas columnas (calculadas
    con otro corte), se reemplazan, no se conservan. Productos sin
    historia hasta el corte quedan con 0 (mismo criterio que los demás
    NaN de las features).
    """
    historia = df_mensual if fecha_corte is None else df_mensual[df_mensual["fecha"] <= fecha_corte]
    estadisticas = (
        historia.groupby("producto_id")["demanda"]
        .agg(media_producto="mean", desviacion_producto="std")
        .reset_index()
    )
    base = df.drop(columns=[c for c in COLUMNAS_ESTADISTICAS_PRODUCTO if c in df.columns])
    resultado = base.merge(estadisticas, on="producto_id", how="left")
    for col in COLUMNAS_ESTADISTICAS_PRODUCTO:
        resultado[col] = resultado[col].fillna(0)
    return resultado


def construir_features(df_mensual, producto_comportamiento):
    """
    Enriquece df_mensual con las variables que usan los modelos (fecha
    cíclica, lags, medias/desviaciones móviles), arma df_model (solo filas
    con suficiente historial para tener lags válidos) y la lista final de
    columnas de features según lo que traiga el CSV (precio/inventario son
    opcionales).

    Las columnas de COLUMNAS_ESTADISTICAS_PRODUCTO aparecen en feature_cols
    pero NO se calculan acá: hay que llamar a incorporar_estadisticas_producto
    sobre df_model con el corte del split correspondiente antes de usarlas.

    Devuelve (df_mensual, df_model, feature_cols).
    """
    df_mensual = df_mensual.merge(
        producto_comportamiento[["producto_id", "categoria_abc", "variabilidad"]],
        on="producto_id", how="left"
    )

    df_mensual["año"] = df_mensual["fecha"].dt.year
    df_mensual["mes"] = df_mensual["fecha"].dt.month

    # EXTENSIÓN PROPIA — no viene de ningún ejemplo de referencia (los 4
    # usan mes/estación como valor entero o categoría dummy, no cíclico).
    # Codificación cíclica en vez de "mes"/"trimestre" crudos: con solo
    # ~12 meses de historia, los primeros 3 meses del dataset siempre
    # quedan fuera del entrenamiento (no alcanzan a tener lag_1..lag_3),
    # así que ese valor de mes nunca aparece como target de entrenamiento.
    # seno/coseno lo representan como un punto continuo entre los meses
    # vecinos que sí se entrenaron, en vez de una categoría nunca vista
    # (que hacía que XGBoost extrapolara mal y colapsara la predicción
    # cerca de 0 para el mes siguiente al último del CSV).
    df_mensual["mes_sin"] = np.sin(2 * np.pi * df_mensual["mes"] / 12)
    df_mensual["mes_cos"] = np.cos(2 * np.pi * df_mensual["mes"] / 12)

    # Ejemplo 1 (SKU Demand Forecasting)
    df_mensual["lag_1"] = df_mensual.groupby("producto_id")["demanda"].shift(1)
    df_mensual["lag_2"] = df_mensual.groupby("producto_id")["demanda"].shift(2)
    df_mensual["lag_3"] = df_mensual.groupby("producto_id")["demanda"].shift(3)
    df_mensual["lag_6"] = df_mensual.groupby("producto_id")["demanda"].shift(6)

    df_mensual["media_movil_3"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df_mensual["media_movil_6"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
    df_mensual["desviacion_movil_3"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(3).std()
    )

    if "inventario" in df_mensual.columns:
        df_mensual["inventario_lag_1"] = df_mensual.groupby("producto_id")["inventario"].shift(1)

    feature_history = ["lag_1", "lag_2", "lag_3", "media_movil_3"]
    df_model = df_mensual.dropna(subset=feature_history).copy()

    if len(df_model) == 0:
        raise ValueError(
            "Ningún producto tiene al menos 3 meses seguidos de historial, que es lo mínimo "
            "para calcular tendencias. Sube un archivo con más meses continuos de datos."
        )

    feature_cols = [
        "año", "mes_sin", "mes_cos",
        "lag_1", "lag_2", "lag_3", "lag_6",
        "media_movil_3", "media_movil_6", "desviacion_movil_3",
        *COLUMNAS_ESTADISTICAS_PRODUCTO
    ]
    if "precio" in df_model.columns:
        # Solo el NIVEL de precio (estable, se puede seguir usando el
        # último precio conocido al pronosticar). Se excluyó la
        # variación de precio mes a mes: es un feature muy ruidoso
        # (llegaba a acaparar >65% de la importancia del modelo) del
        # que no se puede conocer el valor futuro real, y forzarlo a 0
        # para el pronóstico hacía que el modelo colapsara.
        feature_cols.append("precio")
    # CORRECCIÓN DE FUGA: el inventario del MISMO mes no se usa como
    # variable. Es un saldo de cierre que depende de las ventas de ese mes,
    # así que en el entrenamiento filtra la demanda que se quiere predecir,
    # y al pronosticar no existe (se reemplazaba por el último valor
    # conocido). Solo se usa el inventario del mes anterior, que sí se
    # conoce antes de pronosticar el mes siguiente.
    if "inventario_lag_1" in df_model.columns:
        feature_cols.append("inventario_lag_1")

    for col in feature_cols:
        if col in COLUMNAS_ESTADISTICAS_PRODUCTO:
            continue  # se rellenan en incorporar_estadisticas_producto
        if df_model[col].isna().any():
            df_model[col] = df_model[col].fillna(0)

    return df_mensual, df_model, feature_cols
