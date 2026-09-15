import numpy as np


def construir_features(df_mensual, producto_comportamiento):
    """
    Enriquece df_mensual con las variables que usan los modelos (fecha
    cíclica, lags, medias/desviaciones móviles, estadísticas históricas
    por producto), arma df_model (solo filas con suficiente historial
    para tener lags válidos) y la lista final de columnas de features
    según lo que traiga el CSV (precio/inventario son opcionales).

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

    # Ejemplo 3 (Kaggle Grupo Bimbo):
    stats_cols = ["demanda_total", "media", "desviacion"]
    df_mensual = df_mensual.merge(
        producto_comportamiento[["producto_id"] + stats_cols].rename(
            columns={"media": "media_producto", "desviacion": "desviacion_producto"}
        ),
        on="producto_id", how="left"
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
        "media_producto", "desviacion_producto"
    ]
    if "precio" in df_model.columns:
        # Solo el NIVEL de precio (estable, se puede seguir usando el
        # último precio conocido al pronosticar). Se excluyó la
        # variación de precio mes a mes: es un feature muy ruidoso
        # (llegaba a acaparar >65% de la importancia del modelo) del
        # que no se puede conocer el valor futuro real, y forzarlo a 0
        # para el pronóstico hacía que el modelo colapsara.
        feature_cols.append("precio")
    if "inventario" in df_model.columns:
        feature_cols.append("inventario")
        if "inventario_lag_1" in df_model.columns:
            feature_cols.append("inventario_lag_1")

    for col in feature_cols:
        if df_model[col].isna().any():
            df_model[col] = df_model[col].fillna(0)

    return df_mensual, df_model, feature_cols
