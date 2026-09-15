import numpy as np
import pandas as pd

from app.services.prediccion.utils import nan_a_none


def _valor_mas_frecuente(serie):
    serie = serie.dropna()
    return serie.mode().iloc[0] if not serie.empty else None


# EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de referencia.
# Ninguno de los 4 ejemplos clasifica productos por importancia (ABC) ni
# variabilidad. Acá se usa como segmentación descriptiva para el
# dashboard — qué tan importante es cada producto y qué tan errático es
# su consumo — mientras que la elección de QUÉ MÉTODO pronostica cada
# producto se decide aparte, de forma empírica, en
# modelos.py::elegir_campeon_por_producto (mide el error de backtest de
# los tres métodos por producto y usa el que ganó), no con una regla
# fija basada en esta clasificación.
def calcular_comportamiento_producto(df, df_mensual):
    """
    Clasifica cada producto por importancia (ABC, según % acumulado de
    demanda) y por variabilidad (coeficiente de variación), y le agrega
    etiquetas descriptivas (categoría, nombre) para el dashboard — esto
    último tampoco viene de los ejemplos, es una mejora de usabilidad
    propia (los productos del CSV original solo tienen un ID numérico).
    """
    producto_comportamiento = (
        df_mensual.groupby("producto_id")["demanda"]
        .agg(demanda_total="sum", media="mean", desviacion="std")
        .reset_index()
    )
    producto_comportamiento["cv"] = (
        producto_comportamiento["desviacion"] / producto_comportamiento["media"]
    ).replace([np.inf, -np.inf], np.nan)

    def clasificar_variabilidad(cv):
        if pd.isna(cv):
            return "Sin datos"
        if cv < 0.5:
            return "Baja"
        elif cv < 1:
            return "Media"
        else:
            return "Alta"

    producto_comportamiento["variabilidad"] = producto_comportamiento["cv"].apply(clasificar_variabilidad)

    producto_comportamiento = producto_comportamiento.sort_values("demanda_total", ascending=False).reset_index(drop=True)
    total_demanda = producto_comportamiento["demanda_total"].sum()
    if total_demanda > 0:
        producto_comportamiento["demanda_acumulada_pct"] = producto_comportamiento["demanda_total"].cumsum() / total_demanda
    else:
        producto_comportamiento["demanda_acumulada_pct"] = 0

    def clasificar_abc(pct):
        if pct <= 0.80:
            return "A"
        elif pct <= 0.95:
            return "B"
        else:
            return "C"

    producto_comportamiento["categoria_abc"] = producto_comportamiento["demanda_acumulada_pct"].apply(clasificar_abc)

    # Etiquetas descriptivas del producto (categoría y, si el CSV la
    # trae, nombre). Los productos solo tienen un ID numérico — sin
    # esto el dashboard sería una tabla de números sin contexto. Se usa
    # el valor MÁS FRECUENTE por producto en los datos crudos (moda),
    # no el "primero": en un inventario real la categoría de un
    # producto no cambia mes a mes, pero si el CSV tiene alguna fila
    # inconsistente, la moda es más representativa que quedarse con
    # cualquier fila al azar.
    columnas_etiqueta = [
        c for c in ["categoria", "nombre_producto", "proveedor_principal", "proveedor_alterno"]
        if c in df.columns
    ]
    if columnas_etiqueta:
        etiquetas_producto = (
            df.groupby("producto_id")[columnas_etiqueta]
            .agg(_valor_mas_frecuente)
            .reset_index()
        )
        producto_comportamiento = producto_comportamiento.merge(
            etiquetas_producto, on="producto_id", how="left"
        )
    columnas_texto_opcionales = ["categoria", "nombre_producto", "proveedor_principal", "proveedor_alterno"]
    for col in columnas_texto_opcionales:
        if col not in producto_comportamiento.columns:
            producto_comportamiento[col] = None
    producto_comportamiento = nan_a_none(producto_comportamiento, columnas_texto_opcionales)

    return producto_comportamiento
