"""
Fuga de información por inventario del mismo mes.

El inventario de cierre de un mes depende de las ventas de ese mes. Si entra
como variable para predecir la demanda de ese mismo mes, el modelo ve la
respuesta. Estas pruebas comprueban que solo se usa el inventario del mes
anterior (inventario_lag_1), que sí se conoce al pronosticar.
"""

import numpy as np
import pandas as pd

from app.services.prediccion.features import construir_features


def _datos():
    meses = pd.date_range("2025-01-01", periods=8, freq="MS")
    df = pd.DataFrame({
        "producto_id": ["P"] * len(meses),
        "fecha": meses,
        "demanda": np.arange(10, 10 + len(meses), dtype=float),
        "inventario": np.arange(100, 100 + len(meses), dtype=float),
        "nombre_producto": "P",
        "categoria": "T",
    })
    comportamiento = pd.DataFrame([{
        "producto_id": "P", "categoria_abc": "A", "variabilidad": "Baja",
    }])
    return df, comportamiento


def test_inventario_del_mismo_mes_no_es_feature():
    df, comportamiento = _datos()
    _, _, feature_cols = construir_features(df, comportamiento)
    assert "inventario" not in feature_cols


def test_inventario_del_mes_anterior_si_es_feature():
    df, comportamiento = _datos()
    _, _, feature_cols = construir_features(df, comportamiento)
    assert "inventario_lag_1" in feature_cols


def test_inventario_lag_1_es_el_del_mes_anterior():
    df, comportamiento = _datos()
    df_mensual, df_model, _ = construir_features(df, comportamiento)
    fila = df_mensual[df_mensual["fecha"] == pd.Timestamp("2025-04-01")].iloc[0]
    assert fila["inventario_lag_1"] == fila["inventario"] - 1 == 102
