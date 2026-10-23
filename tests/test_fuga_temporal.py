"""
Pruebas de no fuga temporal en las variables por producto
(media_producto / desviacion_producto).

Qué protege: las estadísticas que usa el modelo para un corte temporal
no pueden depender de meses posteriores a ese corte. Si dependen, el
modelo ve parte del período que después se usa para medirlo.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.prediccion.features import (
    construir_features,
    incorporar_estadisticas_producto,
)
from app.services.prediccion.modelos import entrenar_xgboost
from app.services.prediccion.regresion_lineal import entrenar_regresion_lineal
from app.services.prediccion.validacion import elegir_campeon_fold
from app.services.prediccion.pronostico_futuro import (
    evaluar_pronostico_recursivo,
    resumen_pronostico_por_horizonte,
)


def _df_mensual(demandas_por_producto, inicio="2021-01-01"):
    """Construye df_mensual (producto_id, fecha, demanda) con demandas dadas."""
    filas = []
    for producto, demandas in demandas_por_producto.items():
        fechas = pd.date_range(inicio, periods=len(demandas), freq="MS")
        for fecha, demanda in zip(fechas, demandas):
            filas.append({"producto_id": producto, "fecha": fecha, "demanda": float(demanda)})
    return pd.DataFrame(filas)


def _producto_comportamiento(df_mensual):
    productos = df_mensual["producto_id"].unique()
    return pd.DataFrame({
        "producto_id": productos,
        "categoria_abc": "A",
        "variabilidad": "Media",
    })


def test_estadisticas_solo_usan_meses_hasta_el_corte():
    df_mensual = _df_mensual({"A": [10, 10, 10, 500, 900], "B": [1, 2, 3, 4, 5]})
    df = df_mensual[["producto_id", "fecha"]].drop_duplicates()
    corte = pd.Timestamp("2021-03-01")

    resultado = incorporar_estadisticas_producto(df, df_mensual, fecha_corte=corte)
    a = resultado[resultado["producto_id"] == "A"].iloc[0]

    # Hasta marzo, A vendió 10 todos los meses: media 10, desvío 0.
    # Los 500 y 900 posteriores no deben aparecer.
    assert a["media_producto"] == pytest.approx(10.0)
    assert a["desviacion_producto"] == pytest.approx(0.0)


def test_cambiar_datos_posteriores_al_corte_no_altera_estadisticas():
    base = _df_mensual({"A": [10, 12, 11, 500, 900], "B": [1, 2, 3, 4, 5]})
    alterado = base.copy()
    alterado.loc[alterado["fecha"] >= "2021-04-01", "demanda"] = 77.0
    df = base[["producto_id", "fecha"]].drop_duplicates()
    corte = pd.Timestamp("2021-03-01")

    primero = incorporar_estadisticas_producto(df, base, fecha_corte=corte)
    segundo = incorporar_estadisticas_producto(df, alterado, fecha_corte=corte)

    pd.testing.assert_frame_equal(primero, segundo)


def test_incorporar_reemplaza_estadisticas_previas_en_vez_de_conservarlas():
    df_mensual = _df_mensual({"A": [10, 10, 10, 500, 900]})
    df = df_mensual[["producto_id", "fecha"]].drop_duplicates()

    # Primero con un corte que incluye los meses altos...
    con_corte_tardio = incorporar_estadisticas_producto(df, df_mensual, pd.Timestamp("2021-05-01"))
    # ...y luego con uno temprano: debe reemplazar, no quedarse con el anterior.
    reaplicado = incorporar_estadisticas_producto(con_corte_tardio, df_mensual, pd.Timestamp("2021-03-01"))
    directo = incorporar_estadisticas_producto(df, df_mensual, pd.Timestamp("2021-03-01"))

    pd.testing.assert_frame_equal(reaplicado, directo)


def test_modelo_entrenado_no_depende_de_meses_de_prueba():
    """
    Entrena XGBoost solo con meses de entrenamiento (con estadísticas
    calculadas hasta el corte) y luego cambia la demanda de los meses de
    prueba. Las predicciones sobre los meses de entrenamiento NO deben
    cambiar: el modelo nunca vio esos meses de prueba.

    Antes de la corrección, media_producto/desviacion_producto salían del
    historial completo, así que este test fallaba.
    """
    rng = np.random.default_rng(0)
    n_meses = 20
    demandas = {
        p: list(rng.integers(5, 60, size=n_meses)) for p in ["A", "B", "C"]
    }
    df_mensual = _df_mensual(demandas)
    comportamiento = _producto_comportamiento(df_mensual)

    meses = sorted(df_mensual["fecha"].unique())
    corte = meses[14]  # entrenamiento hasta el mes 15; prueba después
    meses_train = [m for m in meses if m <= corte]

    def predicciones_sobre_entrenamiento(df_mensual_actual):
        _, df_model, feature_cols = construir_features(df_mensual_actual, comportamiento)
        df_model = incorporar_estadisticas_producto(df_model, df_mensual_actual, fecha_corte=corte)
        train = df_model[df_model["fecha"].isin(meses_train)]
        model, pred = entrenar_xgboost(
            train[feature_cols], np.log1p(train["demanda"]), train[feature_cols]
        )
        return pred

    pred_original = predicciones_sobre_entrenamiento(df_mensual)

    # Cambiar solo los meses de prueba (posteriores al corte)
    alterado = df_mensual.copy()
    mascara_prueba = alterado["fecha"] > corte
    alterado.loc[mascara_prueba, "demanda"] = alterado.loc[mascara_prueba, "demanda"] * 10 + 300
    pred_alterado = predicciones_sobre_entrenamiento(alterado)

    assert len(pred_original) == len(pred_alterado) > 0
    np.testing.assert_allclose(pred_original, pred_alterado, rtol=1e-6, atol=1e-6)


def test_producto_sin_historia_hasta_el_corte_queda_en_cero():
    # NUEVO solo tiene demanda en abril, después del corte (marzo).
    df_mensual = _df_mensual({"A": [5, 5, 5, 5]})
    fila_nuevo = pd.DataFrame({
        "producto_id": ["NUEVO"], "fecha": [pd.Timestamp("2021-04-01")], "demanda": [40.0],
    })
    df_mensual = pd.concat([df_mensual, fila_nuevo], ignore_index=True)
    df = df_mensual[["producto_id", "fecha"]].drop_duplicates()

    resultado = incorporar_estadisticas_producto(df, df_mensual, pd.Timestamp("2021-03-01"))
    nuevo = resultado[resultado["producto_id"] == "NUEVO"]
    assert (nuevo["media_producto"] == 0).all()
    assert (nuevo["desviacion_producto"] == 0).all()


def _datos_sinteticos(n_meses=20, semilla=0):
    rng = np.random.default_rng(semilla)
    demandas = {p: list(rng.integers(5, 60, size=n_meses)) for p in ["A", "B", "C", "D"]}
    return _df_mensual(demandas)


def _modelos_hasta(df_mensual, corte, comportamiento):
    """Entrena XGBoost y Regresión Lineal solo con meses hasta 'corte'."""
    _, df_model, feature_cols = construir_features(df_mensual, comportamiento)
    df_model = incorporar_estadisticas_producto(df_model, df_mensual, fecha_corte=corte)
    train = df_model[df_model["fecha"] <= corte]
    model, _ = entrenar_xgboost(train[feature_cols], np.log1p(train["demanda"]), train[feature_cols])
    modelo_lr, _, features_lr = entrenar_regresion_lineal(
        train[feature_cols], train["demanda"], train[feature_cols], feature_cols
    )
    return model, modelo_lr, features_lr, feature_cols


def test_campeon_de_fold_no_depende_de_su_propio_periodo_de_prueba():
    """
    A.3: el método elegido para un fold solo puede depender de meses
    anteriores a su test. Alterar la demanda del test del fold no debe
    cambiar el campeón.
    """
    df_mensual = _datos_sinteticos()
    comportamiento = _producto_comportamiento(df_mensual)
    _, df_model, feature_cols = construir_features(df_mensual, comportamiento)
    meses = sorted(df_mensual["fecha"].unique())
    meses_train, meses_test = meses[:14], meses[14:]

    original = elegir_campeon_fold(df_model, df_mensual, feature_cols, meses_train, meses_test)

    alterado = df_mensual.copy()
    mascara = alterado["fecha"].isin(meses_test)
    alterado.loc[mascara, "demanda"] = alterado.loc[mascara, "demanda"] * 7 + 500
    _, df_model_alt, _ = construir_features(alterado, comportamiento)
    nuevo = elegir_campeon_fold(df_model_alt, alterado, feature_cols, meses_train, meses_test)

    pd.testing.assert_series_equal(original, nuevo)


def test_campeon_de_fold_es_none_sin_historia_suficiente():
    df_mensual = _datos_sinteticos(n_meses=8)
    comportamiento = _producto_comportamiento(df_mensual)
    _, df_model, feature_cols = construir_features(df_mensual, comportamiento)
    meses = sorted(df_mensual["fecha"].unique())
    # Solo 3 meses de entrenamiento: no alcanza para validar sin usar el test.
    assert elegir_campeon_fold(df_model, df_mensual, feature_cols, meses[:3], meses[3:6]) is None


def test_pronostico_desde_origen_no_depende_de_meses_posteriores():
    """
    A.4: pronosticar desde un origen solo puede usar historia hasta ese
    origen. Alterar la demanda posterior no debe cambiar ninguna
    predicción, solo los valores reales con los que se compara.
    """
    df_mensual = _datos_sinteticos()
    comportamiento = _producto_comportamiento(df_mensual)
    meses = sorted(df_mensual["fecha"].unique())
    origen = meses[14]
    model, modelo_lr, features_lr, feature_cols = _modelos_hasta(df_mensual, origen, comportamiento)
    comportamiento = comportamiento.assign(metodo_campeon="XGBoost")

    original, _ = evaluar_pronostico_recursivo(
        df_mensual, comportamiento, model, feature_cols, modelo_lr, features_lr,
        origen=origen, horizonte_meses=5,
    )

    alterado = df_mensual.copy()
    mascara = alterado["fecha"] > origen
    alterado.loc[mascara, "demanda"] = alterado.loc[mascara, "demanda"] * 3 + 100
    nuevo, _ = evaluar_pronostico_recursivo(
        alterado, comportamiento, model, feature_cols, modelo_lr, features_lr,
        origen=origen, horizonte_meses=5,
    )

    assert len(original) > 0
    np.testing.assert_allclose(original["prediccion_final"].values, nuevo["prediccion_final"].values)
    np.testing.assert_allclose(original["prediccion_baseline"].values, nuevo["prediccion_baseline"].values)
    assert not np.allclose(original["demanda"].values, nuevo["demanda"].values)


def test_evaluacion_por_horizonte_tiene_una_fila_por_mes_pronosticado():
    df_mensual = _datos_sinteticos()
    comportamiento = _producto_comportamiento(df_mensual).assign(metodo_campeon="XGBoost")
    meses = sorted(df_mensual["fecha"].unique())
    origen = meses[14]
    model, modelo_lr, features_lr, feature_cols = _modelos_hasta(df_mensual, origen, comportamiento)

    comparacion, sin_pronostico = evaluar_pronostico_recursivo(
        df_mensual, comportamiento, model, feature_cols, modelo_lr, features_lr,
        origen=origen, horizonte_meses=5,
    )
    resumen = resumen_pronostico_por_horizonte(comparacion)

    assert sin_pronostico == 0
    assert [fila["horizonte"] for fila in resumen] == [1, 2, 3, 4, 5]
    assert all(fila["productos"] == 4 for fila in resumen)
    assert all(fila["mae_sistema"] >= 0 for fila in resumen)
