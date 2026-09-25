"""
Pruebas de VERIFICACIÓN (oráculo conocido, mismo criterio que
test_modelos.py) para la combinación de pronósticos por mínimos
cuadrados (app/services/prediccion/combinacion.py).

El peso w le da a la predicción A la fracción que minimiza el error
cuadrático de w*A + (1-w)*B contra la demanda real. Los casos de abajo
se resuelven a mano: en cada uno, el peso ideal es obvio.
"""

import numpy as np
import pandas as pd

from app.services.prediccion.combinacion import (
    _peso_minimos_cuadrados,
    evaluar_combinacion,
)


def test_peso_todo_para_el_modelo_perfecto():
    """A acierta exacto y B se pasa por 5 siempre:
      y - B = -5, A - B = -5 -> w = (25 + 25 + 25) / (25 + 25 + 25) = 1."""
    y = np.array([10.0, 20.0, 30.0])
    pred_a = y.copy()
    pred_b = y + 5
    assert _peso_minimos_cuadrados(y, pred_a, pred_b) == 1.0


def test_peso_cero_si_el_otro_modelo_es_el_perfecto():
    """B acierta exacto y A se pasa por 5 siempre:
      y - B = 0 -> el numerador es 0 -> w = 0."""
    y = np.array([10.0, 20.0, 30.0])
    pred_a = y + 5
    pred_b = y.copy()
    assert _peso_minimos_cuadrados(y, pred_a, pred_b) == 0.0


def test_peso_a_medias_si_se_equivocan_igual_pero_para_lados_opuestos():
    """A = y + 5 y B = y - 5: el promedio de los dos acierta exacto.
      y - B = 5, A - B = 10 -> w = (50 + 50 + 50) / (100 + 100 + 100) = 0.5."""
    y = np.array([10.0, 20.0, 30.0])
    pred_a = y + 5
    pred_b = y - 5
    assert _peso_minimos_cuadrados(y, pred_a, pred_b) == 0.5


def test_peso_se_recorta_a_cero_uno():
    """Si la solución de mínimos cuadrados cae fuera de [0, 1] (acá A y B
    quedan los dos por debajo de y, así que lo ideal sería extrapolar más
    allá de A), el peso se recorta a 1 para que siga siendo un promedio
    ponderado: y - B = 8 y A - B = 2 -> w = 16 / 4 = 4 -> se recorta a 1."""
    y = np.array([10.0, 10.0])
    pred_a = np.array([4.0, 4.0])
    pred_b = np.array([2.0, 2.0])
    assert _peso_minimos_cuadrados(y, pred_a, pred_b) == 1.0


def test_peso_con_predicciones_identicas_es_un_medio():
    """Si los dos modelos predicen lo mismo, no hay nada que decidir."""
    y = np.array([10.0, 20.0])
    pred = np.array([9.0, 21.0])
    assert _peso_minimos_cuadrados(y, pred, pred) == 0.5


def _tabla_backtest():
    """Tres meses, dos productos. XGBoost siempre acierta exacto; Media
    móvil siempre se pasa por 10. La combinación tendría que apoyarse
    casi por completo en XGBoost."""
    filas = []
    for mes in ["2024-01-01", "2024-02-01", "2024-03-01"]:
        for producto, demanda in [("A", 100.0), ("B", 200.0)]:
            filas.append({
                "fecha": mes, "producto_id": producto, "demanda": demanda,
                "prediccion_xgboost": demanda,
                "prediccion_media_movil": demanda + 10,
                "prediccion_regresion_lineal": demanda + 3,
            })
    return filas


def test_evaluar_combinacion_devuelve_las_tres_opciones_y_el_peso():
    filas, detalle = evaluar_combinacion(_tabla_backtest())

    assert [f["Modelo"] for f in filas] == [
        "Media móvil sola", "XGBoost solo", "Combinación XGBoost + Media móvil",
    ]
    # XGBoost acierta exacto (MAE 0); Media móvil se pasa por 10 (MAE 10).
    assert filas[1]["MAE"] == 0.0
    assert filas[0]["MAE"] == 10.0
    # Los pesos salen completos a XGBoost, así que la combinación también
    # acierta exacto — igual de buena que el mejor de los dos solos.
    assert detalle["peso_xgboost"] == 1.0
    assert detalle["peso_media_movil"] == 0.0
    assert filas[2]["MAE"] == 0.0
    assert set(detalle["pesos_por_mes"]) == {"2024-01-01", "2024-02-01", "2024-03-01"}


def test_evaluar_combinacion_sin_predicciones_devuelve_none():
    assert evaluar_combinacion([]) == (None, None)
