"""
Pruebas de VERIFICACIÓN (oráculo conocido, mismo criterio que
test_reposicion.py) para elegir_campeon_por_producto y
seleccionar_prediccion_final (app/services/prediccion/modelos.py).

Estas dos funciones se reescribieron para ser más rápidas (ver el
comentario en el código: antes usaban groupby().apply()/apply(axis=1),
que en un catálogo de cientos de productos resultó ser más de la
mitad del tiempo total de procesar un archivo — medido con cProfile).
Estas pruebas confirman que la versión vectorizada da EXACTAMENTE el
mismo resultado que la versión anterior fila por fila, con valores
calculados a mano.
"""

import pandas as pd
import pytest

from app.services.prediccion.modelos import (
    COLUMNA_PREDICCION_POR_METODO,
    elegir_campeon_por_producto,
    seleccionar_prediccion_final,
)


def test_elegir_campeon_por_producto():
    """
    Cálculo a mano (MAE = promedio del error absoluto):

    Producto A — demanda real [10, 20]:
      Media móvil       [12, 18] -> errores [2, 2]   -> MAE = 2.0   (el más bajo)
      Regresión Lineal  [10, 25] -> errores [0, 5]   -> MAE = 2.5
      XGBoost           [15, 15] -> errores [5, 5]   -> MAE = 5.0
      -> gana "Media móvil"

    Producto B — demanda real [100, 200]:
      Media móvil       [50, 50]   -> errores [50, 150] -> MAE = 100.0
      Regresión Lineal  [100, 210] -> errores [0, 10]   -> MAE = 5.0   (el más bajo)
      XGBoost           [90, 190]  -> errores [10, 10]  -> MAE = 10.0
      -> gana "Regresión Lineal"
    """
    tabla = pd.DataFrame({
        "producto_id": ["A", "A", "B", "B"],
        "demanda": [10, 20, 100, 200],
        "prediccion_media_movil": [12, 18, 50, 50],
        "prediccion_regresion_lineal": [10, 25, 100, 210],
        "prediccion_xgboost": [15, 15, 90, 190],
    })

    campeon = elegir_campeon_por_producto(tabla)

    assert campeon["A"] == "Media móvil"
    assert campeon["B"] == "Regresión Lineal"


def test_elegir_campeon_por_producto_con_empate_toma_el_primero():
    """Si dos métodos empatan en MAE, idxmin() se queda con el primero
    según el orden de COLUMNA_PREDICCION_POR_METODO (Media móvil,
    Regresión Lineal, XGBoost) — mismo comportamiento que antes, ya que
    pandas Series.idxmin() siempre devuelve la primera ocurrencia."""
    tabla = pd.DataFrame({
        "producto_id": ["C", "C"],
        "demanda": [10, 10],
        "prediccion_media_movil": [12, 12],   # error 2
        "prediccion_regresion_lineal": [8, 8],  # error 2 (empate)
        "prediccion_xgboost": [20, 20],       # error 10
    })
    campeon = elegir_campeon_por_producto(tabla)
    assert campeon["C"] == "Media móvil"


def test_seleccionar_prediccion_final():
    """Para cada fila, prediccion_final tiene que ser el valor de la
    columna que corresponde al método en metodo_usado — ni el de otro
    método, ni un promedio entre todos."""
    tabla = pd.DataFrame({
        "metodo_usado": ["Media móvil", "Regresión Lineal", "XGBoost"],
        "prediccion_media_movil": [1, 2, 3],
        "prediccion_regresion_lineal": [10, 20, 30],
        "prediccion_xgboost": [100, 200, 300],
    })
    resultado = seleccionar_prediccion_final(tabla)
    assert list(resultado) == [1, 20, 300]


def test_seleccionar_prediccion_final_cubre_los_tres_metodos():
    """Confirma que no se está usando por error siempre la primera
    columna de COLUMNA_PREDICCION_POR_METODO — se prueban los 3 métodos
    posibles explícitamente."""
    tabla = pd.DataFrame({
        "metodo_usado": list(COLUMNA_PREDICCION_POR_METODO.keys()),
        "prediccion_media_movil": [111, 111, 111],
        "prediccion_regresion_lineal": [222, 222, 222],
        "prediccion_xgboost": [333, 333, 333],
    })
    resultado = seleccionar_prediccion_final(tabla)
    assert list(resultado) == [111, 222, 333]
