"""
Pruebas de VERIFICACIÓN (oráculo conocido, mismo criterio que
test_reposicion.py) para elegir_campeon_por_producto y
seleccionar_prediccion_final (app/services/prediccion/modelos.py).

seleccionar_prediccion_final se reescribió para ser más rápida (ver el
comentario en el código: antes usaba apply(axis=1), que en un catálogo
de cientos de productos resultó ser más de la mitad del tiempo total de
procesar un archivo — medido con cProfile). Esas pruebas confirman que
la versión vectorizada da EXACTAMENTE el mismo resultado que la versión
anterior fila por fila.

elegir_campeon_por_producto ya no elige simplemente el método de menor
MAE: antes de declarar ganador, compara ese método contra el segundo
mejor con la prueba de Diebold-Mariano (con la corrección de Harvey-
Leybourne-Newbold para muestras chicas — ver comentario en el código
fuente). Si la diferencia no es estadísticamente significativa, se
prefiere "Media móvil" en vez del que ganó por una diferencia que podría
ser puro ruido — relevante porque el backtest real tiene muy pocos meses
(3 a 5). Los valores esperados de estas pruebas ya incorporan ese
comportamiento, no solo el MAE.

construir_tabla_predicciones ya NO elige su propio campeón (antes lo
calculaba con la misma tabla de test que después evaluaba — fuga de
datos: el "sistema final" casi no podía perder contra Media Móvil en
ese test, porque la elección ya conocía la respuesta de ese mismo
período). Ahora recibe el campeón como parámetro, ya decidido con un
período de VALIDACIÓN distinto — las pruebas de más abajo confirman que
usa EXACTAMENTE el que se le pasa, sin recalcular uno propio a partir
de la demanda/predicciones de test_df.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.prediccion.modelos import (
    COLUMNA_PREDICCION_POR_METODO,
    calcular_errores_por_producto,
    construir_tabla_predicciones,
    elegir_campeon_por_producto,
    seleccionar_prediccion_final,
    tamano_test_backtest,
)


def test_tamano_test_backtest_archivo_corto_usa_default():
    """Con un archivo de 12 meses (9 usables tras perder 3 por rezagos,
    9 + 3 = 12 < 18), se mantiene el test chico de siempre."""
    assert tamano_test_backtest(9) == 3


def test_tamano_test_backtest_justo_en_el_umbral_usa_test_grande():
    """15 meses usables + 3 perdidos por rezagos = 18 meses de archivo
    original, justo el umbral — ya debería usar el test grande (6)."""
    assert tamano_test_backtest(15) == 6


def test_tamano_test_backtest_un_mes_antes_del_umbral_usa_default():
    """14 usables + 3 = 17 meses de archivo, un mes por debajo del
    umbral de 18 — todavía debe usar el test chico (3)."""
    assert tamano_test_backtest(14) == 3


def test_elegir_campeon_por_producto_diferencia_no_significativa_usa_media_movil():
    """
    Cálculo a mano (MAE = promedio del error absoluto):

    Producto A — demanda real [10, 20]:
      Media móvil       [12, 18] -> errores [2, 2]   -> MAE = 2.0   (el más bajo)
      Regresión Lineal  [10, 25] -> errores [0, 5]   -> MAE = 2.5
      XGBoost           [15, 15] -> errores [5, 5]   -> MAE = 5.0
      -> el de menor MAE ya es "Media móvil", así que no cambia.

    Producto B — demanda real [100, 200]:
      Media móvil       [50, 50]   -> errores [50, 150] -> MAE = 100.0
      Regresión Lineal  [100, 210] -> errores [0, 10]   -> MAE = 5.0   (el más bajo)
      XGBoost           [90, 190]  -> errores [10, 10]  -> MAE = 10.0  (segundo)
      -> con solo 2 meses de backtest, la prueba de Diebold-Mariano
      (corrección Harvey-Leybourne-Newbold) da p ~= 0.61 comparando
      Regresión Lineal contra XGBoost — muy por encima de 0.05, o sea
      NO significativo, a pesar de que el MAE de Regresión Lineal es la
      mitad del de XGBoost. Con tan pocos datos, esa diferencia podría
      ser puro azar, así que el sistema prefiere el método más simple y
      robusto ("Media móvil") en vez de confiar en ese MAE más bajo.
    """
    tabla = pd.DataFrame({
        "producto_id": ["A", "A", "B", "B"],
        "demanda": [10, 20, 100, 200],
        "prediccion_media_movil": [12, 18, 50, 50],
        "prediccion_regresion_lineal": [10, 25, 100, 210],
        "prediccion_xgboost": [15, 15, 90, 190],
    })

    campeon, errores = elegir_campeon_por_producto(tabla)

    assert campeon["A"] == "Media móvil"
    assert campeon["B"] == "Media móvil"
    # MAE de cada método por producto (calculados a mano en el docstring
    # de arriba) — errores_por_producto los devuelve TODOS, no solo el
    # del campeón, para poder armar el rango de pronóstico futuro.
    assert errores.loc["A", "Media móvil"] == pytest.approx(2.0)
    assert errores.loc["A", "Regresión Lineal"] == pytest.approx(2.5)
    assert errores.loc["A", "XGBoost"] == pytest.approx(5.0)
    assert errores.loc["B", "Regresión Lineal"] == pytest.approx(5.0)


def test_elegir_campeon_por_producto_diferencia_clara_y_sostenida_si_gana():
    """Con más meses de backtest (5) y una diferencia de error grande y
    CONSISTENTE mes a mes (XGBoost siempre cerca de la demanda real,
    Media Móvil y Regresión Lineal siempre muy por encima), la prueba de
    Diebold-Mariano sí encuentra la diferencia estadísticamente
    significativa — confirma que la función no cae en "Media móvil" por
    defecto siempre, solo cuando la evidencia es débil."""
    tabla = pd.DataFrame({
        "producto_id": ["D"] * 5,
        "demanda": [100, 100, 100, 100, 100],
        "prediccion_media_movil": [150, 140, 160, 145, 155],
        "prediccion_regresion_lineal": [130, 135, 125, 132, 128],
        "prediccion_xgboost": [102, 101, 103, 99, 101],
    })
    campeon, _ = elegir_campeon_por_producto(tabla)
    assert campeon["D"] == "XGBoost"


def test_elegir_campeon_por_producto_empate_exacto_usa_media_movil():
    """Si dos métodos tienen exactamente el mismo error mes a mes (acá
    Media Móvil y Regresión Lineal, ambos con error 2 en los dos meses),
    la diferencia entre ellos tiene varianza cero — no hay evidencia de
    que uno sea mejor que el otro, así que se usa "Media móvil" (además
    de ser, por MAE, el primero en orden de COLUMNA_PREDICCION_POR_METODO)."""
    tabla = pd.DataFrame({
        "producto_id": ["C", "C"],
        "demanda": [10, 10],
        "prediccion_media_movil": [12, 12],   # error 2
        "prediccion_regresion_lineal": [8, 8],  # error 2 (empate)
        "prediccion_xgboost": [20, 20],       # error 10
    })
    campeon, _ = elegir_campeon_por_producto(tabla)
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


def test_calcular_errores_por_producto():
    """Mismo cálculo a mano que el producto A del docstring de arriba:
    MAE de Media Móvil = 2.0, Regresión Lineal = 2.5, XGBoost = 5.0."""
    tabla = pd.DataFrame({
        "producto_id": ["A", "A"],
        "demanda": [10, 20],
        "prediccion_media_movil": [12, 18],
        "prediccion_regresion_lineal": [10, 25],
        "prediccion_xgboost": [15, 15],
    })
    errores = calcular_errores_por_producto(tabla)
    assert errores.loc["A", "Media móvil"] == pytest.approx(2.0)
    assert errores.loc["A", "Regresión Lineal"] == pytest.approx(2.5)
    assert errores.loc["A", "XGBoost"] == pytest.approx(5.0)


class TestConstruirTablaPrediccionesSinFugaDeSeleccion:
    """
    construir_tabla_predicciones recibe el campeón como parámetro — no
    lo calcula con test_df. Para el producto "Z", la demanda real de
    los dos meses de test es [10, 10]: XGBoost predice exactamente
    [10, 10] (error 0) y Media Móvil predice [100, 100] (error 90). Si
    la función eligiera su propio campeón con estos mismos datos,
    elegiría XGBoost sin dudarlo — pero se le pasa a propósito un
    campeón que dice "Media móvil" para "Z" (como si así se hubiera
    decidido con el período de validación), y tiene que respetarlo.
    """

    @pytest.fixture
    def resultado(self):
        test_df = pd.DataFrame({
            "fecha": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "producto_id": ["Z", "Z"],
            "demanda": [10, 10],
            "categoria_abc": ["A", "A"],
            "variabilidad": ["Baja", "Baja"],
        })
        campeon_de_validacion = pd.Series({"Z": "Media móvil"})
        return construir_tabla_predicciones(
            test_df,
            pred_baseline_3=np.array([100, 100]),
            pred_lr=np.array([50, 50]),
            pred_xgb=np.array([10, 10]),
            campeon=campeon_de_validacion,
        )

    def test_usa_el_metodo_del_campeon_recibido_no_el_de_menor_error_en_test(self, resultado):
        resultados_prediccion, *_ = resultado
        assert (resultados_prediccion["metodo_usado"] == "Media móvil").all()
        assert (resultados_prediccion["prediccion_final"] == 100).all()

    def test_metodo_campeon_devuelto_es_el_mismo_que_se_recibio(self, resultado):
        _, _, _, metodo_campeon, _ = resultado
        assert metodo_campeon["Z"] == "Media móvil"

    def test_errores_por_producto_si_reflejan_la_realidad_del_test(self, resultado):
        """No hay fuga en REPORTAR qué tanto se equivocó cada método en
        el test (eso alimenta el rango de pronóstico futuro) — la fuga
        estaba en usar esos mismos números para DECIDIR el método."""
        _, _, _, _, errores = resultado
        assert errores.loc["Z", "XGBoost"] == pytest.approx(0.0)
        assert errores.loc["Z", "Media móvil"] == pytest.approx(90.0)


def test_construir_tabla_predicciones_producto_sin_campeon_cae_a_media_movil():
    """Un producto que no llegó al período de validación (catálogos muy
    chicos) no tiene entrada en "campeon" — tiene que caer a Media Móvil
    como respaldo seguro, no a NaN."""
    test_df = pd.DataFrame({
        "fecha": pd.to_datetime(["2024-01-01"]),
        "producto_id": ["W"],
        "demanda": [5.0],
        "categoria_abc": ["C"],
        "variabilidad": ["Baja"],
    })
    campeon_de_validacion = pd.Series({"Z": "XGBoost"})  # no incluye "W"
    resultados_prediccion, *_ = construir_tabla_predicciones(
        test_df,
        pred_baseline_3=np.array([5.0]),
        pred_lr=np.array([5.0]),
        pred_xgb=np.array([5.0]),
        campeon=campeon_de_validacion,
    )
    assert resultados_prediccion.loc[0, "metodo_usado"] == "Media móvil"
