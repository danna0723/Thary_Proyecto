"""
Pruebas de VERIFICACIÓN para calcular_rango_prediccion
(app/services/prediccion/pronostico_futuro.py) — el rango [mínimo,
máximo] que acompaña a cada predicción del pronóstico futuro, además del
valor puntual de siempre. El margen es el MAE que tuvo, en el backtest,
el método que terminó usándose para ese producto (ver
MAE_POR_METODO_COL y MARGEN_RANGO_MULTIPLICADOR en el código fuente).
"""

import math

import pytest

from app.services.prediccion.pronostico_futuro import calcular_rango_prediccion, es_baja_confianza


def test_margen_cero_da_rango_sin_ancho():
    """Si el método no se equivocó nada en el backtest (caso de juguete,
    no realista), el rango colapsa al valor puntual."""
    assert calcular_rango_prediccion(100, 0) == (100, 100)


def test_margen_positivo_da_rango_simetrico():
    """Con margen 15, el rango es [predicción - 15, predicción + 15] —
    MARGEN_RANGO_MULTIPLICADOR por defecto es 1.0, así que el margen no
    se escala."""
    minimo, maximo = calcular_rango_prediccion(100, 15)
    assert minimo == pytest.approx(85)
    assert maximo == pytest.approx(115)


def test_margen_mayor_a_la_prediccion_no_baja_de_cero():
    """La demanda no puede ser negativa: si el margen es mayor que la
    predicción misma, el mínimo se recorta en 0, no en un número
    negativo."""
    minimo, maximo = calcular_rango_prediccion(10, 25)
    assert minimo == 0
    assert maximo == pytest.approx(35)


def test_margen_nan_se_trata_como_cero():
    """Un producto que nunca llegó a tener datos de backtest (catálogos
    muy chicos) no tiene MAE calculado — NaN se trata como margen 0 (sin
    rango), en vez de fallar o inventar una incertidumbre no medida."""
    assert calcular_rango_prediccion(50, math.nan) == (50, 50)


def test_margen_none_se_trata_como_cero():
    """Mismo caso que el de arriba, pero cuando la columna de MAE ni
    siquiera existe para ese producto (segmento.get(...) devuelve None,
    no NaN)."""
    assert calcular_rango_prediccion(50, None) == (50, 50)


def test_baja_confianza_cuando_el_minimo_se_recorto_en_cero():
    """OXACILINA (caso real del catálogo clínico): predicción 16.33,
    margen 29.33 -> mínimo recortado en 0 -> baja confianza."""
    minimo, _maximo = calcular_rango_prediccion(16.33, 29.33)
    assert minimo == 0
    assert es_baja_confianza(16.33, minimo) is True


def test_no_es_baja_confianza_cuando_el_minimo_no_se_recorta():
    """OMEPRAZOL (caso real): predicción 1992, margen 650 -> el mínimo
    (1342) no llega a 0 -> no se marca como baja confianza, aunque el
    rango sea ancho en términos absolutos."""
    minimo, _maximo = calcular_rango_prediccion(1992, 650)
    assert minimo > 0
    assert es_baja_confianza(1992, minimo) is False


def test_no_es_baja_confianza_sin_margen_medido():
    """Sin margen (producto sin backtest, o con error cero medido como
    EPOPROSTENOL), el mínimo coincide con la predicción — ausencia de
    margen no es lo mismo que mucho margen, así que no se marca."""
    assert es_baja_confianza(0, 0) is False
    assert es_baja_confianza(100, 100) is False
