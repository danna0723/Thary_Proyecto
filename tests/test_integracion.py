"""
Pruebas de INTEGRACIÓN para el pipeline completo de Thary
(app/services/sistema_prediccion.py::ejecutar_sistema).

A diferencia de los demás archivos de tests (unitarios: cada uno prueba
una función aislada, dentro de un mismo módulo, con entradas armadas a
mano), acá se corre el pipeline de punta a punta — desde el CSV crudo
hasta el resultado final — con un archivo sintético que sí pasa por
las 9 etapas: carga_datos, segmentacion, features, modelos, validacion,
pronostico_futuro, reposicion, graficas y exportacion.

El objetivo no es volver a verificar la fórmula de cada etapa (ya está
cubierto por los tests unitarios) sino que las etapas se pasen bien los
datos entre sí: que el producto_id de uno aparezca donde tiene que
aparecer en el siguiente, que el método que una etapa dice haber usado
sea el que otra etapa realmente usó, que los archivos que se dice haber
generado existan de verdad en disco, etc. — justo lo que los tests
unitarios, al probar cada función por separado, no pueden detectar.

El archivo sintético tiene 3 productos pensados para cada uno ejercitar
un caso distinto:
  P1 — historial completo (12 meses) y todas las columnas opcionales
       (precio, inventario, costo, lead time) con datos. Ejercita EOQ,
       costo de ruptura y el camino con todos los datos presentes.
  P2 — historial completo pero con demanda errática (varios meses en
       cero), para que la clasificación de variabilidad y el método
       elegido sean distintos a los de P1.
  P3 — solo 2 meses de historial, sin columnas opcionales. No alcanza
       para calcular los rezagos (hacen falta al menos 3), así que
       tiene que desaparecer de ahí en adelante: no debe aparecer en el
       pronóstico futuro ni en la tabla de reposición, aunque sí en la
       segmentación (que no depende de los rezagos).
"""

import pandas as pd
import pytest

from app.services.sistema_prediccion import ejecutar_sistema

MESES = pd.date_range("2023-01-01", periods=12, freq="MS")

DEMANDA_P1 = [50, 55, 48, 52, 51, 49, 53, 50, 47, 54, 52, 50]
DEMANDA_P2 = [0, 30, 0, 0, 45, 5, 0, 60, 0, 10, 0, 25]


def _armar_csv(ruta):
    filas = []
    for mes, demanda in zip(MESES, DEMANDA_P1):
        filas.append({
            "Fecha": mes, "Codigo": "P1", "Cantidad": demanda,
            "Precio": 100, "Inventario": 500, "Costo_Unitario": 60,
            "Lead_time": 2, "Costo_Pedido": 100_000,
            "Porcentaje_Costo_Mantenimiento": 0.2,
        })
    for mes, demanda in zip(MESES, DEMANDA_P2):
        filas.append({
            "Fecha": mes, "Codigo": "P2", "Cantidad": demanda,
            "Precio": 50, "Inventario": 80, "Costo_Unitario": 30,
            "Lead_time": 3, "Costo_Pedido": 50_000,
            "Porcentaje_Costo_Mantenimiento": 0.25,
        })
    for mes, demanda in zip(MESES[:2], [20, 22]):
        filas.append({"Fecha": mes, "Codigo": "P3", "Cantidad": demanda})

    df = pd.DataFrame(filas)
    df.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="module")
def resultado(tmp_path_factory):
    directorio = tmp_path_factory.mktemp("integracion")
    ruta_csv = _armar_csv(directorio / "catalogo.csv")
    return ejecutar_sistema(str(ruta_csv), carpeta_resultados=str(directorio / "resultados"))


METODOS_VALIDOS = {"Media móvil", "Regresión Lineal", "XGBoost"}
COLUMNA_POR_METODO = {
    "Media móvil": "prediccion_baseline",
    "Regresión Lineal": "prediccion_regresion_lineal",
    "XGBoost": "prediccion_xgboost",
}
COLUMNA_POR_METODO_BACKTEST = {
    "Media móvil": "prediccion_media_movil",
    "Regresión Lineal": "prediccion_regresion_lineal",
    "XGBoost": "prediccion_xgboost",
}


def test_el_pipeline_completo_corre_sin_errores(resultado):
    """El solo hecho de llegar hasta acá sin excepción ya prueba que las
    9 etapas se conectaron correctamente con datos reales de punta a
    punta — carga, segmentación, features, modelos, validación,
    pronóstico futuro, reposición, gráficas y exportación."""
    assert resultado["mape_final"] is not None
    assert resultado["wape_final"] is not None


def test_segmentacion_tiene_los_tres_productos(resultado):
    """segmentacion.py no depende de los rezagos, así que P3 (solo 2
    meses) sí tiene que aparecer acá, a diferencia de las etapas
    siguientes."""
    productos = {s["producto_id"] for s in resultado["segmentacion"]}
    assert productos == {"P1", "P2", "P3"}


def test_p3_no_llega_al_pronostico_futuro_ni_a_reposicion(resultado):
    """P3 tiene menos de 3 meses de historial: no alcanza para calcular
    lag_3, así que generar_pronostico_futuro() tiene que excluirlo — y,
    en cadena, calcular_tabla_reorden() también, porque no tiene de
    dónde sacar su demanda promedio pronosticada."""
    productos_pronostico = {p["producto_id"] for p in resultado["pronostico_futuro"]}
    productos_reorder = {r["producto_id"] for r in resultado["reorder"]}
    assert "P3" not in productos_pronostico
    assert "P3" not in productos_reorder
    assert {"P1", "P2"} <= productos_pronostico
    assert {"P1", "P2"} <= productos_reorder


def test_metodo_campeon_es_siempre_uno_de_los_tres_validos(resultado):
    """Si por algún motivo una etapa guardara mal el nombre del método
    (un typo, un valor de otra columna por error), esto lo detecta."""
    for fila in resultado["segmentacion"]:
        assert fila["metodo_campeon"] in METODOS_VALIDOS


def test_prediccion_final_del_backtest_coincide_con_la_columna_del_metodo_usado(resultado):
    """Prueba de integración real (no la unitaria de test_modelos.py,
    que arma la tabla a mano): sobre los datos que salieron de
    verdad del pipeline completo, prediccion_final tiene que coincidir
    con la predicción del método que esa misma fila dice haber usado."""
    for fila in resultado["predicciones"]:
        columna = COLUMNA_POR_METODO_BACKTEST[fila["metodo_usado"]]
        assert fila["prediccion_final"] == pytest.approx(fila[columna], abs=0.01)


def test_prediccion_final_del_pronostico_futuro_coincide_con_la_columna_del_metodo_usado(resultado):
    for fila in resultado["pronostico_futuro"]:
        columna = COLUMNA_POR_METODO[fila["metodo_usado"]]
        assert fila["prediccion_final"] == pytest.approx(fila[columna], abs=0.01)


def test_el_rango_siempre_contiene_a_la_prediccion_final(resultado):
    """prediccion_min <= prediccion_final <= prediccion_max en cada fila
    del pronóstico futuro — si esto fallara, significaría que el margen
    se está aplicando con el signo invertido o sobre el valor
    equivocado en algún punto de la cadena."""
    for fila in resultado["pronostico_futuro"]:
        assert fila["prediccion_min"] <= fila["prediccion_final"] + 0.01
        assert fila["prediccion_final"] <= fila["prediccion_max"] + 0.01
        assert fila["prediccion_min"] >= 0


def test_reposicion_tiene_valores_no_negativos(resultado):
    for fila in resultado["reorder"]:
        assert fila["stock_seguridad"] >= 0
        assert fila["punto_reorden"] >= 0


def test_p1_con_todos_los_datos_de_costo_calcula_eoq(resultado):
    """P1 tiene costo_unitario, costo_pedido y porcentaje de
    mantenimiento en todos sus meses — con eso, calcular_tabla_reorden
    tiene que poder calcular el EOQ (no caer al método alternativo)."""
    fila_p1 = next(r for r in resultado["reorder"] if r["producto_id"] == "P1")
    assert fila_p1["eoq"] is not None
    assert fila_p1["costo_estimado_pedido"] is not None


def test_sin_fuga_de_seleccion_con_doce_meses_de_historial(resultado):
    """Con 12 meses de archivo (9 utilizables), alcanza para separar
    entrenamiento, validación y test — no debería caer al respaldo con
    fuga."""
    assert resultado["seleccion_metodo_con_fuga"] is False


def test_los_archivos_que_dice_haber_generado_existen_de_verdad(resultado):
    """graficas.py y exportacion.py devuelven rutas de archivos — acá se
    confirma que esos archivos realmente se escribieron en disco, no
    solo que la función no lanzó una excepción."""
    import os
    rutas = resultado["archivos"]
    assert os.path.exists(rutas["predicciones"])
    assert os.path.exists(rutas["reorder"])
    assert os.path.exists(rutas["segmentacion"])
    assert os.path.exists(rutas["importancia"])


def test_el_pipeline_es_determinista(tmp_path):
    """Correr el pipeline dos veces con el mismo archivo tiene que dar
    exactamente el mismo resultado numérico — los modelos usan
    random_state fijo, así que cualquier diferencia acá señalaría una
    fuente de no-determinismo real (por ejemplo, un orden de filas que
    no quedó fijo en algún groupby)."""
    ruta_csv = _armar_csv(tmp_path / "catalogo.csv")
    r1 = ejecutar_sistema(str(ruta_csv), carpeta_resultados=str(tmp_path / "resultados_1"))
    r2 = ejecutar_sistema(str(ruta_csv), carpeta_resultados=str(tmp_path / "resultados_2"))
    assert r1["mape_final"] == r2["mape_final"]
    assert r1["wape_final"] == r2["wape_final"]
    assert r1["pronostico_futuro"] == r2["pronostico_futuro"]
