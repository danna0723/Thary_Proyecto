"""
Pruebas de VERIFICACIÓN (no de validación estadística) para
calcular_tabla_reorden (app/services/prediccion/reposicion.py).

Diferencia con la validación cruzada de modelos.py/validacion.py: ahí se
mide qué tan cerca estuvo una PREDICCIÓN de la demanda real (no hay una
respuesta "correcta" única, solo una mejor o peor aproximación). Acá, en
cambio, el punto de reorden, el stock de seguridad, el EOQ y los costos
son CÁLCULOS determinísticos — dado un conjunto de entradas, hay un único
resultado matemáticamente correcto. Por eso esta prueba no compara contra
datos reales: arma entradas conocidas, calcula a mano (documentado en cada
bloque de abajo) qué debería salir, y confirma que el código produce
exactamente eso. Esto es lo que en ingeniería de software se llama una
prueba con "oráculo conocido" (known-answer / oracle test).

Se usan tres productos, cada uno pensado para ejercitar una parte
distinta de la fórmula:

  A100 — stock de seguridad, punto de reorden, y el camino cuando NO hay
         datos de costo (fallback a punto_reorden - inventario_actual).
  B200 — EOQ con todos los datos de costo presentes, sin que el
         presupuesto lo limite.
  C300 — EOQ limitado por el presupuesto de capital de trabajo (el
         resultado final queda por debajo del EOQ "ideal"), y el camino
         cuando NO hay dato de inventario_actual.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.services.prediccion.reposicion import calcular_tabla_reorden


FECHA_REFERENCIA = pd.Timestamp("2026-06-01")


@pytest.fixture
def resultado():
    """Corre calcular_tabla_reorden() una sola vez con las entradas
    descritas arriba y comparte el resultado entre todas las pruebas de
    este archivo (evita repetir la construcción de los DataFrames en
    cada test)."""
    df_model = pd.DataFrame({"producto_id": ["A100", "B200", "C300"]})

    df_mensual = pd.DataFrame([
        {
            "producto_id": "A100", "fecha": FECHA_REFERENCIA,
            "lead_time_semanas": np.nan,  # sin dato -> usa el default (0.5 meses)
            "costo_pedido": np.nan, "porcentaje_costo_mantenimiento": np.nan,
            "costo_unitario": np.nan, "inventario": 40,
        },
        {
            "producto_id": "B200", "fecha": FECHA_REFERENCIA,
            "lead_time_semanas": 4.345,  # -> 4.345 / 4.345 = 1.0 mes exacto
            "costo_pedido": 50_000, "porcentaje_costo_mantenimiento": 0.2,
            "costo_unitario": 2_000, "inventario": 200,
        },
        {
            "producto_id": "C300", "fecha": FECHA_REFERENCIA,
            "lead_time_semanas": 4.345,
            "costo_pedido": 20_000, "porcentaje_costo_mantenimiento": 0.1,
            "costo_unitario": 50_000, "inventario": np.nan,  # sin dato de inventario
        },
    ])

    producto_comportamiento = pd.DataFrame([
        {
            "producto_id": "A100", "nombre_producto": "Producto A", "categoria": "TEST",
            "proveedor_principal": None, "proveedor_alterno": None,
            "categoria_abc": "A", "variabilidad": "Baja", "metodo_campeon": "Media móvil",
        },
        {
            "producto_id": "B200", "nombre_producto": "Producto B", "categoria": "TEST",
            "proveedor_principal": "Proveedor X", "proveedor_alterno": None,
            "categoria_abc": "B", "variabilidad": "Media", "metodo_campeon": "XGBoost",
        },
        {
            "producto_id": "C300", "nombre_producto": "Producto C", "categoria": "TEST",
            "proveedor_principal": None, "proveedor_alterno": None,
            "categoria_abc": "C", "variabilidad": "Alta", "metodo_campeon": "Regresión Lineal",
        },
    ])

    # 3 meses de pronóstico por producto — mismo horizonte que usa el
    # sistema real (HORIZONTE_MESES=3 en pronostico_futuro.py).
    pronostico_futuro_df = pd.DataFrame([
        {"producto_id": "A100", "prediccion_final": 90},
        {"producto_id": "A100", "prediccion_final": 100},
        {"producto_id": "A100", "prediccion_final": 110},
        {"producto_id": "B200", "prediccion_final": 100},
        {"producto_id": "B200", "prediccion_final": 100},
        {"producto_id": "B200", "prediccion_final": 100},
        {"producto_id": "C300", "prediccion_final": 100},
        {"producto_id": "C300", "prediccion_final": 100},
        {"producto_id": "C300", "prediccion_final": 100},
    ])

    # Presupuesto total tal que, repartido entre los 3 productos, dé
    # números redondos: 6.000.000 / 3 = 2.000.000 por producto.
    reorder_df, presupuesto_usado, presupuesto_por_producto = calcular_tabla_reorden(
        df_model, df_mensual, producto_comportamiento, pronostico_futuro_df,
        presupuesto_capital_trabajo=6_000_000,
    )
    return reorder_df.set_index("producto_id"), presupuesto_usado, presupuesto_por_producto


def fila(resultado, producto_id):
    reorder_df, _, _ = resultado
    return reorder_df.loc[producto_id]


def test_presupuesto_se_reparte_en_partes_iguales(resultado):
    _, presupuesto_usado, presupuesto_por_producto = resultado
    assert presupuesto_usado == 6_000_000
    assert presupuesto_por_producto == pytest.approx(2_000_000.0)


class TestA100StockDeSeguridadYPuntoDeReorden:
    """
    Cálculo a mano:
      desviación de la demanda pronosticada [90, 100, 110] (desvío
      estándar poblacional, igual que np.std):
        media = 100
        varianza = ((90-100)² + (100-100)² + (110-100)²) / 3
                 = (100 + 0 + 100) / 3 = 66.6667
        desvío = √66.6667 = 8.164966

      lead time: sin dato en el CSV -> usa el default de 0.5 meses.

      stock de seguridad = Z × desvío × √lead_time
                          = 1.65 × 8.164966 × √0.5
                          = 1.65 × 8.164966 × 0.707107
                          = 9.526279

      punto de reorden = demanda_promedio × lead_time + stock_seguridad
                        = 100 × 0.5 + 9.526279 = 59.526279
    """

    def test_stock_de_seguridad(self, resultado):
        desvio_esperado = math.sqrt(((90 - 100) ** 2 + (100 - 100) ** 2 + (110 - 100) ** 2) / 3)
        stock_seguridad_esperado = 1.65 * desvio_esperado * math.sqrt(0.5)
        assert fila(resultado, "A100")["stock_seguridad"] == pytest.approx(stock_seguridad_esperado, abs=0.01)

    def test_punto_de_reorden(self, resultado):
        punto_reorden_esperado = 100 * 0.5 + fila(resultado, "A100")["stock_seguridad"]
        assert fila(resultado, "A100")["punto_reorden"] == pytest.approx(punto_reorden_esperado, abs=0.01)

    def test_sin_datos_de_costo_no_hay_eoq_y_cae_al_metodo_alternativo(self, resultado):
        """Sin costo_unitario/costo_pedido/porcentaje_costo_mantenimiento
        en el CSV, el EOQ no se puede calcular — la cantidad sugerida
        tiene que caer a (punto_reorden - inventario_actual), no quedar
        vacía."""
        f = fila(resultado, "A100")
        assert f["eoq"] is None
        assert f["costo_estimado_pedido"] is None
        cantidad_esperada = f["punto_reorden"] - 40
        assert f["cantidad_sugerida_pedido"] == pytest.approx(cantidad_esperada, abs=0.01)

    def test_alerta_de_pedido_cuando_inventario_esta_por_debajo_del_punto_de_reorden(self, resultado):
        f = fila(resultado, "A100")
        assert 40 <= f["punto_reorden"]  # confirma la premisa del caso de prueba
        assert f["ordenar"] == "SI"
        assert f["dias_para_pedido"] == 0
        assert f["fecha_estimada_pedido"] == "2026-06-01"


class TestB200EOQSinLimiteDePresupuesto:
    """
    Cálculo a mano:
      demanda_promedio = 100 (pronóstico constante), lead_time = 4.345
      semanas / 4.345 = 1.0 mes exacto -> stock_seguridad = 0 (sin
      variabilidad) -> punto_reorden = 100 × 1.0 = 100.

      demanda anual estimada = 100 × 12 = 1.200
      costo de mantenimiento unitario = costo_unitario × porcentaje
                                       = 2.000 × 0.2 = 400
      EOQ = √(2 × D_anual × costo_pedido / costo_mantenimiento)
          = √(2 × 1.200 × 50.000 / 400)
          = √300.000 = 547.7225...

      límite por presupuesto = presupuesto_por_producto / costo_unitario
                              = 2.000.000 / 2.000 = 1.000 unidades
      Como 1.000 > EOQ (547.72), el presupuesto NO limita este pedido.
    """

    def test_eoq(self, resultado):
        eoq_esperado = math.sqrt((2 * 1200 * 50_000) / 400)
        assert eoq_esperado == pytest.approx(547.72, abs=0.01)
        assert fila(resultado, "B200")["eoq"] == pytest.approx(round(eoq_esperado), abs=1)

    def test_cantidad_sugerida_es_el_eoq_completo_sin_recortar(self, resultado):
        eoq_esperado = math.sqrt((2 * 1200 * 50_000) / 400)
        f = fila(resultado, "B200")
        assert f["limite_presupuesto_unidades"] == 1000
        assert f["cantidad_sugerida_pedido"] == pytest.approx(eoq_esperado, abs=0.01)

    def test_costo_estimado_del_pedido(self, resultado):
        # Ojo: el costo se calcula en producción sobre la cantidad SIN
        # redondear (547.7225...), no sobre el valor ya redondeado a 2
        # decimales que se guarda en "cantidad_sugerida_pedido" — hay
        # que recalcular el EOQ exacto acá para que el oráculo sea fiel.
        eoq_exacto = math.sqrt((2 * 1200 * 50_000) / 400)
        costo_esperado = eoq_exacto * 2_000
        assert fila(resultado, "B200")["costo_estimado_pedido"] == pytest.approx(costo_esperado, abs=0.01)

    def test_valor_del_inventario_actual(self, resultado):
        # 200 unidades en stock × $2.000 costo unitario
        assert fila(resultado, "B200")["valor_inventario_actual"] == pytest.approx(400_000.0)

    def test_no_hace_falta_pedir_todavia(self, resultado):
        f = fila(resultado, "B200")
        assert f["ordenar"] == "NO"  # inventario (200) > punto de reorden (100)
        # días para llegar al punto de reorden: (200-100) / (100/30.44) = 30.44 días
        assert f["dias_para_pedido"] == pytest.approx(30, abs=1)


class TestC300EOQLimitadoPorElPresupuesto:
    """
    Mismo tipo de cálculo que B200, pero con un producto más caro
    (costo_unitario más alto) para forzar que el presupuesto sí termine
    acotando la cantidad sugerida:

      costo de mantenimiento unitario = 50.000 × 0.1 = 5.000
      EOQ = √(2 × 1.200 × 20.000 / 5.000) = √9.600 = 97.9796...

      límite por presupuesto = 2.000.000 / 50.000 = 40 unidades

      Como el límite (40) es MENOR que el EOQ (97.98), la cantidad
      sugerida final tiene que quedar en 40, no en el EOQ "ideal".
    """

    def test_eoq_ideal_es_mayor_al_limite_de_presupuesto(self, resultado):
        eoq_esperado = math.sqrt((2 * 1200 * 20_000) / 5_000)
        assert eoq_esperado == pytest.approx(97.98, abs=0.01)
        assert fila(resultado, "C300")["limite_presupuesto_unidades"] == 40
        assert eoq_esperado > fila(resultado, "C300")["limite_presupuesto_unidades"]

    def test_cantidad_sugerida_queda_acotada_por_el_presupuesto(self, resultado):
        f = fila(resultado, "C300")
        assert f["cantidad_sugerida_pedido"] == pytest.approx(40, abs=0.01)
        assert f["costo_estimado_pedido"] == pytest.approx(40 * 50_000, abs=0.01)

    def test_sin_dato_de_inventario_no_hay_alerta_ni_fecha_estimada(self, resultado):
        """Sin inventario_actual en el CSV, el sistema no puede saber si
        ya hay que pedir — tiene que abstenerse (ordenar="NO", sin fecha
        estimada) en vez de asumir un valor."""
        f = fila(resultado, "C300")
        assert f["ordenar"] == "NO"
        assert f["fecha_estimada_pedido"] is None
        assert f["dias_para_pedido"] is None
        assert f["valor_inventario_actual"] is None
