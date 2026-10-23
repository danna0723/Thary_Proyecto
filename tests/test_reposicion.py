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
  D400 — EXTENSIÓN PROPIA (corrección): sigma_demanda tiene que ser el
         MAE de backtest del método campeón (ya calculado en
         producto_comportamiento), no el desvío de los pocos valores
         pronosticados entre sí — ver el comentario largo en
         reposicion.py. Caso real del catálogo clínico (OXACILINA): esos
         valores pronosticados tienen muy poca dispersión aunque el
         modelo se equivoque bastante en el backtest.
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
            "costo_unitario": 2_000, "inventario": 200, "precio": 2_800,
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


def test_presupuesto_usado_y_referencia_por_producto(resultado):
    """El presupuesto total se devuelve tal cual. La cifra por producto
    (partes iguales) ya NO limita cada compra: solo se informa como
    referencia (ver la política al inicio de reposicion.py)."""
    reorder_df, presupuesto_usado, presupuesto_por_producto = resultado
    assert presupuesto_usado == 6_000_000
    assert presupuesto_por_producto == pytest.approx(2_000_000.0)
    costos = reorder_df["costo_estimado_pedido"].dropna()
    assert costos.sum() <= presupuesto_usado


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

    def test_punto_de_reorden_lognormal(self, resultado):
        """
        Mismo caso (media_lt=50, desvío_lt=8.164966*√0.5=5.773503), pero
        asumiendo que la demanda del lead time es log-normal en vez de
        normal (moment matching, ver comentario en reposicion.py):

          cv_lt = 5.773503 / 50 = 0.115470
          sigma_log = √ln(1 + cv_lt²) = √ln(1.013333) = 0.115088
          mu_log = ln(50) - sigma_log²/2 = 3.912023 - 0.006623 = 3.905400
          punto_reorden_lognormal = exp(mu_log + 1.65 × sigma_log)
                                   = exp(4.095296) ≈ 60.006
        """
        desvio_esperado = math.sqrt(((90 - 100) ** 2 + (100 - 100) ** 2 + (110 - 100) ** 2) / 3)
        media_lt = 100 * 0.5
        desviacion_lt = desvio_esperado * math.sqrt(0.5)
        cv_lt = desviacion_lt / media_lt
        sigma_log = math.sqrt(math.log(1 + cv_lt ** 2))
        mu_log = math.log(media_lt) - (sigma_log ** 2) / 2
        punto_reorden_lognormal_esperado = math.exp(mu_log + 1.65 * sigma_log)
        assert fila(resultado, "A100")["punto_reorden_lognormal"] == pytest.approx(punto_reorden_lognormal_esperado, abs=0.01)
        # El supuesto log-normal da, para este producto, un punto de
        # reorden MÁS ALTO que el normal — justamente porque captura la
        # cola derecha que la normal simétrica no ve.
        assert fila(resultado, "A100")["punto_reorden_lognormal"] > fila(resultado, "A100")["punto_reorden"]

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
        """B200 se asigna primero (sin alerta, pero es el producto con
        menos días para pedir entre los que tienen costo). Con 6.000.000
        disponibles, lo que queda al decidirlo alcanza para 3.000 unidades
        a $2.000 (6.000.000 / 2.000), así que el EOQ no se recorta."""
        eoq_esperado = math.sqrt((2 * 1200 * 50_000) / 400)
        f = fila(resultado, "B200")
        assert f["limite_presupuesto_unidades"] == 3000
        assert f["cantidad_sugerida_pedido"] == pytest.approx(eoq_esperado, abs=0.01)
        assert f["estado_presupuesto"] == "completo"
        assert f["cantidad_sin_cubrir"] == pytest.approx(0.0, abs=0.01)

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

    def test_punto_de_reorden_lognormal_sin_variabilidad_coincide_con_el_normal(self, resultado):
        """Pronóstico constante (100,100,100) -> desviación 0 -> sin
        variabilidad no hace falta stock de seguridad bajo NINGÚN
        supuesto de distribución — log-normal y normal tienen que dar
        exactamente el mismo punto de reorden (100 × 1.0 mes = 100)."""
        f = fila(resultado, "B200")
        assert f["punto_reorden_lognormal"] == pytest.approx(100.0)
        assert f["punto_reorden_lognormal"] == pytest.approx(f["punto_reorden"])
        assert f["stock_seguridad_lognormal"] == pytest.approx(0.0)

    def test_no_hace_falta_pedir_todavia(self, resultado):
        f = fila(resultado, "B200")
        assert f["ordenar"] == "NO"  # inventario (200) > punto de reorden (100)
        # días para llegar al punto de reorden: (200-100) / (100/30.44) = 30.44 días
        assert f["dias_para_pedido"] == pytest.approx(30, abs=1)

    def test_costo_de_ruptura_estimado(self, resultado):
        """Con precio=2.800 y costo_unitario=2.000, margen = 800/unidad.
        demanda_diaria × lead_time_dias se simplifica exactamente a
        demanda_promedio × lead_time_meses (30.44 se cancela), así que:
        costo_ruptura = 800 × 100 × 1.0 = 80.000 — el margen que se
        perdería si el producto se agota durante todo el lead time."""
        f = fila(resultado, "B200")
        assert f["margen_unitario"] == pytest.approx(800.0)
        assert f["costo_ruptura_estimado"] == pytest.approx(80_000.0, abs=1)


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

    def test_con_presupuesto_suficiente_el_eoq_no_se_recorta(self, resultado):
        """Con 6.000.000 de presupuesto total, después de B200 (1.095.445)
        quedan 4.904.555 = 98 unidades a $50.000. El EOQ de C300 (97.98)
        cabe, así que la cantidad es el EOQ completo."""
        eoq_esperado = math.sqrt((2 * 1200 * 20_000) / 5_000)
        assert eoq_esperado == pytest.approx(97.98, abs=0.01)
        f = fila(resultado, "C300")
        assert f["limite_presupuesto_unidades"] == 98
        assert f["cantidad_sugerida_pedido"] == pytest.approx(eoq_esperado, abs=0.01)
        assert f["cantidad_sin_cubrir"] == pytest.approx(0.0, abs=0.01)
        assert f["estado_presupuesto"] == "completo"
        assert f["costo_estimado_pedido"] == pytest.approx(eoq_esperado * 50_000, abs=0.01)

    def test_sin_dato_de_inventario_no_hay_alerta_ni_fecha_estimada(self, resultado):
        """Sin inventario_actual en el CSV, el sistema no puede saber si
        ya hay que pedir — tiene que abstenerse (ordenar="NO", sin fecha
        estimada) en vez de asumir un valor."""
        f = fila(resultado, "C300")
        assert f["ordenar"] == "NO"
        assert f["fecha_estimada_pedido"] is None
        assert f["dias_para_pedido"] is None
        assert f["valor_inventario_actual"] is None

    def test_sin_precio_no_hay_costo_de_ruptura(self, resultado):
        """C300 tiene costo_unitario pero no precio — sin margen conocido
        no se puede estimar el costo de ruptura, tiene que quedar en
        None (no en 0, que implicaría "sin costo de ruptura")."""
        f = fila(resultado, "C300")
        assert f["margen_unitario"] is None
        assert f["costo_ruptura_estimado"] is None


class TestD400SigmaUsaElMAEDeBacktestNoLaDispersionDelPronostico:
    """
    Caso real del catálogo clínico (OXACILINA): 3 meses pronosticados
    con muy poca dispersión entre sí ([16.33, 5.44, 7.26], desvío ≈
    4.76) aunque el método campeón (Media Móvil) se haya equivocado en
    promedio 29.33 unidades con este producto en el backtest — mucho
    más. Con "mae_media_movil" presente en producto_comportamiento,
    sigma_demanda tiene que ser 29.33 (el MAE), no 4.76 (el desvío de
    esos 3 números) — antes de la corrección, era este último.

    stock_seguridad = Z × MAE × √L = 1.65 × 29.33 × √0.5 ≈ 34.21
    (muy por encima de lo que daría el desvío de los pronósticos:
    1.65 × 4.76 × √0.5 ≈ 5.55)
    """

    @pytest.fixture
    def resultado(self):
        df_model = pd.DataFrame({"producto_id": ["D400"]})
        df_mensual = pd.DataFrame([{
            "producto_id": "D400", "fecha": FECHA_REFERENCIA,
            "lead_time_semanas": np.nan,  # sin dato -> lead time default 0.5 meses
            "costo_pedido": np.nan, "porcentaje_costo_mantenimiento": np.nan,
            "costo_unitario": np.nan, "inventario": 100,
        }])
        producto_comportamiento = pd.DataFrame([{
            "producto_id": "D400", "nombre_producto": "Producto D", "categoria": "TEST",
            "proveedor_principal": None, "proveedor_alterno": None,
            "categoria_abc": "C", "variabilidad": "Alta", "metodo_campeon": "Media móvil",
            "mae_media_movil": 29.33, "mae_regresion_lineal": 38.36, "mae_xgboost": 16.64,
        }])
        pronostico_futuro_df = pd.DataFrame([
            {"producto_id": "D400", "prediccion_final": 16.33},
            {"producto_id": "D400", "prediccion_final": 5.44},
            {"producto_id": "D400", "prediccion_final": 7.26},
        ])
        reorder_df, presupuesto_usado, presupuesto_por_producto = calcular_tabla_reorden(
            df_model, df_mensual, producto_comportamiento, pronostico_futuro_df,
            presupuesto_capital_trabajo=6_000_000,
        )
        return reorder_df.set_index("producto_id"), presupuesto_usado, presupuesto_por_producto

    def test_stock_de_seguridad_usa_el_mae_del_metodo_campeon(self, resultado):
        desvio_pronosticos = np.std([16.33, 5.44, 7.26])
        stock_seguridad_con_mae = 1.65 * 29.33 * math.sqrt(0.5)
        stock_seguridad_con_desvio_viejo = 1.65 * desvio_pronosticos * math.sqrt(0.5)

        f = fila(resultado, "D400")
        assert f["stock_seguridad"] == pytest.approx(stock_seguridad_con_mae, abs=0.01)
        # El criterio viejo (desvío de los pronósticos) habría dado un
        # número bastante menor — confirma que de verdad cambió de uno
        # a otro, no que ambos coincidan por casualidad en este caso.
        assert f["stock_seguridad"] > stock_seguridad_con_desvio_viejo * 3

    def test_punto_de_reorden_lognormal_tambien_usa_el_mae(self, resultado):
        """La versión log-normal reutiliza la misma sigma_demanda ya
        corregida — no tiene su propio cálculo aparte."""
        media_lt = np.mean([16.33, 5.44, 7.26]) * 0.5
        desviacion_lt = 29.33 * math.sqrt(0.5)
        cv_lt = desviacion_lt / media_lt
        sigma_log = math.sqrt(math.log(1 + cv_lt ** 2))
        mu_log = math.log(media_lt) - (sigma_log ** 2) / 2
        punto_reorden_lognormal_esperado = math.exp(mu_log + 1.65 * sigma_log)

        f = fila(resultado, "D400")
        assert f["punto_reorden_lognormal"] == pytest.approx(punto_reorden_lognormal_esperado, abs=0.01)


# ============================================================
# POLÍTICA DE REPOSICIÓN (corrección P1-B)
# Casos de la tabla de validación propuesta en el informe: borde de
# alerta, prioridad y presupuesto insuficiente, cobertura durante la
# entrega, y fuente del error usado para el stock de seguridad.
# ============================================================

def _calcular(productos, presupuesto):
    """
    productos: lista de dicts con: producto, inventario, pronostico (lista
    de 3 valores), y opcionalmente costo_unitario, costo_pedido,
    porcentaje, lead_time_semanas, mae (MAE de backtest del método campeón).
    Devuelve el reorder_df indexado por producto_id.
    """
    df_model = pd.DataFrame({"producto_id": [p["producto"] for p in productos]})
    df_mensual = pd.DataFrame([{
        "producto_id": p["producto"], "fecha": FECHA_REFERENCIA,
        "lead_time_semanas": p.get("lead_time_semanas", np.nan),
        "costo_pedido": p.get("costo_pedido", np.nan),
        "porcentaje_costo_mantenimiento": p.get("porcentaje", np.nan),
        "costo_unitario": p.get("costo_unitario", np.nan),
        "inventario": p["inventario"],
    } for p in productos])
    comportamiento_filas = []
    for p in productos:
        fila = {
            "producto_id": p["producto"], "nombre_producto": p["producto"], "categoria": "TEST",
            "proveedor_principal": None, "proveedor_alterno": None,
            "categoria_abc": p.get("abc", "A"), "variabilidad": "Baja", "metodo_campeon": "Media móvil",
        }
        if p.get("mae") is not None:
            fila["mae_media_movil"] = p["mae"]
        comportamiento_filas.append(fila)
    pronostico = pd.DataFrame([
        {"producto_id": p["producto"], "prediccion_final": v}
        for p in productos for v in p["pronostico"]
    ])
    reorder_df, _, _ = calcular_tabla_reorden(
        df_model, df_mensual, pd.DataFrame(comportamiento_filas), pronostico,
        presupuesto_capital_trabajo=presupuesto,
    )
    return reorder_df.set_index("producto_id")


class TestBordeDeAlerta:
    """
    Sin EOQ (sin datos de costo). Pronóstico constante 100 -> desvío 0 ->
    stock de seguridad 0. Lead time por defecto 0.5 meses -> punto de
    reorden = 100 × 0.5 = 50.
    """

    def test_inventario_exactamente_en_el_punto_de_reorden_pide_un_mes(self):
        df = _calcular([{"producto": "E500", "inventario": 50, "pronostico": [100, 100, 100]}], 1_000_000)
        f = df.loc["E500"]
        assert f["punto_reorden"] == pytest.approx(50.0)
        assert f["ordenar"] == "SI"
        # Antes salía cantidad 0 con alerta activa; ahora un mes de demanda.
        assert f["cantidad_sugerida_pedido"] == pytest.approx(100.0)
        assert f["regla_cantidad"] == "un mes de demanda (inventario en el punto de reorden)"

    def test_inventario_sobre_el_punto_de_reorden_no_pide_nada(self):
        df = _calcular([{"producto": "E500", "inventario": 51, "pronostico": [100, 100, 100]}], 1_000_000)
        f = df.loc["E500"]
        assert f["ordenar"] == "NO"
        assert f["cantidad_sugerida_pedido"] == pytest.approx(0.0)
        assert f["regla_cantidad"] == "sin necesidad de pedido"


class TestPrioridadYPresupuestoInsuficiente:
    """
    URG: alerta (inventario 0 <= punto de reorden 50), costo 1.000, sin EOQ
         -> cantidad ideal = 50 - 0 = 50 unidades ($50.000).
    NOURG: como B200 (EOQ 547.72, costo 2.000, inventario 200, sin alerta).
    """

    def _productos(self):
        urg = {"producto": "URG", "inventario": 0, "pronostico": [100, 100, 100], "costo_unitario": 1_000}
        nourg = {
            "producto": "NOURG", "inventario": 200, "pronostico": [100, 100, 100],
            "lead_time_semanas": 4.345, "costo_pedido": 50_000, "porcentaje": 0.2, "costo_unitario": 2_000,
        }
        return [nourg, urg]  # el orden de entrada no debe importar

    def test_los_productos_con_alerta_se_cubren_primero(self):
        # Presupuesto 1.100.000: si NOURG se asignara primero (EOQ = 1.095.445),
        # a URG le quedarían solo 4.555 = 4 unidades. Con prioridad, URG recibe
        # sus 50 unidades completas y NOURG se recorta con el resto.
        df = _calcular(self._productos(), 1_100_000)
        urg = df.loc["URG"]
        nourg = df.loc["NOURG"]
        assert urg["cantidad_sugerida_pedido"] == pytest.approx(50.0)
        assert urg["estado_presupuesto"] == "completo"
        eoq = math.sqrt((2 * 1200 * 50_000) / 400)
        assert nourg["cantidad_sugerida_pedido"] == pytest.approx(525.0)  # 1.050.000 / 2.000
        assert nourg["cantidad_ideal_pedido"] == pytest.approx(eoq, abs=0.01)
        assert nourg["cantidad_sin_cubrir"] == pytest.approx(eoq - 525, abs=0.01)
        assert nourg["estado_presupuesto"] == "parcial"

    def test_lo_que_no_alcanza_queda_marcado_como_sin_cubrir(self):
        # Con 50.000 de presupuesto, URG consume exactamente sus 50.000 y
        # no queda dinero para NOURG (que necesita 2.000 por unidad).
        df = _calcular(self._productos(), 50_000)
        nourg = df.loc["NOURG"]
        assert nourg["cantidad_sugerida_pedido"] == pytest.approx(0.0)
        assert nourg["estado_presupuesto"] == "sin presupuesto"
        assert nourg["costo_estimado_pedido"] == pytest.approx(0.0)
        assert nourg["cantidad_sin_cubrir"] == pytest.approx(
            math.sqrt((2 * 1200 * 50_000) / 400), abs=0.01
        )

    def test_el_gasto_total_nunca_supera_el_presupuesto(self):
        for presupuesto in [50_000, 500_000, 1_100_000, 6_000_000]:
            df = _calcular(self._productos(), presupuesto)
            assert df["costo_estimado_pedido"].dropna().sum() <= presupuesto + 1e-6


class TestCoberturaDuranteYDespuesDeLaEntrega:
    def test_cobertura_con_el_pedido_recomendado(self):
        # Inventario 200, demanda 100/mes, lead time 1 mes -> al llegar el
        # pedido quedan 100 unidades. Con EOQ 547.72 la cobertura es
        # (100 + 547.72) / 100 = 6.48 meses.
        df = _calcular([{
            "producto": "B200", "inventario": 200, "pronostico": [100, 100, 100],
            "lead_time_semanas": 4.345, "costo_pedido": 50_000, "porcentaje": 0.2, "costo_unitario": 2_000,
        }], 6_000_000)
        f = df.loc["B200"]
        assert f["stock_al_llegar_pedido"] == pytest.approx(100.0)
        assert f["riesgo_ruptura_durante_entrega"] == "NO"
        assert f["cobertura_despues_entrega_meses"] == pytest.approx(
            (100 + math.sqrt((2 * 1200 * 50_000) / 400)) / 100, abs=0.01
        )

    def test_riesgo_de_ruptura_si_el_stock_no_alcanza_para_la_entrega(self):
        # Inventario 20, demanda 100/mes, lead time 1 mes -> faltan 80 unidades
        # antes de que llegue el pedido.
        df = _calcular([{
            "producto": "R1", "inventario": 20, "pronostico": [100, 100, 100],
            "lead_time_semanas": 4.345, "costo_unitario": 1_000,
        }], 6_000_000)
        f = df.loc["R1"]
        assert f["stock_al_llegar_pedido"] == pytest.approx(-80.0)
        assert f["riesgo_ruptura_durante_entrega"] == "SI"


class TestFuenteDelErrorDelStockDeSeguridad:
    def test_sin_backtest_se_declara_el_desvio_del_pronostico(self):
        df = _calcular([{"producto": "E500", "inventario": 50, "pronostico": [100, 110, 90]}], 1_000_000)
        assert df.loc["E500"]["fuente_sigma"] == "desvío del pronóstico (sin backtest)"

    def test_con_backtest_se_declara_el_mae(self):
        df = _calcular([{"producto": "E500", "inventario": 50, "pronostico": [100, 110, 90], "mae": 29.33}], 1_000_000)
        assert df.loc["E500"]["fuente_sigma"] == "MAE de backtest"
        assert df.loc["E500"]["stock_seguridad"] == pytest.approx(1.65 * 29.33 * math.sqrt(0.5), abs=0.01)


class TestOrdenDePrioridadDelPresupuesto:
    def test_a_igual_urgencia_gana_la_clase_abc_mas_alta(self):
        # Ambos con alerta y sin riesgo de ruptura ni días distintos; 50.000 de
        # presupuesto alcanza solo para 50 unidades a $1.000.
        df = _calcular([
            {"producto": "X", "inventario": 0, "pronostico": [100, 100, 100], "costo_unitario": 1_000, "abc": "C"},
            {"producto": "Y", "inventario": 0, "pronostico": [100, 100, 100], "costo_unitario": 1_000, "abc": "A"},
        ], 50_000)
        assert df.loc["Y"]["cantidad_sugerida_pedido"] == pytest.approx(50.0)
        assert df.loc["X"]["cantidad_sugerida_pedido"] == pytest.approx(0.0)
        assert df.loc["X"]["estado_presupuesto"] == "sin presupuesto"

    def test_el_riesgo_de_ruptura_antes_de_la_entrega_gana_al_orden_abc(self):
        # Demanda 100/mes, lead time 1 mes: al llegar el pedido, el stock
        # restante es inventario - 100.
        # W: inventario 20 -> faltan 80 antes de la entrega (riesgo SI), clase C.
        # V: inventario 100 -> llega justo (queda 0, riesgo NO), clase A.
        # Con 30.000 solo alcanza para 30 unidades: primero W por riesgo.
        df = _calcular([
            {"producto": "V", "inventario": 100, "pronostico": [100, 100, 100], "costo_unitario": 1_000, "abc": "A", "lead_time_semanas": 4.345},
            {"producto": "W", "inventario": 20, "pronostico": [100, 100, 100], "costo_unitario": 1_000, "abc": "C", "lead_time_semanas": 4.345},
        ], 30_000)
        assert df.loc["W"]["riesgo_ruptura_durante_entrega"] == "SI"
        assert df.loc["V"]["riesgo_ruptura_durante_entrega"] == "NO"
        assert df.loc["W"]["cantidad_sugerida_pedido"] == pytest.approx(30.0)
        assert df.loc["V"]["cantidad_sugerida_pedido"] == pytest.approx(0.0)
