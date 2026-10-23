"""
Pruebas de la comunicación de incertidumbre (corrección D del informe).

Qué protege:
- El margen del rango se distingue en tres estados: medido, medido en cero
  y no disponible. Un producto sin backtest no puede verse como confiable.
- Las corridas guardadas antes de esta columna no reciben un estado inventado.
- El estado sobrevive guardar_resultado -> MySQL -> cargar_ultima_corrida.
- El texto de la interfaz habla de margen descriptivo, no de probabilidad.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.prediccion.pronostico_futuro import (
    ESTADO_ERROR_CERO,
    ESTADO_ERROR_MEDIDO,
    ESTADO_ERROR_NO_DISPONIBLE,
    es_baja_confianza,
    estado_error_del_producto,
    estado_error_historico,
)
from app.services.prediccion.reposicion import calcular_tabla_reorden
from app.services.resultados import _armar_pronostico_pivot

RAIZ = Path(__file__).resolve().parent.parent


class TestEstadoDelErrorHistorico:
    def test_sin_valor_es_no_disponible(self):
        assert estado_error_historico(None) == ESTADO_ERROR_NO_DISPONIBLE
        assert estado_error_historico(np.nan) == ESTADO_ERROR_NO_DISPONIBLE

    def test_error_exactamente_cero_es_medido_en_cero(self):
        assert estado_error_historico(0.0) == ESTADO_ERROR_CERO

    def test_error_positivo_es_medido(self):
        assert estado_error_historico(29.33) == ESTADO_ERROR_MEDIDO

    def test_resumen_de_un_producto(self):
        # Si algún mes no tiene error medido, el producto entero no queda "medido".
        assert estado_error_del_producto([ESTADO_ERROR_MEDIDO, ESTADO_ERROR_NO_DISPONIBLE]) == ESTADO_ERROR_NO_DISPONIBLE
        assert estado_error_del_producto([ESTADO_ERROR_CERO, ESTADO_ERROR_CERO]) == ESTADO_ERROR_CERO
        assert estado_error_del_producto([ESTADO_ERROR_MEDIDO, ESTADO_ERROR_CERO]) == ESTADO_ERROR_MEDIDO
        assert estado_error_del_producto([]) is None


class TestEstadoEnLaTablaDeReposicion:
    def _calcular(self, mae):
        df_model = pd.DataFrame({"producto_id": ["P"]})
        df_mensual = pd.DataFrame([{
            "producto_id": "P", "fecha": pd.Timestamp("2026-06-01"),
            "lead_time_semanas": np.nan, "costo_pedido": np.nan,
            "porcentaje_costo_mantenimiento": np.nan, "costo_unitario": np.nan, "inventario": 40,
        }])
        comportamiento = {
            "producto_id": "P", "nombre_producto": "P", "categoria": "T",
            "proveedor_principal": None, "proveedor_alterno": None,
            "categoria_abc": "A", "variabilidad": "Baja", "metodo_campeon": "Media móvil",
        }
        if mae is not None:
            comportamiento["mae_media_movil"] = mae
        pronostico = pd.DataFrame([{"producto_id": "P", "prediccion_final": v} for v in [90, 100, 110]])
        reorder_df, _, _ = calcular_tabla_reorden(
            df_model, df_mensual, pd.DataFrame([comportamiento]), pronostico, presupuesto_capital_trabajo=1_000_000
        )
        return reorder_df.iloc[0]

    def test_sin_backtest_el_estado_es_no_disponible_aunque_el_stock_exista(self):
        f = self._calcular(mae=None)
        assert f["fuente_sigma"] == "desvío del pronóstico (sin backtest)"
        assert f["estado_error_historico"] == ESTADO_ERROR_NO_DISPONIBLE
        # El stock de seguridad sigue calculándose (con el desvío del pronóstico).
        assert not math.isnan(f["stock_seguridad"])

    def test_backtest_con_error_cero_se_marca_como_cero(self):
        f = self._calcular(mae=0.0)
        assert f["fuente_sigma"] == "MAE de backtest"
        assert f["estado_error_historico"] == ESTADO_ERROR_CERO

    def test_backtest_con_error_se_marca_como_medido(self):
        f = self._calcular(mae=29.33)
        assert f["fuente_sigma"] == "MAE de backtest"
        assert f["estado_error_historico"] == ESTADO_ERROR_MEDIDO


class TestRecargaDesdeLaBaseDeDatos:
    def test_corrida_anterior_sin_estado_queda_sin_estado(self):
        """Filas guardadas antes de la columna estado_error llegan como None
        desde MySQL. No deben presentarse como "medido" ni como otro estado."""
        registros = [
            {"producto_id": "A", "horizonte": 1, "prediccion_final": 10.0, "prediccion_min": 5.0,
             "prediccion_max": 15.0, "nombre_producto": "A", "categoria": "T", "estado_error": None},
        ]
        pivot = _armar_pronostico_pivot(registros)
        assert pivot[0]["estado_error"] is None
        assert pivot[0]["estados_error"] == [None]

    def test_corrida_nueva_conserva_el_estado_por_mes_y_por_producto(self):
        registros = [
            {"producto_id": "A", "horizonte": 1, "prediccion_final": 10.0, "prediccion_min": 10.0,
             "prediccion_max": 10.0, "nombre_producto": "A", "categoria": "T", "estado_error": ESTADO_ERROR_NO_DISPONIBLE},
            {"producto_id": "A", "horizonte": 2, "prediccion_final": 12.0, "prediccion_min": 8.0,
             "prediccion_max": 16.0, "nombre_producto": "A", "categoria": "T", "estado_error": ESTADO_ERROR_MEDIDO},
        ]
        pivot = _armar_pronostico_pivot(registros)
        assert pivot[0]["estados_error"] == [ESTADO_ERROR_NO_DISPONIBLE, ESTADO_ERROR_MEDIDO]
        assert pivot[0]["estado_error"] == ESTADO_ERROR_NO_DISPONIBLE


class TestSinMargenNoSeMuestraComoConfiable:
    """D3: un producto sin margen medido no puede verse como confiable."""

    def test_pivot_marca_sin_margen_aunque_el_rango_sea_angosto(self):
        registros = [
            {"producto_id": "A", "horizonte": 1, "prediccion_final": 10.0, "prediccion_min": 10.0,
             "prediccion_max": 10.0, "nombre_producto": "A", "categoria": "T", "estado_error": ESTADO_ERROR_NO_DISPONIBLE},
            {"producto_id": "A", "horizonte": 2, "prediccion_final": 12.0, "prediccion_min": 8.0,
             "prediccion_max": 16.0, "nombre_producto": "A", "categoria": "T", "estado_error": ESTADO_ERROR_MEDIDO},
            {"producto_id": "B", "horizonte": 1, "prediccion_final": 5.0, "prediccion_min": 3.0,
             "prediccion_max": 7.0, "nombre_producto": "B", "categoria": "T", "estado_error": ESTADO_ERROR_MEDIDO},
        ]
        por_id = {p["producto_id"]: p for p in _armar_pronostico_pivot(registros)}
        assert por_id["A"]["sin_margen"] is True
        assert por_id["A"]["sin_margen_por_mes"] == [True, False]
        assert por_id["B"]["sin_margen"] is False

    def test_es_baja_confianza_no_cambia_su_definicion(self):
        # Sin margen el mínimo no se recorta: sigue sin ser "baja confianza";
        # la advertencia de sin margen es una bandera aparte.
        assert es_baja_confianza(10.0, 10.0) is False
        assert es_baja_confianza(10.0, 0.0) is True


class TestPersistenciaEnMySQL:
    """El estado del error tiene que sobrevivir a guardar_resultado +
    cargar_ultima_corrida (la app recarga desde MySQL)."""

    def test_estado_error_sobrevive_guardar_y_cargar(self):
        import uuid
        from sqlalchemy import text as sql_text
        from app.services.db import get_engine
        from app.services.resultados import cargar_ultima_corrida, guardar_resultado

        engine = get_engine()
        try:
            engine.connect().close()
        except Exception as e:  # pragma: no cover - depende del entorno
            pytest.skip(f"MySQL no disponible: {e}")

        empresa_id = uuid.uuid4().hex
        with engine.begin() as conn:
            conn.execute(sql_text("INSERT INTO empresas (id) VALUES (:id)"), {"id": empresa_id})
        try:
            filas = [
                {"producto_id": "A", "nombre_producto": "A", "categoria": "T", "mes_pronosticado": "2026-07",
                 "horizonte": 1, "prediccion_xgboost": 10.0, "prediccion_regresion_lineal": 10.0,
                 "prediccion_baseline": 10.0, "prediccion_final": 10.0, "prediccion_min": 10.0,
                 "prediccion_max": 10.0, "estado_error": ESTADO_ERROR_NO_DISPONIBLE, "metodo_usado": "Media móvil"},
                {"producto_id": "B", "nombre_producto": "B", "categoria": "T", "mes_pronosticado": "2026-07",
                 "horizonte": 1, "prediccion_xgboost": 5.0, "prediccion_regresion_lineal": 5.0,
                 "prediccion_baseline": 5.0, "prediccion_final": 5.0, "prediccion_min": 3.0,
                 "prediccion_max": 7.0, "estado_error": ESTADO_ERROR_MEDIDO, "metodo_usado": "Media móvil"},
            ]
            guardar_resultado({"horizonte_meses": 1, "pronostico_futuro": filas}, empresa_id)
            cargada = cargar_ultima_corrida(empresa_id)
            por_id = {p["producto_id"]: p for p in cargada["pronostico_pivot"]}
            assert por_id["A"]["estado_error"] == ESTADO_ERROR_NO_DISPONIBLE
            assert por_id["A"]["sin_margen"] is True
            assert por_id["B"]["estado_error"] == ESTADO_ERROR_MEDIDO
            assert por_id["B"]["sin_margen"] is False
        finally:
            with engine.begin() as conn:
                conn.execute(sql_text("DELETE FROM corridas WHERE empresa_id = :id"), {"id": empresa_id})
                conn.execute(sql_text("DELETE FROM empresas WHERE id = :id"), {"id": empresa_id})


class TestTextoDeLaInterfaz:
    def test_el_dashboard_no_presenta_el_rango_como_probabilidad(self):
        texto = (RAIZ / "app" / "templates" / "dashboard.html").read_text(encoding="utf-8")
        assert "margen descriptivo" in texto
        assert "No es una probabilidad" in texto
        # Frases que afirmaban una probabilidad o una "seguridad" del número.
        assert "cantidad más probable" not in texto
        assert "qué tan segura es" not in texto
        # D3: el ícono de sin margen existe y dice que no es confiable.
        assert "Sin margen medido" in texto
