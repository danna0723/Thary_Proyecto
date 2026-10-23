"""
D3 de punta a punta: el dashboard (HTML real, servido por la ruta) advierte
que un producto sin margen medido no es confiable, y no lo advierte en un
producto con margen medido.
"""

import os
import uuid

import pytest
from sqlalchemy import text as sql_text

from app import create_app
from app.services.db import get_engine
from app.services.resultados import guardar_resultado


def _fila(producto_id, estado_error, mes, horizonte):
    return {
        "producto_id": producto_id, "nombre_producto": f"Prod {producto_id}", "categoria": "T",
        "mes_pronosticado": mes, "horizonte": horizonte, "prediccion_xgboost": 10.0,
        "prediccion_regresion_lineal": 10.0, "prediccion_baseline": 10.0, "prediccion_final": 10.0,
        "prediccion_min": 10.0, "prediccion_max": 10.0, "estado_error": estado_error,
        "metodo_usado": "Media móvil",
    }


@pytest.fixture(scope="module")
def empresa_con_dos_productos():
    try:
        engine = get_engine()
        engine.connect().close()
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip(f"MySQL no disponible: {e}")

    empresa_id = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT INTO empresas (id) VALUES (:id)"), {"id": empresa_id})
    guardar_resultado({"horizonte_meses": 2, "pronostico_futuro": [
        _fila("SINMARGEN", "no disponible", "2026-07", 1),
        _fila("CONMARGEN", "medido", "2026-07", 1),
        _fila("SINMARGEN", "no disponible", "2026-08", 2),
        _fila("CONMARGEN", "medido", "2026-08", 2),
    ]}, empresa_id)

    os.environ.setdefault("SECRET_KEY", uuid.uuid4().hex)
    yield empresa_id

    with engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM corridas WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(sql_text("DELETE FROM empresas WHERE id = :id"), {"id": empresa_id})


def _html_dashboard(empresa_id):
    app = create_app()
    app.config["TESTING"] = True
    cliente = app.test_client()
    with cliente.session_transaction() as sesion:
        sesion.update(username="qa", nombre="QA", rol="admin", empresa_id=empresa_id)
    respuesta = cliente.get("/dashboard")
    assert respuesta.status_code == 200
    return respuesta.get_data(as_text=True)


def test_dashboard_advierte_sin_margen_solo_en_el_producto_sin_margen(empresa_con_dos_productos):
    html = _html_dashboard(empresa_con_dos_productos)
    assert "Prod SINMARGEN" in html
    assert "Prod CONMARGEN" in html
    # Icono por mes (2 meses) + icono del total del producto sin margen.
    assert html.count("Sin margen medido") == 3
    assert "sin error histórico" in html
    assert "margen descriptivo" in html


def test_dashboard_lista_los_productos_sin_pronostico():
    """C1: un producto del archivo que no tiene pronóstico (historial menor a
    4 meses) aparece en un aviso del dashboard, para que no quede oculto."""
    try:
        engine = get_engine()
        engine.connect().close()
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip(f"MySQL no disponible: {e}")

    empresa_id = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT INTO empresas (id) VALUES (:id)"), {"id": empresa_id})
    try:
        segmentacion = [
            {"producto_id": "CONMARGEN", "nombre_producto": "Prod CONMARGEN", "categoria": "T",
             "demanda_total": 10.0, "media": 1.0, "desviacion": 0.5, "cv": 0.5, "variabilidad": "Baja",
             "demanda_acumulada_pct": 50.0, "categoria_abc": "A", "metodo_campeon": "Media móvil",
             "mae_media_movil": 1.0, "mae_regresion_lineal": 1.0, "mae_xgboost": 1.0,
             "proveedor_principal": None, "proveedor_alterno": None},
            {"producto_id": "CORTO", "nombre_producto": "Prod CORTO", "categoria": "T",
             "demanda_total": 4.0, "media": 2.0, "desviacion": 0.0, "cv": 0.0, "variabilidad": "Baja",
             "demanda_acumulada_pct": 100.0, "categoria_abc": "C", "metodo_campeon": "Media móvil",
             "mae_media_movil": None, "mae_regresion_lineal": None, "mae_xgboost": None,
             "proveedor_principal": None, "proveedor_alterno": None},
        ]
        guardar_resultado({"horizonte_meses": 1, "segmentacion": segmentacion,
                           "pronostico_futuro": [_fila("CONMARGEN", "medido", "2026-07", 1)]}, empresa_id)

        os.environ.setdefault("SECRET_KEY", uuid.uuid4().hex)
        app = create_app()
        app.config["TESTING"] = True
        cliente = app.test_client()
        with cliente.session_transaction() as sesion:
            sesion.update(username="qa", nombre="QA", rol="admin", empresa_id=empresa_id)
        html = cliente.get("/dashboard").get_data(as_text=True)

        assert "1 producto(s) no tienen pronóstico" in html
        assert "menos de 4 meses de historial" in html
        assert "Prod CORTO" in html.split("no entran en el pronóstico")[1].split("</div>")[0]
    finally:
        with engine.begin() as conn:
            conn.execute(sql_text("DELETE FROM corridas WHERE empresa_id = :id"), {"id": empresa_id})
            conn.execute(sql_text("DELETE FROM empresas WHERE id = :id"), {"id": empresa_id})
