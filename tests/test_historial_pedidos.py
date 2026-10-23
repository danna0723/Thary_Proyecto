"""
Historial de pedidos (corrección E de la revisión, ciclo de pedidos).

pedidos guarda el estado vigente por producto; pedidos_movimientos guarda
cada confirmación y cada deshacer. Estas pruebas comprueban que un segundo
pedido del mismo producto no borra el primero del historial, y que deshacer
deja registro con la cantidad que tenía.
"""

import uuid

import pytest
from sqlalchemy import text as sql_text

from app.services.db import get_engine
from app.services.pedidos import (
    cargar_movimientos,
    cargar_pedidos,
    quitar_pedido,
    registrar_pedido,
)


@pytest.fixture
def empresa():
    try:
        engine = get_engine()
        engine.connect().close()
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip(f"MySQL no disponible: {e}")

    empresa_id = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(sql_text("INSERT INTO empresas (id) VALUES (:id)"), {"id": empresa_id})
    yield empresa_id
    with engine.begin() as conn:
        for tabla in ("pedidos_movimientos", "pedidos", "corridas"):
            conn.execute(sql_text(f"DELETE FROM {tabla} WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(sql_text("DELETE FROM empresas WHERE id = :id"), {"id": empresa_id})


def test_segundo_pedido_no_borra_el_primero_del_historial(empresa):
    registrar_pedido(empresa, "P1", "Proveedor X", 10, 100.0, "ana")
    registrar_pedido(empresa, "P1", "Proveedor Y", 25, 260.0, "beto")

    # El estado vigente es el último pedido.
    vigente = cargar_pedidos(empresa)["P1"]
    assert vigente["cantidad"] == 25
    assert vigente["proveedor"] == "Proveedor Y"

    # El historial conserva los dos.
    movimientos = cargar_movimientos(empresa, "P1")
    assert [m["accion"] for m in movimientos] == ["confirmado", "confirmado"]
    assert [m["cantidad"] for m in movimientos] == [10, 25]
    assert [m["usuario"] for m in movimientos] == ["ana", "beto"]


def test_deshacer_queda_en_el_historial_con_la_cantidad_que_tenia(empresa):
    registrar_pedido(empresa, "P1", "Proveedor X", 10, 100.0, "ana")
    quitar_pedido(empresa, "P1")

    assert "P1" not in cargar_pedidos(empresa)
    movimientos = cargar_movimientos(empresa, "P1")
    assert [m["accion"] for m in movimientos] == ["confirmado", "deshecho"]
    assert movimientos[1]["cantidad"] == 10
    assert movimientos[1]["usuario"] == "ana"


def test_deshacer_sin_pedido_no_registra_nada(empresa):
    quitar_pedido(empresa, "P9")
    assert cargar_movimientos(empresa) == []


def test_historial_separa_empresas_y_productos(empresa):
    registrar_pedido(empresa, "P1", "X", 1, 1.0, "ana")
    registrar_pedido(empresa, "P2", "X", 2, 2.0, "ana")

    assert len(cargar_movimientos(empresa)) == 2
    assert len(cargar_movimientos(empresa, "P1")) == 1
    assert cargar_movimientos(str(uuid.uuid4().hex)) == []


def test_pantalla_pedido_muestra_el_historial(empresa):
    import os
    from app import create_app
    from app.services.resultados import guardar_resultado

    guardar_resultado({"horizonte_meses": 1, "pronostico_futuro": [{
        "producto_id": "P1", "nombre_producto": "Prod P1", "categoria": "T", "mes_pronosticado": "2026-07",
        "horizonte": 1, "prediccion_xgboost": 10.0, "prediccion_regresion_lineal": 10.0,
        "prediccion_baseline": 10.0, "prediccion_final": 10.0, "prediccion_min": 10.0,
        "prediccion_max": 10.0, "estado_error": "medido", "metodo_usado": "Media móvil",
    }]}, empresa)
    registrar_pedido(empresa, "P1", "Proveedor X", 10, 100.0, "ana")
    registrar_pedido(empresa, "P1", "Proveedor Y", 25, 260.0, "beto")
    quitar_pedido(empresa, "P1")

    os.environ.setdefault("SECRET_KEY", uuid.uuid4().hex)
    app = create_app()
    app.config["TESTING"] = True
    cliente = app.test_client()
    with cliente.session_transaction() as sesion:
        sesion.update(username="qa", nombre="QA", rol="admin", empresa_id=empresa)
    html = cliente.get("/pedido").get_data(as_text=True)

    assert "Historial de movimientos" in html
    assert "Proveedor X" in html and "Proveedor Y" in html
    assert "Deshecho" in html
    assert "beto" in html
