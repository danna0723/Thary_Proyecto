"""
Prueba de aislamiento entre empresas (corrección E de la revisión).

Dos empresas distintas en la misma base. La empresa A tiene una corrida,
un pedido confirmado y un archivo de resultados. Con una sesión de la
empresa B no se debe poder leer, modificar ni descargar nada de A. Las
consultas se filtran por empresa_id de la sesión; esta prueba lo comprueba
de punta a punta contra MySQL.
"""

import os
import uuid

import pytest
from sqlalchemy import text as sql_text

from app import create_app
from app.services.db import get_engine
from app.services.pedidos import cargar_pedidos, registrar_pedido
from app.services.resultados import cargar_ultima_corrida, guardar_resultado

PRODUCTO_A = "PRODUCTO_SOLO_DE_A_7731"
ARCHIVO_A = "secreto_empresa_a_7731.txt"


def _crear_empresa(conn):
    empresa_id = uuid.uuid4().hex
    conn.execute(sql_text("INSERT INTO empresas (id) VALUES (:id)"), {"id": empresa_id})
    return empresa_id


@pytest.fixture(scope="module")
def entorno():
    try:
        engine = get_engine()
        engine.connect().close()
    except Exception as e:  # pragma: no cover - depende del entorno
        pytest.skip(f"MySQL no disponible: {e}")

    # create_app exige SECRET_KEY; para la prueba basta una clave temporal.
    os.environ.setdefault("SECRET_KEY", uuid.uuid4().hex)
    app = create_app()
    app.config["TESTING"] = True

    with engine.begin() as conn:
        empresa_a = _crear_empresa(conn)
        empresa_b = _crear_empresa(conn)

    fila = {
        "producto_id": PRODUCTO_A, "nombre_producto": "Producto de A", "categoria": "T",
        "mes_pronosticado": "2026-07", "horizonte": 1, "prediccion_xgboost": 10.0,
        "prediccion_regresion_lineal": 10.0, "prediccion_baseline": 10.0, "prediccion_final": 10.0,
        "prediccion_min": 8.0, "prediccion_max": 12.0, "estado_error": "medido", "metodo_usado": "Media móvil",
    }
    guardar_resultado({"horizonte_meses": 1, "pronostico_futuro": [fila]}, empresa_a)
    registrar_pedido(empresa_a, PRODUCTO_A, "Proveedor A", 5, 100.0, "usuario_a")

    carpeta_a = os.path.join(app.config["RESULTADOS_FOLDER_BASE"], empresa_a)
    os.makedirs(carpeta_a, exist_ok=True)
    with open(os.path.join(carpeta_a, ARCHIVO_A), "w", encoding="utf-8") as f:
        f.write("datos de A")

    yield {"app": app, "empresa_a": empresa_a, "empresa_b": empresa_b, "engine": engine}

    with engine.begin() as conn:
        for empresa in (empresa_a, empresa_b):
            conn.execute(sql_text("DELETE FROM pedidos_movimientos WHERE empresa_id = :id"), {"id": empresa})
            conn.execute(sql_text("DELETE FROM pedidos WHERE empresa_id = :id"), {"id": empresa})
            conn.execute(sql_text("DELETE FROM actividad WHERE empresa_id = :id"), {"id": empresa})
            conn.execute(sql_text("DELETE FROM corridas WHERE empresa_id = :id"), {"id": empresa})
            conn.execute(sql_text("DELETE FROM empresas WHERE id = :id"), {"id": empresa})
    import shutil
    shutil.rmtree(carpeta_a, ignore_errors=True)


def _cliente_como_empresa_b(entorno):
    cliente = entorno["app"].test_client()
    with cliente.session_transaction() as sesion:
        sesion["username"] = "usuario_b"
        sesion["nombre"] = "Usuario B"
        sesion["rol"] = "admin"
        sesion["empresa_id"] = entorno["empresa_b"]
    return cliente


class TestServiciosFiltranPorEmpresa:
    def test_b_no_ve_la_corrida_de_a(self, entorno):
        assert cargar_ultima_corrida(entorno["empresa_b"]) is None
        assert cargar_ultima_corrida(entorno["empresa_a"]) is not None

    def test_b_no_ve_los_pedidos_de_a(self, entorno):
        assert PRODUCTO_A not in cargar_pedidos(entorno["empresa_b"])
        assert PRODUCTO_A in cargar_pedidos(entorno["empresa_a"])


class TestRutasNoExponenDatosDeOtraEmpresa:
    @pytest.mark.parametrize("ruta", ["/dashboard", "/reporte-demanda", "/pedido", "/presupuesto"])
    def test_b_no_recibe_datos_de_a(self, entorno, ruta):
        cliente = _cliente_como_empresa_b(entorno)
        respuesta = cliente.get(ruta)
        assert respuesta.status_code in (200, 302)
        assert PRODUCTO_A not in respuesta.get_data(as_text=True)

    def test_b_no_puede_descargar_el_archivo_de_a(self, entorno):
        cliente = _cliente_como_empresa_b(entorno)
        respuesta = cliente.get(f"/descargar/{ARCHIVO_A}")
        assert respuesta.status_code == 404
        assert b"datos de A" not in respuesta.data

    def test_b_no_puede_deshacer_el_pedido_de_a(self, entorno):
        cliente = _cliente_como_empresa_b(entorno)
        cliente.post(f"/pedido/deshacer/{PRODUCTO_A}")
        # El pedido de A tiene que seguir existiendo.
        assert PRODUCTO_A in cargar_pedidos(entorno["empresa_a"])
