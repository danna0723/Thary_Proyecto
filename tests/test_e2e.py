"""
Pruebas END-TO-END para Thary, usando el cliente de pruebas de Flask
(app.test_client()). Simulan exactamente las peticiones HTTP que manda
un navegador — login, subir un CSV, procesarlo, ver el dashboard,
confirmar y deshacer un pedido — contra una base de datos MySQL real
(no una simulada), igual que la validación manual con usuarios
documentada en la sección 3.4.4 de la tesis, pero automatizada.

No ejecutan JavaScript (no hay un navegador real detrás, así que cosas
como el ordenamiento de las tablas con DataTables o los gráficos con
Morris.js no se prueban acá) — lo que sí se verifica de punta a punta
es que las rutas, las sesiones, los permisos por rol y la base de datos
funcionen juntos correctamente.

Necesitan un servidor MySQL alcanzable con las mismas variables de
entorno que usa la aplicación (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD).
Si no hay uno disponible, todo el archivo se salta en vez de fallar
(ver _mysql_disponible más abajo) — para no romper la suite en una
máquina que no tenga MySQL corriendo (por ejemplo, en GitHub Actions,
que hoy no lo tiene configurado).

A propósito, NO usan una base de datos separada (crearla requeriría
permisos que el usuario de la aplicación no tiene por diseño — ver
db.py — y pedirle a quien corra los tests que use el usuario root no
es razonable). En cambio, crean una "empresa" de prueba bien
identificada (USERNAME_ADMIN más abajo) dentro de la base real, y la
borran por completo al terminar — nunca tocan ninguna otra empresa, y
el usuario de la aplicación ya tiene permiso de sobra para esto, porque
es exactamente lo que la aplicación hace todo el tiempo.
"""

import io
import os
import re

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

USERNAME_ADMIN = "admin_e2e_thary_pruebas"
USERNAME_EMPLEADO = "empleado_e2e_thary_pruebas"


def _config_mysql():
    return {
        "host": os.environ.get("DB_HOST", "127.0.0.1"),
        "port": os.environ.get("DB_PORT", "3307"),
        "user": os.environ.get("DB_USER", "thary_app"),
        "password": os.environ.get("DB_PASSWORD", ""),
    }


def _mysql_disponible():
    cfg = _config_mysql()
    try:
        engine = create_engine(
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/?charset=utf8mb4"
        )
        with engine.connect():
            pass
        engine.dispose()
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _mysql_disponible(),
    reason=(
        "No hay un servidor MySQL disponible con las credenciales de entorno "
        "(DB_HOST/DB_PORT/DB_USER/DB_PASSWORD) — las pruebas end-to-end necesitan "
        "una base de datos real, igual que la aplicación en uso normal."
    ),
)


def _empresa_id_de_username_si_existe(engine, username):
    with engine.connect() as conn:
        fila = conn.execute(
            text("SELECT empresa_id FROM usuarios WHERE username = :u"), {"u": username}
        ).first()
    return fila[0] if fila else None


def _borrar_empresa_de_prueba(engine, empresa_id):
    """Borra en cascada todo lo que haya quedado de la empresa de
    prueba (y solo de ella — nunca de otra). "corridas" cascadea solo
    hacia sus tablas hijas (segmentacion, reorder, etc., que tienen ON
    DELETE CASCADE a corridas.id), pero "empresas" no cascadea hacia
    usuarios/pedidos/corridas/actividad, así que se borra en el orden
    correcto a mano."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM corridas WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(text("DELETE FROM pedidos_movimientos WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(text("DELETE FROM pedidos WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(text("DELETE FROM actividad WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(text("DELETE FROM usuarios WHERE empresa_id = :id"), {"id": empresa_id})
        conn.execute(text("DELETE FROM empresas WHERE id = :id"), {"id": empresa_id})


def _armar_csv_e2e():
    """Dos productos: SALUD con muy poco inventario (tiene que quedar en
    alerta de reposición sí o sí, para poder probar la confirmación de
    pedidos de forma determinística) y SOBRA con inventario de sobra
    (no debería quedar en alerta)."""
    meses = pd.date_range("2023-01-01", periods=12, freq="MS")
    demanda_salud = [50, 55, 48, 52, 51, 49, 53, 50, 47, 54, 52, 50]
    demanda_sobra = [20, 22, 19, 21, 20, 18, 23, 20, 19, 22, 21, 20]

    filas = []
    for mes, demanda in zip(meses, demanda_salud):
        filas.append({"Fecha": mes, "Codigo": "SALUD", "Cantidad": demanda, "Inventario": 5})
    for mes, demanda in zip(meses, demanda_sobra):
        filas.append({"Fecha": mes, "Codigo": "SOBRA", "Cantidad": demanda, "Inventario": 5000})

    buffer = io.StringIO()
    pd.DataFrame(filas).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.environ.setdefault("SECRET_KEY", "clave-de-pruebas-e2e-no-usar-en-produccion")

    from app import create_app
    aplicacion = create_app()
    aplicacion.config["TESTING"] = True

    # Sin esto, /procesar escribiría el CSV subido y los resultados
    # (gráficas, CSVs exportados) dentro de app/uploads y app/resultados
    # del proyecto real — se redirigen a una carpeta temporal.
    carpeta_tmp = tmp_path_factory.mktemp("e2e_archivos")
    aplicacion.config["UPLOAD_FOLDER_BASE"] = str(carpeta_tmp / "uploads")
    aplicacion.config["RESULTADOS_FOLDER_BASE"] = str(carpeta_tmp / "resultados")
    os.makedirs(aplicacion.config["UPLOAD_FOLDER_BASE"], exist_ok=True)
    os.makedirs(aplicacion.config["RESULTADOS_FOLDER_BASE"], exist_ok=True)

    return aplicacion


class _Contexto:
    pass


# EXTENSIÓN PROPIA — reproduce, en código, el mismo recorrido de la
# validación manual con usuarios (sección 3.4.4 de la tesis): registro
# del administrador, carga y procesamiento del historial, creación de
# una cuenta de personal de inventario, inicio de sesión de ese
# empleado. Se arma UNA sola vez (fixture de módulo) porque incluye
# correr el pipeline completo de predicción, que no es instantáneo —
# las pruebas de acá abajo reutilizan ese mismo estado para no repetirlo.
@pytest.fixture(scope="module")
def contexto(app):
    from app.services.db import get_engine
    engine = get_engine()

    # Por si una corrida anterior de este archivo se cortó a la mitad
    # (por ejemplo, si una prueba falló antes de llegar a la limpieza
    # final) — se borra ese resto antes de empezar, para arrancar de
    # una base limpia.
    empresa_vieja = _empresa_id_de_username_si_existe(engine, USERNAME_ADMIN)
    if empresa_vieja:
        _borrar_empresa_de_prueba(engine, empresa_vieja)

    ctx = _Contexto()
    ctx.client_admin = app.test_client()

    resp = ctx.client_admin.post("/registro", data={
        "nombre": "Admin de Prueba",
        "nombre_empresa": "Empresa E2E (Thary - pruebas automatizadas)",
        "username": USERNAME_ADMIN,
        "email": "admin_e2e_thary_pruebas@example.com",
        "password": "claveSegura123",
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/inicio")

    with ctx.client_admin.session_transaction() as sess:
        ctx.empresa_id = sess["empresa_id"]

    resp = ctx.client_admin.post("/procesar", data={
        "archivo": (io.BytesIO(_armar_csv_e2e()), "catalogo_e2e.csv"),
        "presupuesto": "",
        "horizonte_meses": "3",
    }, content_type="multipart/form-data", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")

    resp = ctx.client_admin.post("/usuarios/nuevo", data={
        "nombre": "Empleado de Prueba",
        "username": USERNAME_EMPLEADO,
        "email": "empleado_e2e_thary_pruebas@example.com",
        "password": "otraClave123",
        "rol": "empleado",
    }, follow_redirects=False)
    assert resp.status_code == 302

    ctx.client_empleado = app.test_client()
    resp = ctx.client_empleado.post("/login", data={
        "identificador": USERNAME_EMPLEADO,
        "password": "otraClave123",
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/inicio")

    yield ctx

    # Se borra la empresa de prueba completa (admin, empleado, el CSV
    # procesado, los pedidos confirmados/deshechos, la actividad) sin
    # importar si alguna prueba de arriba falló — nunca queda nada de
    # esta corrida en la base de datos real.
    _borrar_empresa_de_prueba(engine, ctx.empresa_id)


# ---------------------------------------------------------------
# Administrador
# ---------------------------------------------------------------

def test_login_explicito_del_administrador_funciona(app, contexto):
    """Además del auto-login que ya pasó al registrarse (dentro del
    fixture "contexto"), se prueba la ruta de login en sí: cerrar
    sesión y volver a entrar con las mismas credenciales, como haría un
    administrador que vuelve otro día."""
    cliente = app.test_client()
    resp = cliente.post("/login", data={
        "identificador": USERNAME_ADMIN, "password": "claveSegura123",
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/inicio")


def test_login_con_contrasena_incorrecta_no_entra(app, contexto):
    cliente = app.test_client()
    resp = cliente.post("/login", data={
        "identificador": USERNAME_ADMIN, "password": "contraseña-equivocada",
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert "incorrectos".encode("utf-8") in resp.data


def test_admin_ve_el_panel_tecnico_con_la_comparacion_de_modelos(contexto):
    resp = contexto.client_admin.get("/desarrollo")
    assert resp.status_code == 200
    assert "Juicio de expertos".encode("utf-8") in resp.data


def test_admin_ve_en_la_bitacora_lo_que_hizo(contexto):
    """Subir el archivo y crear la cuenta del empleado (ambos hechos
    dentro del fixture "contexto") tienen que quedar registrados acá —
    actividad.html muestra una etiqueta legible para cada tipo de
    movimiento ("Archivo subido", "Usuario creado"), no el valor interno
    ("archivo_subido", "usuario_creado") que se guarda en la base."""
    resp = contexto.client_admin.get("/actividad")
    assert resp.status_code == 200
    assert "Archivo subido".encode("utf-8") in resp.data
    assert "Usuario creado".encode("utf-8") in resp.data


def test_empleado_no_puede_entrar_a_la_pantalla_de_carga_del_admin(contexto):
    """admin_required tiene que redirigir a un empleado que intente
    entrar a una pantalla de administrador, en vez de mostrársela."""
    resp = contexto.client_empleado.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/dashboard")


# ---------------------------------------------------------------
# Personal de inventario
# ---------------------------------------------------------------

def test_empleado_ve_la_demanda_pronosticada(contexto):
    resp = contexto.client_empleado.get("/dashboard")
    assert resp.status_code == 200
    assert b"SALUD" in resp.data


def test_empleado_ve_estado_del_inventario(contexto):
    resp = contexto.client_empleado.get("/inventario")
    assert resp.status_code == 200


def test_empleado_ve_presupuesto_y_costos(contexto):
    resp = contexto.client_empleado.get("/presupuesto")
    assert resp.status_code == 200


def test_empleado_ve_calendario_de_pedidos(contexto):
    resp = contexto.client_empleado.get("/calendario")
    assert resp.status_code == 200


def test_empleado_confirma_y_despues_deshace_un_pedido(contexto):
    """SALUD tiene inventario_actual=5 con demanda mensual de ~50
    unidades, así que tiene que quedar en alerta de reposición sí o sí
    — se busca su casilla en la pantalla de Pedido en vez de asumir
    directamente el producto_id, para que la prueba siga funcionando
    aunque cambie cuál producto queda en alerta."""
    resp = contexto.client_empleado.get("/pedido")
    assert resp.status_code == 200
    coincidencia = re.search(r'name="confirmar" value="([^"]+)"', resp.data.decode("utf-8"))
    assert coincidencia is not None, "Se esperaba al menos un producto en alerta (SALUD, con poco inventario)"
    producto_id = coincidencia.group(1)

    resp = contexto.client_empleado.post("/pedido/confirmar", data={
        "confirmar": producto_id,
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert "Se confirmaron 1 pedido".encode("utf-8") in resp.data

    resp = contexto.client_empleado.get("/pedido")
    assert producto_id.encode("utf-8") in resp.data
    assert f'value="{producto_id}"'.encode("utf-8") not in resp.data  # ya no está pendiente

    resp = contexto.client_empleado.post(f"/pedido/deshacer/{producto_id}", follow_redirects=True)
    assert resp.status_code == 200
    assert "Se deshizo la confirmaci".encode("utf-8") in resp.data

    resp = contexto.client_empleado.get("/pedido")
    assert f'name="confirmar" value="{producto_id}"'.encode("utf-8") in resp.data  # vuelve a estar pendiente


def test_empleado_ve_su_perfil_con_lo_que_hizo(contexto):
    """El pedido confirmado y deshecho en el test anterior tiene que
    aparecer en el historial propio del empleado."""
    resp = contexto.client_empleado.get("/perfil")
    assert resp.status_code == 200
    assert "Empleado de Prueba".encode("utf-8") in resp.data
    assert b"pedido_confirmado" in resp.data
    assert b"pedido_deshecho" in resp.data
