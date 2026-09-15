"""
Bitácora de movimientos: quién hizo qué y cuándo dentro de una empresa
(subir un CSV, confirmar o deshacer un pedido, crear un empleado). Se
usa tanto en "Mi perfil" (cada usuario ve lo suyo) como en "Actividad
de la empresa" (solo el admin, ve todo lo de su empresa).
"""

from sqlalchemy import text

from app.services.db import get_engine


def registrar(empresa_id, username, nombre_usuario, tipo, descripcion):
    """Agrega una fila a la bitácora. Se llama desde la ruta que
    realiza la acción, justo después de que se confirma que salió
    bien — si la acción falla, no queda registrada."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO actividad (empresa_id, username, nombre_usuario, tipo, descripcion) "
                "VALUES (:empresa_id, :username, :nombre_usuario, :tipo, :descripcion)"
            ),
            {
                "empresa_id": empresa_id,
                "username": username,
                "nombre_usuario": nombre_usuario,
                "tipo": tipo,
                "descripcion": descripcion,
            },
        )


def _con_fecha_legible(filas):
    resultado = []
    for fila in filas:
        fila = dict(fila)
        if fila.get("fecha"):
            fila["fecha"] = fila["fecha"].strftime("%d/%m/%Y %H:%M")
        resultado.append(fila)
    return resultado


def listar_propia(empresa_id, username, limite=200):
    """Movimientos de un solo usuario (pantalla "Mi perfil")."""
    engine = get_engine()
    with engine.connect() as conn:
        filas = conn.execute(
            text(
                "SELECT * FROM actividad WHERE empresa_id = :empresa_id AND username = :username "
                "ORDER BY fecha DESC LIMIT :limite"
            ),
            {"empresa_id": empresa_id, "username": username, "limite": limite},
        ).mappings().all()
    return _con_fecha_legible(filas)


def listar_empresa(empresa_id, limite=300):
    """Movimientos de todos los usuarios de una empresa (pantalla
    "Actividad de la empresa", solo para el admin)."""
    engine = get_engine()
    with engine.connect() as conn:
        filas = conn.execute(
            text(
                "SELECT * FROM actividad WHERE empresa_id = :empresa_id "
                "ORDER BY fecha DESC LIMIT :limite"
            ),
            {"empresa_id": empresa_id, "limite": limite},
        ).mappings().all()
    return _con_fecha_legible(filas)
