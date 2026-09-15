from sqlalchemy import text

from app.services.db import get_engine


def cargar_nombre_empresa(empresa_id):
    """Devuelve el nombre que el admin le puso a su empresa, o None si
    todavía no lo ingresó — se usa para mostrarlo en el sidebar y en
    los reportes, y para precargarlo la próxima vez que suba un CSV."""
    engine = get_engine()
    with engine.connect() as conn:
        fila = conn.execute(
            text("SELECT nombre_empresa FROM empresas WHERE id = :id"),
            {"id": empresa_id},
        ).first()
    return fila[0] if fila else None


def guardar_nombre_empresa(empresa_id, nombre_empresa):
    """Guarda (o actualiza) el nombre de la empresa. Se llama desde
    /procesar cada vez que se sube un archivo — así el campo queda
    editable sin necesitar una pantalla de configuración aparte."""
    nombre_empresa = (nombre_empresa or "").strip()
    if not nombre_empresa:
        return
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE empresas SET nombre_empresa = :nombre WHERE id = :id"),
            {"nombre": nombre_empresa, "id": empresa_id},
        )
