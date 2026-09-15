from sqlalchemy import text

from app.services.db import get_engine


def cargar_pedidos(empresa_id):
    """Devuelve el registro de pedidos confirmados de una empresa:
    {producto_id: {...}}. Vive en su propia tabla, separada de las
    corridas del pipeline — el historial de "qué ya se pidió" no
    debería borrarse cada vez que se sube un CSV nuevo para actualizar
    el pronóstico."""
    engine = get_engine()
    with engine.connect() as conn:
        filas = conn.execute(
            text(
                "SELECT producto_id, proveedor, cantidad, costo, usuario, fecha_hora "
                "FROM pedidos WHERE empresa_id = :empresa_id"
            ),
            {"empresa_id": empresa_id},
        ).mappings().all()
    return {
        fila["producto_id"]: {
            "proveedor": fila["proveedor"],
            "cantidad": fila["cantidad"],
            "costo": fila["costo"],
            "usuario": fila["usuario"],
            "fecha_hora": fila["fecha_hora"].strftime("%Y-%m-%d %H:%M:%S") if fila["fecha_hora"] else None,
        }
        for fila in filas
    }


def registrar_pedido(empresa_id, producto_id, proveedor, cantidad, costo, usuario):
    """Marca un producto como pedido: guarda proveedor elegido, cantidad
    y costo (tomados del análisis vigente, no de lo que mande el
    formulario) junto con quién lo confirmó y cuándo. Si el producto ya
    tenía un pedido confirmado, lo reemplaza (ON DUPLICATE KEY UPDATE,
    sobre la clave única empresa_id+producto_id) en vez de duplicarlo."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO pedidos (empresa_id, producto_id, proveedor, cantidad, costo, usuario, fecha_hora) "
                "VALUES (:empresa_id, :producto_id, :proveedor, :cantidad, :costo, :usuario, NOW()) "
                "ON DUPLICATE KEY UPDATE proveedor = VALUES(proveedor), cantidad = VALUES(cantidad), "
                "costo = VALUES(costo), usuario = VALUES(usuario), fecha_hora = VALUES(fecha_hora)"
            ),
            {
                "empresa_id": empresa_id, "producto_id": str(producto_id), "proveedor": proveedor,
                "cantidad": cantidad, "costo": costo, "usuario": usuario,
            },
        )


def quitar_pedido(empresa_id, producto_id):
    """Deshace la confirmación de un pedido (por si se marcó por error)."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM pedidos WHERE empresa_id = :empresa_id AND producto_id = :producto_id"),
            {"empresa_id": empresa_id, "producto_id": str(producto_id)},
        )
