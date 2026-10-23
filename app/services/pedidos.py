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


def cargar_movimientos(empresa_id, producto_id=None):
    """Historial de pedidos de una empresa, del más antiguo al más reciente.
    Cada fila es una confirmación ("confirmado") o un deshacer ("deshecho"),
    con la cantidad, costo y usuario de ese movimiento. Si se pasa
    producto_id, solo devuelve los movimientos de ese producto."""
    engine = get_engine()
    consulta = (
        "SELECT producto_id, accion, proveedor, cantidad, costo, usuario, fecha_hora "
        "FROM pedidos_movimientos WHERE empresa_id = :empresa_id"
    )
    parametros = {"empresa_id": empresa_id}
    if producto_id is not None:
        consulta += " AND producto_id = :producto_id"
        parametros["producto_id"] = str(producto_id)
    consulta += " ORDER BY fecha_hora, id"
    with engine.connect() as conn:
        filas = conn.execute(text(consulta), parametros).mappings().all()
    return [
        {
            "producto_id": f["producto_id"],
            "accion": f["accion"],
            "proveedor": f["proveedor"],
            "cantidad": f["cantidad"],
            "costo": f["costo"],
            "usuario": f["usuario"],
            "fecha_hora": f["fecha_hora"].strftime("%Y-%m-%d %H:%M:%S") if f["fecha_hora"] else None,
        }
        for f in filas
    ]


def registrar_pedido(empresa_id, producto_id, proveedor, cantidad, costo, usuario):
    """Marca un producto como pedido: guarda proveedor elegido, cantidad
    y costo (tomados del análisis vigente, no de lo que mande el
    formulario) junto con quién lo confirmó y cuándo.

    Ciclo de pedidos — dos tablas con roles distintos:
      - pedidos: estado VIGENTE, una fila por producto (clave única
        empresa_id+producto_id). Un segundo pedido del mismo producto
        reemplaza al anterior en esta tabla.
      - pedidos_movimientos: historial, una fila por cada confirmación y
        cada deshacer, con su cantidad y costo. Nada se reemplaza ahí.
    Ambas escrituras van en la misma transacción."""
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
        conn.execute(
            text(
                "INSERT INTO pedidos_movimientos (empresa_id, producto_id, accion, proveedor, cantidad, costo, usuario, fecha_hora) "
                "VALUES (:empresa_id, :producto_id, 'confirmado', :proveedor, :cantidad, :costo, :usuario, NOW())"
            ),
            {
                "empresa_id": empresa_id, "producto_id": str(producto_id), "proveedor": proveedor,
                "cantidad": cantidad, "costo": costo, "usuario": usuario,
            },
        )


def quitar_pedido(empresa_id, producto_id):
    """Deshace la confirmación de un pedido (por si se marcó por error).
    El pedido que se deshace queda registrado en el historial como
    "deshecho", con la cantidad y el costo que tenía."""
    engine = get_engine()
    with engine.begin() as conn:
        vigente = conn.execute(
            text(
                "SELECT proveedor, cantidad, costo, usuario FROM pedidos "
                "WHERE empresa_id = :empresa_id AND producto_id = :producto_id FOR UPDATE"
            ),
            {"empresa_id": empresa_id, "producto_id": str(producto_id)},
        ).mappings().first()
        if vigente is None:
            return
        conn.execute(
            text(
                "INSERT INTO pedidos_movimientos (empresa_id, producto_id, accion, proveedor, cantidad, costo, usuario, fecha_hora) "
                "VALUES (:empresa_id, :producto_id, 'deshecho', :proveedor, :cantidad, :costo, :usuario, NOW())"
            ),
            {
                "empresa_id": empresa_id, "producto_id": str(producto_id), "proveedor": vigente["proveedor"],
                "cantidad": vigente["cantidad"], "costo": vigente["costo"], "usuario": vigente["usuario"],
            },
        )
        conn.execute(
            text("DELETE FROM pedidos WHERE empresa_id = :empresa_id AND producto_id = :producto_id"),
            {"empresa_id": empresa_id, "producto_id": str(producto_id)},
        )
