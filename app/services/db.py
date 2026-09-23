"""
Capa de acceso a MySQL — reemplaza la persistencia anterior en archivos
JSON (usuarios.json, pedidos.json, perfil.json, ultimo_resultado.json)
por tablas relacionales. El motivo del cambio y las alternativas
consideradas están documentados como ADR en el capítulo 3 de la tesis.

Los archivos que siguen siendo archivos de verdad (el CSV que sube el
usuario, las gráficas .png, el modelo de XGBoost entrenado) NO se
tocan — siguen en disco bajo app/uploads/ y app/resultados/, porque no
tiene sentido meter binarios/imágenes en la base de datos. Lo que se
mueve a MySQL es exclusivamente la parte tabular: cuentas, empresas,
pedidos, y el resultado de cada corrida del pipeline (predicciones,
métricas, segmentación, reposición, validación cruzada).

Configuración por variables de entorno (mismo criterio que ya se usa
para Ollama en app/__init__.py) — nunca hardcodeada, para no dejar una
contraseña real en el código fuente:

    DB_HOST      (default "127.0.0.1")
    DB_PORT      (default "3307")
    DB_USER      (default "thary_app")
    DB_PASSWORD  (sin default — hay que definirla)
    DB_NAME      (default "sistema_inventario")
"""

import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

_engine = None


def _config():
    return {
        "host": os.environ.get("DB_HOST", "127.0.0.1"),
        "port": os.environ.get("DB_PORT", "3307"),
        "user": os.environ.get("DB_USER", "thary_app"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "sistema_inventario"),
    }


def get_engine():
    """Motor de SQLAlchemy, creado una sola vez y reutilizado en toda la
    app (crear uno por request sería un desperdicio — el engine ya
    mantiene su propio pool de conexiones)."""
    global _engine
    if _engine is None:
        cfg = _config()
        url = (
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset=utf8mb4"
        )
        # pool_pre_ping: antes de reusar una conexión del pool, hace un
        # ping liviano — evita el error típico de "MySQL server has
        # gone away" cuando el servidor cerró una conexión inactiva
        # mientras el proceso de Flask seguía corriendo.
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


TABLAS_DDL = [
    """
    CREATE TABLE IF NOT EXISTS empresas (
        id VARCHAR(32) PRIMARY KEY,
        nombre_empresa VARCHAR(120),
        fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS usuarios (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nombre VARCHAR(120) NOT NULL,
        username VARCHAR(80) NOT NULL UNIQUE,
        email VARCHAR(160) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        rol ENUM('admin', 'empleado') NOT NULL DEFAULT 'empleado',
        empresa_id VARCHAR(32) NOT NULL,
        FOREIGN KEY (empresa_id) REFERENCES empresas(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pedidos (
        id INT AUTO_INCREMENT PRIMARY KEY,
        empresa_id VARCHAR(32) NOT NULL,
        producto_id VARCHAR(64) NOT NULL,
        proveedor VARCHAR(160),
        cantidad DOUBLE,
        costo DOUBLE,
        usuario VARCHAR(120),
        fecha_hora DATETIME,
        FOREIGN KEY (empresa_id) REFERENCES empresas(id),
        UNIQUE KEY uq_pedido_producto (empresa_id, producto_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # "corrida" = una vez que se sube y procesa un CSV. Es la tabla
    # padre de todo el resultado del pipeline — antes era un solo
    # ultimo_resultado.json que se PISABA en cada subida; ahora cada
    # corrida queda como una fila propia, así que de paso se gana
    # historial de corridas pasadas, no solo la última (aunque hoy la
    # app solo lee la más reciente, ver main.py::cargar_ultima_corrida).
    """
    CREATE TABLE IF NOT EXISTS corridas (
        id INT AUTO_INCREMENT PRIMARY KEY,
        empresa_id VARCHAR(32) NOT NULL,
        fecha_corrida DATETIME DEFAULT CURRENT_TIMESTAMP,
        horizonte_meses INT,
        presupuesto_capital_trabajo DOUBLE,
        presupuesto_por_producto DOUBLE,
        mape_final DOUBLE,
        wape_final DOUBLE,
        costo_total_pedidos_urgentes DOUBLE,
        valor_total_inventario DOUBLE,
        productos_sin_dato_inventario INT,
        productos_en_alerta INT,
        features_regresion_lineal TEXT,
        FOREIGN KEY (empresa_id) REFERENCES empresas(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    # Bitácora de movimientos: qué usuario hizo qué y cuándo (subir un
    # CSV, confirmar/deshacer un pedido, crear un empleado). Se usa
    # tanto para "Mi perfil" (cada usuario ve lo suyo, filtrando por
    # username) como para "Actividad de la empresa" (el admin ve todo,
    # filtrando solo por empresa_id). username y nombre_usuario quedan
    # los dos guardados (no solo un id) para no depender de un JOIN ni
    # de que el usuario siga existiendo más adelante.
    """
    CREATE TABLE IF NOT EXISTS actividad (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        empresa_id VARCHAR(32) NOT NULL,
        username VARCHAR(80) NOT NULL,
        nombre_usuario VARCHAR(120) NOT NULL,
        tipo VARCHAR(30) NOT NULL,
        descripcion VARCHAR(300) NOT NULL,
        fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (empresa_id) REFERENCES empresas(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS segmentacion (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        producto_id VARCHAR(64),
        nombre_producto VARCHAR(200),
        categoria VARCHAR(120),
        demanda_total DOUBLE,
        media DOUBLE,
        desviacion DOUBLE,
        cv DOUBLE,
        variabilidad VARCHAR(20),
        demanda_acumulada_pct DOUBLE,
        categoria_abc VARCHAR(5),
        metodo_campeon VARCHAR(40),
        proveedor_principal VARCHAR(160),
        proveedor_alterno VARCHAR(160),
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS metricas_modelos (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        modelo VARCHAR(60),
        mae DOUBLE,
        rmse DOUBLE,
        r2 DOUBLE,
        mediana_error DOUBLE,
        mape DOUBLE,
        wape DOUBLE,
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS predicciones_backtest (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        fecha VARCHAR(10),
        producto_id VARCHAR(64),
        demanda DOUBLE,
        categoria_abc VARCHAR(5),
        variabilidad VARCHAR(20),
        prediccion_media_movil DOUBLE,
        prediccion_regresion_lineal DOUBLE,
        prediccion_xgboost DOUBLE,
        prediccion_final DOUBLE,
        error_final DOUBLE,
        metodo_usado VARCHAR(40),
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS pronostico_futuro (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        producto_id VARCHAR(64),
        nombre_producto VARCHAR(200),
        categoria VARCHAR(120),
        mes_pronosticado VARCHAR(7),
        horizonte INT,
        prediccion_xgboost DOUBLE,
        prediccion_regresion_lineal DOUBLE,
        prediccion_baseline DOUBLE,
        prediccion_final DOUBLE,
        metodo_usado VARCHAR(40),
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS reorder (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        producto_id VARCHAR(64),
        nombre_producto VARCHAR(200),
        categoria VARCHAR(120),
        proveedor_principal VARCHAR(160),
        proveedor_alterno VARCHAR(160),
        categoria_abc VARCHAR(5),
        variabilidad VARCHAR(20),
        metodo_pronostico VARCHAR(40),
        demanda_promedio_mensual_pronosticada DOUBLE,
        lead_time_semanas DOUBLE,
        lead_time_meses DOUBLE,
        stock_seguridad DOUBLE,
        punto_reorden DOUBLE,
        inventario_actual DOUBLE,
        eoq DOUBLE,
        limite_presupuesto_unidades DOUBLE,
        cantidad_sugerida_pedido DOUBLE,
        costo_unitario DOUBLE,
        costo_estimado_pedido DOUBLE,
        valor_inventario_actual DOUBLE,
        fecha_estimada_pedido VARCHAR(10),
        dias_para_pedido INT,
        ordenar VARCHAR(5),
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS validacion_cruzada_folds (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        fold INT,
        meses_entrenamiento INT,
        periodo_test VARCHAR(30),
        modelo VARCHAR(40),
        mae DOUBLE,
        rmse DOUBLE,
        mape DOUBLE,
        wape DOUBLE,
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS validacion_cruzada_resumen (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        corrida_id INT NOT NULL,
        modelo VARCHAR(40),
        metrica VARCHAR(20),
        promedio DOUBLE,
        desvio DOUBLE,
        folds INT,
        FOREIGN KEY (corrida_id) REFERENCES corridas(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def crear_base_de_datos_si_no_existe():
    """Se conecta al servidor SIN seleccionar una base (algunas
    instalaciones frescas todavía no la tienen) y la crea si hace
    falta. Idempotente — si ya existe (como en esta máquina), no hace
    nada.

    En hostings gratuitos (ej. PythonAnywhere) el usuario de la base
    de datos normalmente NO tiene permiso para crear bases nuevas —
    la base ya viene creada de antemano con un nombre fijo. En ese
    caso el CREATE DATABASE falla con un error de permisos, que acá
    se ignora: si la base ya existe (que es la situación esperada en
    esos hosts) no hace falta crearla, así que seguir es seguro."""
    cfg = _config()
    url_servidor = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/?charset=utf8mb4"
    engine_servidor = create_engine(url_servidor, pool_pre_ping=True)
    try:
        with engine_servidor.begin() as conn:
            conn.execute(text(
                f"CREATE DATABASE IF NOT EXISTS {cfg['database']} "
                "CHARACTER SET utf8mb4"
            ))
    except OperationalError as error:
        if "access denied" not in str(error).lower():
            raise
    finally:
        engine_servidor.dispose()


# CREATE TABLE IF NOT EXISTS no altera una tabla que ya existe — en
# instalaciones que arrancaron antes de agregar estas columnas (como
# esta máquina), hay que sumarlas a mano. "ADD COLUMN IF NOT EXISTS"
# de MySQL no funcionó acá (da error de sintaxis pese a estar en la
# versión que debería soportarlo), así que se verifica primero contra
# information_schema — funciona en cualquier versión de MySQL.
COLUMNAS_NUEVAS = [
    ("reorder", "margen_unitario", "DOUBLE"),
    ("reorder", "costo_ruptura_estimado", "DOUBLE"),
    ("metricas_modelos", "mediana_error", "DOUBLE"),
]


def _agregar_columnas_faltantes(conn):
    cfg = _config()
    for tabla, columna, tipo in COLUMNAS_NUEVAS:
        existe = conn.execute(
            text(
                "SELECT 1 FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = :db "
                "AND TABLE_NAME = :tabla AND COLUMN_NAME = :columna"
            ),
            {"db": cfg["database"], "tabla": tabla, "columna": columna},
        ).first()
        if existe is None:
            conn.execute(text(f"ALTER TABLE {tabla} ADD COLUMN {columna} {tipo}"))


def crear_tablas():
    """Crea todas las tablas si no existen — se llama una vez al
    arrancar la app (ver app/__init__.py), igual que antes se hacía
    os.makedirs(..., exist_ok=True) para las carpetas de JSON."""
    crear_base_de_datos_si_no_existe()
    engine = get_engine()
    with engine.begin() as conn:
        for ddl in TABLAS_DDL:
            conn.execute(text(ddl))
        _agregar_columnas_faltantes(conn)
