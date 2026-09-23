"""
Persistencia del resultado del pipeline (lo que antes era
ultimo_resultado.json) en MySQL. Cada vez que se procesa un CSV se crea
una fila nueva en "corridas" — a diferencia del JSON, que se pisaba en
cada subida, acá queda historial de corridas anteriores (aunque hoy la
app solo lee la más reciente, ver cargar_ultima_corrida).

Este módulo no sabe nada de archivos en disco (CSVs, gráficas, el
modelo entrenado) — de eso se sigue encargando app/routes/main.py, que
es quien conoce las rutas de carpetas por empresa.
"""

import pandas as pd
from sqlalchemy import text

from app.services.db import get_engine
from app.services.prediccion.pronostico_futuro import NOMBRES_MES_ES

# Qué clave del dict que arma ejecutar_sistema() va en qué tabla hija.
TABLAS_RESULTADO = {
    "metricas": "metricas_modelos",
    "predicciones": "predicciones_backtest",
    "segmentacion": "segmentacion",
    "pronostico_futuro": "pronostico_futuro",
    "reorder": "reorder",
    "validacion_cruzada_detalle": "validacion_cruzada_folds",
    "validacion_cruzada_resumen": "validacion_cruzada_resumen",
}

# Los nombres de columna de "metricas" vienen de evaluar_modelo()
# (modelos.py) con mayúsculas y símbolos ("MAPE (%)") porque así se
# muestran en la tabla comparativa — no son nombres válidos de columna
# SQL, así que se renombran solo para esta tabla.
RENOMBRES_METRICAS = {
    "Modelo": "modelo", "MAE": "mae", "RMSE": "rmse", "R2": "r2",
    "Mediana Error": "mediana_error", "MAPE (%)": "mape", "WAPE (%)": "wape",
}
RENOMBRES_METRICAS_INVERSO = {v: k for k, v in RENOMBRES_METRICAS.items()}


def guardar_resultado(resultado, empresa_id):
    """Guarda el resultado completo de una corrida del pipeline.
    Devuelve el id de la corrida creada."""
    engine = get_engine()
    with engine.begin() as conn:
        fila = conn.execute(
            text(
                "INSERT INTO corridas ("
                "  empresa_id, horizonte_meses, presupuesto_capital_trabajo, presupuesto_por_producto,"
                "  mape_final, wape_final, costo_total_pedidos_urgentes, valor_total_inventario,"
                "  productos_sin_dato_inventario, productos_en_alerta, features_regresion_lineal"
                ") VALUES ("
                "  :empresa_id, :horizonte_meses, :presupuesto_capital_trabajo, :presupuesto_por_producto,"
                "  :mape_final, :wape_final, :costo_total_pedidos_urgentes, :valor_total_inventario,"
                "  :productos_sin_dato_inventario, :productos_en_alerta, :features_regresion_lineal"
                ")"
            ),
            {
                "empresa_id": empresa_id,
                "horizonte_meses": resultado.get("horizonte_meses"),
                "presupuesto_capital_trabajo": resultado.get("presupuesto_capital_trabajo"),
                "presupuesto_por_producto": resultado.get("presupuesto_por_producto"),
                "mape_final": resultado.get("mape_final"),
                "wape_final": resultado.get("wape_final"),
                "costo_total_pedidos_urgentes": resultado.get("costo_total_pedidos_urgentes"),
                "valor_total_inventario": resultado.get("valor_total_inventario"),
                "productos_sin_dato_inventario": resultado.get("productos_sin_dato_inventario"),
                "productos_en_alerta": resultado.get("productos_en_alerta"),
                "features_regresion_lineal": ",".join(resultado.get("features_regresion_lineal") or []),
            },
        )
        corrida_id = fila.lastrowid

        for clave_resultado, nombre_tabla in TABLAS_RESULTADO.items():
            registros = resultado.get(clave_resultado) or []
            if not registros:
                continue
            df = pd.DataFrame(registros)
            if clave_resultado == "metricas":
                df = df.rename(columns=RENOMBRES_METRICAS)
            if "fecha" in df.columns:
                # Timestamp de pandas -> texto, igual que antes hacía
                # json.dump(..., default=str) al guardar en el JSON.
                df["fecha"] = df["fecha"].astype(str)
            df["corrida_id"] = corrida_id
            df.to_sql(nombre_tabla, conn, if_exists="append", index=False)

    return corrida_id


def cargar_ultima_corrida(empresa_id):
    """Devuelve el resultado de la corrida más reciente de una empresa
    (mismo formato de dict que devolvía ejecutar_sistema(), sin la
    clave "archivos" — esa la agrega main.py, que es quien conoce las
    rutas de archivos en disco), o None si todavía no procesó ningún CSV."""
    engine = get_engine()
    with engine.connect() as conn:
        fila_corrida = conn.execute(
            text(
                "SELECT * FROM corridas WHERE empresa_id = :empresa_id "
                "ORDER BY fecha_corrida DESC, id DESC LIMIT 1"
            ),
            {"empresa_id": empresa_id},
        ).mappings().first()

        if fila_corrida is None:
            return None

        corrida_id = fila_corrida["id"]

        resultado = {
            "horizonte_meses": fila_corrida["horizonte_meses"],
            "presupuesto_capital_trabajo": fila_corrida["presupuesto_capital_trabajo"],
            "presupuesto_por_producto": fila_corrida["presupuesto_por_producto"],
            "mape_final": fila_corrida["mape_final"],
            "wape_final": fila_corrida["wape_final"],
            "costo_total_pedidos_urgentes": fila_corrida["costo_total_pedidos_urgentes"],
            "valor_total_inventario": fila_corrida["valor_total_inventario"],
            "productos_sin_dato_inventario": fila_corrida["productos_sin_dato_inventario"],
            "productos_en_alerta": fila_corrida["productos_en_alerta"],
            "features_regresion_lineal": (
                fila_corrida["features_regresion_lineal"].split(",")
                if fila_corrida["features_regresion_lineal"] else []
            ),
        }

        for clave_resultado, nombre_tabla in TABLAS_RESULTADO.items():
            filas = conn.execute(
                text(f"SELECT * FROM {nombre_tabla} WHERE corrida_id = :corrida_id ORDER BY id"),
                {"corrida_id": corrida_id},
            ).mappings().all()
            registros = [{k: v for k, v in f.items() if k not in ("id", "corrida_id")} for f in filas]
            if clave_resultado == "metricas":
                registros = [{RENOMBRES_METRICAS_INVERSO[k]: v for k, v in r.items()} for r in registros]
            resultado[clave_resultado] = registros

    # meses_pronosticados / meses_pronosticados_legibles / pronostico_pivot
    # ya no se guardan aparte — son una vista derivada de pronostico_futuro,
    # se reconstruyen acá con la misma lógica que pronostico_futuro.py.
    resultado["meses_pronosticados"] = _meses_ordenados(resultado["pronostico_futuro"])
    resultado["meses_pronosticados_legibles"] = _meses_legibles_ordenados(resultado["pronostico_futuro"])
    resultado["pronostico_pivot"] = _armar_pronostico_pivot(resultado["pronostico_futuro"])

    return resultado


def _mes_por_horizonte(registros_pronostico):
    """{horizonte: 'YYYY-MM'} — todas las filas de un mismo horizonte
    comparten el mismo mes (una por producto)."""
    return {r["horizonte"]: r["mes_pronosticado"] for r in registros_pronostico}


def _meses_ordenados(registros_pronostico):
    por_horizonte = _mes_por_horizonte(registros_pronostico)
    return [por_horizonte[h] for h in sorted(por_horizonte)]


def _meses_legibles_ordenados(registros_pronostico):
    por_horizonte = _mes_por_horizonte(registros_pronostico)
    legibles = []
    for h in sorted(por_horizonte):
        anio, mes = por_horizonte[h].split("-")
        legibles.append(f"{NOMBRES_MES_ES[int(mes)]} {anio}")
    return legibles


def _armar_pronostico_pivot(registros_pronostico):
    por_producto = {}
    for r in registros_pronostico:
        por_producto.setdefault(r["producto_id"], []).append(r)

    pivot = []
    for producto_id, filas in por_producto.items():
        filas_ordenadas = sorted(filas, key=lambda f: f["horizonte"])
        valores_crudos = [f["prediccion_final"] for f in filas_ordenadas]
        pivot.append({
            "producto_id": producto_id,
            "nombre_producto": filas_ordenadas[0]["nombre_producto"],
            "categoria": filas_ordenadas[0]["categoria"],
            "valores": [round(v) for v in valores_crudos],
            "total": round(sum(valores_crudos)),
        })
    pivot.sort(key=lambda f: f["total"], reverse=True)
    return pivot
