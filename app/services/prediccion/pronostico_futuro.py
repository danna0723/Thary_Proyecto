import numpy as np
import pandas as pd

from app.services.prediccion.utils import nan_a_none

NOMBRES_MES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

HORIZONTE_MESES_DEFAULT = 3
HORIZONTE_MESES_MIN = 1
HORIZONTE_MESES_MAX = 12

# EXTENSIÓN PROPIA — en vez de mostrar solo un número cerrado, el
# pronóstico futuro también reporta un rango [mínimo, máximo] alrededor
# de cada predicción puntual. El margen es el MAE que tuvo, en el
# backtest, el método que termina usándose para ese producto (ver
# MAE_POR_METODO_COL más abajo) — "en el pasado reciente, este método se
# equivocó en promedio esto para este producto". MARGEN_RANGO_MULTIPLICADOR
# escala ese margen (1.0 = ±1 MAE a cada lado); es un criterio propio,
# sin respaldo bibliográfico, pensado para dar una idea simple y barata
# de incertidumbre sin entrenar modelos adicionales (a diferencia de un
# enfoque de regresión por cuantiles, que sí requeriría eso). El rango no
# crece con el horizonte (sería más honesto que lo hiciera, ya que el
# pronóstico es recursivo y el error puede acumularse — ver el
# comentario sobre HORIZONTE_MESES_MAX más abajo), queda como mejora futura.
MARGEN_RANGO_MULTIPLICADOR = 1.0

# Qué columna de producto_comportamiento (agregada en
# sistema_prediccion.py a partir de elegir_campeon_por_producto) tiene el
# MAE de cada método, para poder elegir la que corresponde al
# metodo_usado real de cada producto-mes (que puede no ser el método
# campeón, si actuó la salvaguarda de más abajo).
MAE_POR_METODO_COL = {
    "Media móvil": "mae_media_movil",
    "Regresión Lineal": "mae_regresion_lineal",
    "XGBoost": "mae_xgboost",
}


def calcular_rango_prediccion(prediccion, margen):
    """
    Rango [mínimo, máximo] alrededor de una predicción puntual, dado un
    margen (el MAE del método usado, en el backtest, para ese producto).
    Un margen nulo o NaN (producto sin datos de backtest) da un rango sin
    ancho, igual al valor puntual — no se inventa una incertidumbre que
    no se midió. El mínimo nunca baja de 0 (no hay demanda negativa).
    """
    if margen is None or (isinstance(margen, float) and np.isnan(margen)):
        margen = 0.0
    margen = margen * MARGEN_RANGO_MULTIPLICADOR
    return max(0.0, prediccion - margen), prediccion + margen


# EXTENSIÓN PROPIA — bandera de "baja confianza" para el panel del
# usuario: se activa cuando el mínimo del rango (arriba) quedó
# recortado en 0. Eso pasa exactamente cuando el margen de error (MAE
# de backtest del método usado) es igual o mayor que la predicción
# misma — el modelo se equivocó, en el pasado, al menos tanto como el
# valor que está prediciendo ahora, así que ese número puntual no es
# confiable por sí solo (típico de productos de demanda intermitente,
# casi siempre cero con picos esporádicos — ningún método, de este
# sistema o de cualquier otro, predice bien CUÁNDO cae el próximo
# pico). Sin margen medido (producto sin backtest, o con error cero
# medido) el mínimo no se recorta, así que tampoco se marca como baja
# confianza: ausencia de margen no es lo mismo que mucho margen.
def es_baja_confianza(prediccion, minimo):
    return prediccion > 0 and minimo <= 0


# ============================================================
# PRONÓSTICO REAL DE LOS PRÓXIMOS N MESES (recursivo)
# ============================================================
# EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de
# referencia del proyecto (ver conversación de metodología). Ningún
# ejemplo pronostica varios pasos hacia adelante de forma recursiva;
# esta sección extiende el enfoque de un solo paso (Ejemplo 1/3) a un
# horizonte de varios meses, elegido por el usuario al subir el archivo
# (ver "horizonte" en el formulario de app/templates/index.html).
#
# A diferencia de la evaluación de backtest (que mide el modelo contra
# meses que YA pasaron), esta función pronostica los meses SIGUIENTES
# al último mes disponible en el CSV. Como esos meses todavía no
# ocurrieron, no hay demanda real con la cual comparar — por eso no se
# calcula ningún error acá.
#
# Es un pronóstico RECURSIVO (multi-step): la predicción del mes 1 se
# usa como si fuera dato real para calcular los lags del mes 2, la del
# mes 2 para el mes 3, etc. Es el enfoque estándar para extender un
# modelo de un paso a varios pasos sin tener que entrenar un modelo
# distinto por horizonte (con ~12 meses de historia, partir los datos
# en más subconjuntos los haría aún más chicos). El costo es que el
# error se puede ir acumulando de un mes a otro — y ese costo crece con
# el horizonte, por eso HORIZONTE_MESES_MAX pone un techo razonable.
def generar_pronostico_futuro(
    df_mensual, producto_comportamiento, model, feature_cols, modelo_lr, features_lr,
    horizonte_meses=HORIZONTE_MESES_DEFAULT,
):
    """
    modelo_lr/features_lr: el modelo de Regresión Lineal ya entrenado
    (regresion_lineal.py) y las variables que usa — se necesitan acá
    porque "metodo_campeon" (producto_comportamiento, calculado en
    modelos.py::elegir_campeon_por_producto) puede ser "Regresión
    Lineal" para algunos productos, no solo XGBoost o Media Móvil.

    horizonte_meses: cuántos meses hacia adelante pronosticar, elegido
    por el usuario (acotado entre HORIZONTE_MESES_MIN y
    HORIZONTE_MESES_MAX por quien llama a esta función — ver
    sistema_prediccion.py).

    Devuelve (pronostico_futuro_df, pronostico_pivot,
    meses_pronosticados, meses_pronosticados_legibles).
    """
    proximo_mes = df_mensual["fecha"].max() + pd.DateOffset(months=1)
    meses_pronosticados = [
        (proximo_mes + pd.DateOffset(months=h)).strftime("%Y-%m")
        for h in range(horizonte_meses)
    ]
    meses_pronosticados_legibles = [
        f"{NOMBRES_MES_ES[(proximo_mes + pd.DateOffset(months=h)).month]} "
        f"{(proximo_mes + pd.DateOffset(months=h)).year}"
        for h in range(horizonte_meses)
    ]

    # Preparar el estado de cada producto UNA sola vez (no en cada
    # horizonte): historial de demandas, última fila de df_mensual y su
    # fila de segmentación. Se arma con groupby/set_index en vez de
    # filtrar df_mensual/producto_comportamiento con una máscara booleana
    # por cada producto (esto era O(n²) — con 856 productos, cientos de
    # miles de comparaciones de más solo para armar el estado inicial).
    producto_comportamiento_idx = producto_comportamiento.set_index("producto_id")
    estado_por_producto = {}
    for producto, historial_producto in df_mensual.sort_values("fecha").groupby("producto_id"):
        demandas = list(historial_producto["demanda"].values)
        if len(demandas) < 3:
            # No hay suficiente historial para calcular lag_1..lag_3
            continue
        estado_por_producto[producto] = {
            "demandas": demandas,
            "ultima_fila": historial_producto.iloc[-1],
            "segmento": producto_comportamiento_idx.loc[producto],
        }

    productos_validos = sorted(estado_por_producto.keys())

    pronostico_futuro = []

    # Recursivo, pero por HORIZONTE en vez de por producto: en cada paso
    # se arma un lote con la fila de features de los ~800+ productos y
    # se llama a model.predict() UNA vez para todo el lote, en lugar de
    # una vez por producto (2568 llamadas individuales a XGBoost, cada
    # una con su propio overhead de por medio, es la parte más lenta de
    # todo el pipeline — pasar a predicción por lotes lo resuelve sin
    # cambiar ningún cálculo, cada producto se sigue prediciendo con
    # exactamente los mismos datos que antes, solo que juntos).
    for h in range(horizonte_meses):
        mes_objetivo = proximo_mes + pd.DateOffset(months=h)

        filas_features = []
        metadatos = []

        for producto in productos_validos:
            estado = estado_por_producto[producto]
            demandas = estado["demandas"]
            ultima_fila = estado["ultima_fila"]

            lag_1 = demandas[-1]
            lag_2 = demandas[-2]
            lag_3 = demandas[-3]
            lag_6 = demandas[-6] if len(demandas) >= 6 else 0

            media_movil_3 = np.mean(demandas[-3:])
            media_movil_6 = np.mean(demandas[-6:]) if len(demandas) >= 6 else np.mean(demandas)
            desviacion_movil_3 = np.std(demandas[-3:])

            fila_features = {
                "año": mes_objetivo.year,
                "mes_sin": np.sin(2 * np.pi * mes_objetivo.month / 12),
                "mes_cos": np.cos(2 * np.pi * mes_objetivo.month / 12),
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_6": lag_6,
                "media_movil_3": media_movil_3,
                "media_movil_6": media_movil_6,
                "desviacion_movil_3": desviacion_movil_3,
                "media_producto": ultima_fila.get("media_producto", 0),
                "desviacion_producto": ultima_fila.get("desviacion_producto", 0),
            }

            if "precio" in feature_cols:
                fila_features["precio"] = ultima_fila.get("precio", 0)

            if "inventario" in feature_cols:
                fila_features["inventario"] = ultima_fila.get("inventario", 0)
                if "inventario_lag_1" in feature_cols:
                    fila_features["inventario_lag_1"] = ultima_fila.get("inventario", 0)

            filas_features.append(fila_features)
            metadatos.append((producto, media_movil_3))

        X_futuro_lote = pd.DataFrame(filas_features)[feature_cols]
        preds_xgb_lote = np.maximum(np.expm1(model.predict(X_futuro_lote)), 0)
        preds_lr_lote = np.maximum(modelo_lr.predict(X_futuro_lote[features_lr]), 0)

        for (producto, media_movil_3), pred_xgb_futuro, pred_lr_futuro in zip(metadatos, preds_xgb_lote, preds_lr_lote):
            estado = estado_por_producto[producto]
            segmento = estado["segmento"]

            pred_xgb_futuro = float(pred_xgb_futuro)
            pred_lr_futuro = float(pred_lr_futuro)
            pred_baseline_futuro = float(max(media_movil_3, 0))

            predicciones_por_metodo = {
                "Media móvil": pred_baseline_futuro,
                "Regresión Lineal": pred_lr_futuro,
                "XGBoost": pred_xgb_futuro,
            }

            # Salvaguarda: en un pronóstico recursivo el error se puede
            # ir acumulando de un mes a otro (a diferencia del backtest,
            # acá no hay demanda real contra la cual verificar). Si el
            # método campeón de este producto es un modelo (no la Media
            # Móvil) y su predicción se desploma a menos de la mitad del
            # nivel reciente, no se confía en ese número — se usa la
            # Media Móvil para ese mes en su lugar.
            metodo_campeon = segmento["metodo_campeon"]
            if (
                metodo_campeon != "Media móvil"
                and media_movil_3 > 0
                and predicciones_por_metodo[metodo_campeon] < media_movil_3 * 0.5
            ):
                metodo_usado = "Media móvil"
            else:
                metodo_usado = metodo_campeon

            pred_final_futuro = predicciones_por_metodo[metodo_usado]

            # Rango alrededor de la predicción final, según qué tanto se
            # equivocó en el backtest el método que se terminó usando
            # para este producto (ver calcular_rango_prediccion).
            margen = segmento.get(MAE_POR_METODO_COL[metodo_usado])
            pred_min_futuro, pred_max_futuro = calcular_rango_prediccion(pred_final_futuro, margen)

            pronostico_futuro.append({
                "producto_id": producto,
                "nombre_producto": segmento.get("nombre_producto"),
                "categoria": segmento.get("categoria"),
                "mes_pronosticado": mes_objetivo.strftime("%Y-%m"),
                "horizonte": h + 1,
                "prediccion_xgboost": round(pred_xgb_futuro, 2),
                "prediccion_regresion_lineal": round(pred_lr_futuro, 2),
                "prediccion_baseline": round(pred_baseline_futuro, 2),
                "prediccion_final": round(pred_final_futuro, 2),
                "prediccion_min": round(pred_min_futuro, 2),
                "prediccion_max": round(pred_max_futuro, 2),
                "metodo_usado": metodo_usado,
            })

            # La predicción de este mes alimenta los lags del siguiente
            estado["demandas"].append(pred_final_futuro)

    pronostico_futuro_df = pd.DataFrame(pronostico_futuro)
    # Ver utils.nan_a_none: reconstruir el DataFrame desde una lista de
    # diccionarios puede convertir a NaN los None de nombre_producto/categoria.
    pronostico_futuro_df = nan_a_none(pronostico_futuro_df, ["nombre_producto", "categoria"])

    pronostico_pivot = []
    if not pronostico_futuro_df.empty:
        for producto, grupo in pronostico_futuro_df.sort_values("horizonte").groupby("producto_id"):
            valores = grupo["prediccion_final"].tolist()
            minimos = grupo["prediccion_min"].tolist()
            maximos = grupo["prediccion_max"].tolist()
            confianzas_bajas = [es_baja_confianza(v, mn) for v, mn in zip(valores, minimos)]
            pronostico_pivot.append({
                "producto_id": producto,
                "nombre_producto": grupo["nombre_producto"].iloc[0],
                "categoria": grupo["categoria"].iloc[0],
                # Redondeado a unidades enteras para la vista del usuario
                # final: fracciones de unidad ("45.63") no aportan nada
                # al que tiene que decidir cuánto pedir.
                "valores": [round(v) for v in valores],
                # Rango [mínimo, máximo] de cada mes, mismo orden que
                # "valores" — ver MARGEN_RANGO_MULTIPLICADOR más arriba.
                "rangos": [[round(mn), round(mx)] for mn, mx in zip(minimos, maximos)],
                # Baja confianza por mes (ver es_baja_confianza) — el total
                # hereda la bandera de cualquiera de sus meses, porque ese
                # total ya incluye un número poco confiable.
                "confianzas_bajas": confianzas_bajas,
                "total": round(sum(valores)),
                "total_min": round(sum(minimos)),
                "total_max": round(sum(maximos)),
                "total_baja_confianza": any(confianzas_bajas),
            })
        pronostico_pivot.sort(key=lambda fila: fila["total"], reverse=True)

    return pronostico_futuro_df, pronostico_pivot, meses_pronosticados, meses_pronosticados_legibles
