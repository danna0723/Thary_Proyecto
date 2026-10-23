import numpy as np
import pandas as pd

from app.services.prediccion.utils import nan_a_none
from app.services.prediccion.features import incorporar_estadisticas_producto
from app.services.prediccion.modelos import evaluar_modelo

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


# EXTENSIÓN PROPIA (corrección D — estados del error histórico) — el
# margen del rango sale del MAE de backtest, y ese MAE puede estar en tres
# situaciones distintas que la interfaz no debe confundir:
#   - "medido": hubo backtest y el método tuvo error; el margen es ese error.
#   - "medido en cero": hubo backtest y el error fue exactamente 0. Con muy
#     pocos meses de prueba, esto suele significar muestra insuficiente, no
#     un modelo perfecto.
#   - "no disponible": el producto no tiene backtest (historia corta). No hay
#     margen medido y NO se puede presentar como un rango estrecho ni como
#     confiable.
ESTADO_ERROR_MEDIDO = "medido"
ESTADO_ERROR_CERO = "medido en cero"
ESTADO_ERROR_NO_DISPONIBLE = "no disponible"


def estado_error_historico(margen):
    if margen is None or pd.isna(margen):
        return ESTADO_ERROR_NO_DISPONIBLE
    if margen == 0:
        return ESTADO_ERROR_CERO
    return ESTADO_ERROR_MEDIDO


def estado_error_del_producto(estados):
    """Resumen de los meses de un producto: si algún mes no tiene error
    medido, el producto entero queda como "no disponible" (no se puede
    decir que está bien medido); si todos están en cero, "medido en cero";
    en otro caso, "medido". Sin meses, None."""
    if not estados:
        return None
    if ESTADO_ERROR_NO_DISPONIBLE in estados:
        return ESTADO_ERROR_NO_DISPONIBLE
    if all(e == ESTADO_ERROR_CERO for e in estados):
        return ESTADO_ERROR_CERO
    return ESTADO_ERROR_MEDIDO


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

            # Solo inventario_lag_1 (inventario del mes anterior); el del mismo
            # mes no es variable (ver features.py). Para horizontes > 1 se usa el
            # último inventario conocido: aproximación, no un valor pronosticado.
            if "inventario_lag_1" in feature_cols:
                fila_features["inventario_lag_1"] = ultima_fila.get("inventario", 0)

            filas_features.append(fila_features)
            metadatos.append((producto, media_movil_3))

        # Mismo tratamiento de faltantes que en el entrenamiento (features.py
        # rellena NaN con 0): un producto sin dato de inventario en su último
        # mes no debe romper la predicción de los modelos que usan esa variable.
        X_futuro_lote = pd.DataFrame(filas_features)[feature_cols].fillna(0)
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
                "estado_error": estado_error_historico(margen),
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
            estados_error = grupo["estado_error"].tolist()
            pronostico_pivot.append({
                "producto_id": producto,
                "nombre_producto": grupo["nombre_producto"].iloc[0],
                "categoria": grupo["categoria"].iloc[0],
                # Estado del error histórico por mes (ver estado_error_historico)
                # y resumen del producto (ver estado_error_del_producto).
                "estados_error": estados_error,
                "estado_error": estado_error_del_producto(estados_error),
                # D3: sin margen medido el rango no tiene ancho, pero eso NO es
                # certeza: se marca aparte para que la interfaz no lo muestre
                # como confiable (es_baja_confianza no cambia su definición).
                "sin_margen_por_mes": [e == ESTADO_ERROR_NO_DISPONIBLE for e in estados_error],
                "sin_margen": ESTADO_ERROR_NO_DISPONIBLE in estados_error,
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


# ============================================================
# EVALUACIÓN DEL PRONÓSTICO RECURSIVO POR HORIZONTE (corrección A.4)
# ============================================================
# EXTENSIÓN PROPIA — la evaluación de backtest de modelos.py mide cada
# mes con la demanda REAL del mes anterior (un paso). El pronóstico que
# recibe el usuario es distinto: se genera desde el último mes conocido y
# cada mes se alimenta con la predicción del anterior. Esta función mide
# ese mismo procedimiento: pronostica desde UN origen con la historia
# hasta ese origen, y compara contra la demanda real de los meses que
# siguen, separando el error por horizonte (mes 1, mes 2, ...).
#
# No reutiliza la tabla de backtest: esa mezcla meses de origen distintos
# y no reproduce lo que pasa al pronosticar varios meses seguidos.
def evaluar_pronostico_recursivo(
    df_mensual, producto_comportamiento, model, feature_cols, modelo_lr, features_lr,
    origen, horizonte_meses,
):
    """
    origen: último mes con historia que el pronóstico puede usar (todo lo
    posterior se trata como desconocido, incluidas las estadísticas por
    producto, que se recalculan solo con la historia hasta ese mes).

    Devuelve (comparacion, productos_sin_pronostico):
      - comparacion: una fila por (producto, horizonte) con la demanda real
        y las predicciones del sistema final y de la media móvil recursiva.
      - productos_sin_pronostico: cuántos productos con demanda real en el
        período no se pudieron pronosticar (historia menor a 3 meses, ver
        generar_pronostico_futuro). Se reporta para que no se cuenten en
        silencio como si no existieran.
    """
    historia = df_mensual[df_mensual["fecha"] <= origen]
    historia = incorporar_estadisticas_producto(historia, historia, fecha_corte=None)

    pronostico, _, _, _ = generar_pronostico_futuro(
        historia, producto_comportamiento, model, feature_cols, modelo_lr, features_lr,
        horizonte_meses=horizonte_meses,
    )

    fin = origen + pd.DateOffset(months=horizonte_meses)
    real = df_mensual[(df_mensual["fecha"] > origen) & (df_mensual["fecha"] <= fin)].copy()
    real["mes_pronosticado"] = real["fecha"].dt.strftime("%Y-%m")

    productos_reales = set(real["producto_id"])
    productos_pronosticados = set(pronostico["producto_id"]) if not pronostico.empty else set()
    productos_sin_pronostico = len(productos_reales - productos_pronosticados)

    if pronostico.empty:
        return pd.DataFrame(columns=["producto_id", "horizonte", "demanda", "prediccion_final", "prediccion_baseline"]), productos_sin_pronostico

    comparacion = pronostico.merge(
        real[["producto_id", "mes_pronosticado", "demanda"]],
        on=["producto_id", "mes_pronosticado"], how="inner",
    )
    return comparacion[["producto_id", "horizonte", "demanda", "prediccion_final", "prediccion_baseline"]], productos_sin_pronostico


def resumen_pronostico_por_horizonte(comparacion):
    """
    Una fila por horizonte con el error del sistema final y de la media
    móvil recursiva (MAE y WAPE), sobre los productos comparados en ese
    horizonte. Lista de dicts lista para guardar o mostrar.
    """
    filas = []
    for horizonte, grupo in comparacion.groupby("horizonte"):
        real = grupo["demanda"].values
        sistema = evaluar_modelo(real, grupo["prediccion_final"].values, "Sistema final")
        baseline = evaluar_modelo(real, grupo["prediccion_baseline"].values, "Media móvil")
        filas.append({
            "horizonte": int(horizonte),
            "productos": int(grupo["producto_id"].nunique()),
            "mae_sistema": round(float(sistema["MAE"]), 2),
            "wape_sistema": None if pd.isna(sistema["WAPE (%)"]) else round(float(sistema["WAPE (%)"]), 2),
            "mae_media_movil": round(float(baseline["MAE"]), 2),
            "wape_media_movil": None if pd.isna(baseline["WAPE (%)"]) else round(float(baseline["WAPE (%)"]), 2),
        })
    return filas
