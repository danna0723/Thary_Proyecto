import numpy as np
import pandas as pd

from app.services.prediccion.pronostico_futuro import MAE_POR_METODO_COL, estado_error_historico
from app.services.prediccion.utils import nan_a_none

SERVICE_LEVEL_Z = 1.65
LEAD_TIME_DEFAULT_MESES = 0.5
WORKING_CAPITAL_BUDGET_DEFAULT = 27_000_000
DIAS_POR_MES = 30.44

# Techo de seguridad para "días hasta que haga falta pedir": muy por
# encima de los 90 días que usa la pantalla de presupuesto para
# "próximos pedidos" (así nunca cambia qué se le muestra al usuario),
# pero muy por debajo del límite que soporta pd.Timedelta (~106.751
# días). Sin este techo, un producto con demanda pronosticada muy
# cercana a cero puede dar una cantidad de días astronómica —
# matemáticamente "correcta" pero sin sentido práctico— que desborda
# pd.Timedelta y frena el procesamiento de todo el archivo.
DIAS_PARA_PEDIDO_TECHO = 36_500  # 100 años


# ============================================================
# PUNTOS DE REORDEN Y EOQ
# ============================================================
# Ejemplo 1 (SKU Demand Forecasting, sección "Inventory Reorder
# Recommendations") + Ejemplo 2 (Samir Saci, reglas de reposición
# basadas en EOQ considerando la variabilidad de la demanda): mismas
# fórmulas de Safety Stock / Reorder Point / EOQ / cantidad acotada por
# presupuesto (con las variables renombradas al español pero la misma
# estructura matemática — ver comentarios línea a línea más abajo).
#
# El promedio y la desviación de demanda para el punto de reorden se
# calculan sobre el pronóstico futuro (el horizonte que haya elegido el
# usuario al subir el archivo), no sobre el backtest histórico: para
# decidir cuánto pedir hoy interesa lo que se espera vender, no lo bien
# que el modelo acertó en el pasado.
#
# ----------------------------------------------------------------------
# POLÍTICA DE CANTIDAD (corrección P1-B — supuestos a validar con el
# gestor de inventarios; no son valores bibliográficos):
#
#   1. Alerta: se ordena cuando inventario <= punto de reorden.
#   2. Cantidad ideal, en este orden de preferencia:
#        a. EOQ, si hay datos de costo para calcularlo.
#        b. punto de reorden - inventario, si es positivo.
#        c. Si hay alerta pero esa diferencia es 0 (inventario exactamente
#           en el punto de reorden): un mes de demanda pronosticada, para
#           que la alerta nunca salga con cantidad 0 sin explicación.
#        d. Sin alerta y sin EOQ: 0 (no hay nada que pedir todavía).
#   3. Presupuesto: se asigna por prioridad, no en partes iguales. Orden:
#      productos con alerta; luego los que se agotarían antes de que llegue
#      el pedido; luego clase ABC (A, B, C); luego menos días para llegar al
#      punto de reorden. Cada producto recibe lo que alcance con
#      el dinero que queda; lo que no alcanza se reporta como "sin cubrir"
#      en vez de quedar oculto.
#   4. Productos sin costo unitario no consumen presupuesto (no se pueden
#      valorar) y conservan su cantidad ideal.
#
# La regla 2.c y la prioridad de la regla 3 son propuestas del desarrollo
# y deben acordarse con el gestor antes de presentarse como la política
# de la clínica.
# ----------------------------------------------------------------------
def calcular_tabla_reorden(
    df_model, df_mensual, producto_comportamiento, pronostico_futuro_df,
    presupuesto_capital_trabajo=None
):
    """
    presupuesto_capital_trabajo: monto total disponible para comprar
    inventario, ingresado por el usuario en el formulario de subida. Si
    no se especifica (None), se usa WORKING_CAPITAL_BUDGET_DEFAULT como
    respaldo — el mismo comportamiento que tenía el sistema antes de que
    el usuario pudiera ingresar su propio presupuesto.

    Devuelve (reorder_df, working_capital_budget, presupuesto_por_producto).
    presupuesto_por_producto (partes iguales) se sigue devolviendo como
    referencia informativa; ya no decide cuánto se asigna a cada producto.
    """
    working_capital_budget = (
        presupuesto_capital_trabajo
        if presupuesto_capital_trabajo is not None and presupuesto_capital_trabajo > 0
        else WORKING_CAPITAL_BUDGET_DEFAULT
    )
    n_productos_activos = df_model["producto_id"].nunique()
    presupuesto_por_producto = working_capital_budget / n_productos_activos

    # "Hoy" para este cálculo es el último mes con datos del CSV (que es
    # de cuándo es el inventario_actual que se está usando), no la fecha
    # real del sistema — así la fecha estimada de pedido es coherente
    # con la fecha del inventario, sin importar cuándo se corra el
    # sistema ni qué tan viejo sea el archivo subido.
    fecha_referencia = df_mensual["fecha"].max()

    # Precalculado UNA vez fuera del loop (en vez de filtrar cada
    # DataFrame por producto en cada una de las 856 iteraciones, que es
    # O(n²) y era buena parte del tiempo que tardaba esta sección):
    # estadísticas del pronóstico por producto, última fila de
    # df_mensual por producto, y producto_comportamiento indexado.
    stats_pronostico = pronostico_futuro_df.groupby("producto_id")["prediccion_final"].agg(
        demanda_promedio="mean", desviacion_pronostico_std=lambda x: np.std(x)
    )
    ultima_fila_por_producto = (
        df_mensual.sort_values("fecha").groupby("producto_id").tail(1).set_index("producto_id")
    )
    producto_comportamiento_idx = producto_comportamiento.set_index("producto_id")

    filas = []
    for producto in sorted(df_model["producto_id"].unique()):
        if producto not in stats_pronostico.index:
            continue
        demanda_promedio = stats_pronostico.at[producto, "demanda_promedio"]
        fila_params = ultima_fila_por_producto.loc[producto]
        segmento = producto_comportamiento_idx.loc[producto]

        lead_time_meses = (
            fila_params["lead_time_semanas"] / 4.345
            if "lead_time_semanas" in df_mensual.columns and not pd.isna(fila_params.get("lead_time_semanas"))
            else LEAD_TIME_DEFAULT_MESES
        )
        costo_pedido_producto = fila_params.get("costo_pedido", np.nan)
        porcentaje_mantenimiento = fila_params.get("porcentaje_costo_mantenimiento", np.nan)
        costo_unitario_producto = fila_params.get("costo_unitario", np.nan)

        # EXTENSIÓN PROPIA (corrección) — sigma_demanda tiene que ser qué
        # tanto se EQUIVOCA el pronóstico, no qué tanto varían los valores
        # pronosticados entre sí. Antes se usaba np.std(prediccion_final)
        # a lo largo del horizonte: con un método recursivo como Media
        # Móvil, los pocos meses pronosticados suelen salir parecidos
        # entre sí (la predicción de un mes alimenta la del siguiente), así
        # que ese desvío queda chico incluso para productos con demanda
        # muy errática — p. ej. en el catálogo clínico, un producto con
        # MAE de backtest de 29.33 (la columna "¿Por qué algunos productos
        # tienen baja confianza?") daba un stock de seguridad de apenas
        # 3.77, calculado así. Ahora se usa directamente el MAE de
        # backtest del método campeón (mae_media_movil/..., ya calculado
        # en elegir_campeon_por_producto) — la misma medida de qué tanto
        # se equivoca el pronóstico que ya se usa para el rango de
        # predicción, en vez de una segunda noción de "variabilidad"
        # distinta e inconsistente con esa. Sin MAE medido (producto sin
        # backtest) se cae al criterio anterior, para no dejar el stock de
        # seguridad sin ningún valor — y la fuente queda registrada en
        # "fuente_sigma" para que la interfaz pueda distinguirlo.
        metodo_campeon = segmento["metodo_campeon"]
        mae_campeon = segmento.get(MAE_POR_METODO_COL.get(metodo_campeon))
        if mae_campeon is not None and not pd.isna(mae_campeon):
            desviacion_demanda = mae_campeon
            fuente_sigma = "MAE de backtest"
        else:
            desviacion_demanda = stats_pronostico.at[producto, "desviacion_pronostico_std"]
            fuente_sigma = "desvío del pronóstico (sin backtest)"

        # Ejemplo 1: Safety Stock = Z * sigma_demanda * sqrt(L)
        stock_seguridad = SERVICE_LEVEL_Z * desviacion_demanda * np.sqrt(lead_time_meses)
        # Ejemplo 1: Reorder Point = d_promedio * L + Safety Stock
        punto_reorden = demanda_promedio * lead_time_meses + stock_seguridad
        inventario_actual = fila_params.get("inventario", np.nan)

        # EXTENSIÓN PROPIA — mismo punto de reorden, pero asumiendo que la
        # demanda durante el lead time sigue una distribución LOG-NORMAL en
        # vez de normal (ver stock_seguridad arriba). Es solo informativa:
        # no reemplaza stock_seguridad/punto_reorden (que siguen siendo los
        # que usa el sistema para decidir "ordenar" y la cantidad sugerida),
        # se calcula para comparar ambos supuestos en el panel técnico.
        #
        # La normal es simétrica y, con suficiente variabilidad, permite
        # matemáticamente demanda negativa (algo imposible en la realidad);
        # la log-normal, al asumir que ln(demanda) es normal, respeta que la
        # demanda nunca es negativa y captura mejor la cola derecha típica
        # de productos de rotación baja/intermitente — el caso de hasta el
        # 36% del catálogo clínico clasificado como variabilidad Media/Alta.
        #
        # Mismo criterio de ajuste por momentos que para la normal: se parte
        # de la media y desviación del lead time ya calculadas arriba
        # (mean_lt, std_lt), y se despeja mu/sigma de la log-normal que
        # reproducen ese mismo CV (moment matching), en vez de ajustar la
        # distribución a los datos crudos del backtest con muy pocos meses.
        media_lt = demanda_promedio * lead_time_meses
        desviacion_lt = desviacion_demanda * np.sqrt(lead_time_meses)
        if media_lt > 0 and desviacion_lt > 0:
            cv_lt = desviacion_lt / media_lt
            sigma_log = np.sqrt(np.log(1 + cv_lt ** 2))
            mu_log = np.log(media_lt) - (sigma_log ** 2) / 2
            punto_reorden_lognormal = np.exp(mu_log + SERVICE_LEVEL_Z * sigma_log)
            stock_seguridad_lognormal = punto_reorden_lognormal - media_lt
        elif media_lt > 0:
            # Sin variabilidad (desviacion_lt == 0), no hace falta stock de
            # seguridad bajo ningún supuesto — coincide con la normal.
            punto_reorden_lognormal = media_lt
            stock_seguridad_lognormal = 0.0
        else:
            # Sin demanda pronosticada no hay distribución que asumir.
            punto_reorden_lognormal = np.nan
            stock_seguridad_lognormal = np.nan

        # Ejemplo 1: EOQ = sqrt(2 * D_anual * costo_pedido / costo_mantenimiento)
        demanda_anual_estimada = demanda_promedio * 12
        if not pd.isna(costo_unitario_producto) and not pd.isna(porcentaje_mantenimiento) and not pd.isna(costo_pedido_producto):
            costo_mantenimiento_unitario = costo_unitario_producto * porcentaje_mantenimiento
            eoq = np.sqrt((2 * demanda_anual_estimada * costo_pedido_producto) / costo_mantenimiento_unitario) if costo_mantenimiento_unitario > 0 else np.nan
        else:
            eoq = np.nan

        tiene_costo = not pd.isna(costo_unitario_producto) and costo_unitario_producto > 0
        hay_inventario = not pd.isna(inventario_actual)
        hay_alerta = hay_inventario and inventario_actual <= punto_reorden

        # Cantidad ideal, antes de considerar el presupuesto (ver la
        # política al inicio de la función).
        if not pd.isna(eoq):
            cantidad_ideal = eoq
            regla_cantidad = "EOQ"
        elif hay_inventario:
            diferencia = punto_reorden - inventario_actual
            if diferencia > 0:
                cantidad_ideal = diferencia
                regla_cantidad = "punto de reorden - inventario"
            elif hay_alerta and demanda_promedio > 0:
                cantidad_ideal = demanda_promedio
                regla_cantidad = "un mes de demanda (inventario en el punto de reorden)"
            else:
                cantidad_ideal = 0.0
                regla_cantidad = "sin necesidad de pedido"
        else:
            cantidad_ideal = np.nan
            regla_cantidad = "sin dato de inventario"

        # Costo de ruptura de inventario (stockout) y precio: sin cambios.
        precio_producto = fila_params.get("precio", np.nan)
        demanda_diaria = demanda_promedio / DIAS_POR_MES
        lead_time_dias = lead_time_meses * DIAS_POR_MES
        if (
            not pd.isna(precio_producto) and not pd.isna(costo_unitario_producto)
            and precio_producto > costo_unitario_producto
        ):
            margen_unitario = precio_producto - costo_unitario_producto
            costo_ruptura_estimado = margen_unitario * demanda_diaria * lead_time_dias
        else:
            margen_unitario = np.nan
            costo_ruptura_estimado = np.nan

        # Cobertura durante el lead time: si se pidiera hoy, cuánto stock
        # quedaría cuando llegue el pedido (inventario menos la demanda
        # esperada durante la entrega). Si es negativo, el producto se
        # agota antes de que llegue. Es la pregunta de la tabla del informe
        # "¿qué cobertura ofrece la cantidad durante y después de la entrega?".
        if hay_inventario:
            stock_al_llegar_pedido = inventario_actual - demanda_promedio * lead_time_meses
            riesgo_ruptura_durante_entrega = stock_al_llegar_pedido < 0
        else:
            stock_al_llegar_pedido = np.nan
            riesgo_ruptura_durante_entrega = None

        # Fecha estimada en la que el inventario va a llegar al punto de
        # reorden, asumiendo que se consume al ritmo promedio pronosticado
        # (una aproximación lineal — no sabe de picos puntuales dentro del
        # mes, pero da una fecha concreta en vez de solo un umbral de
        # cantidad). Si ya está en o por debajo del punto de reorden, la
        # fecha es "ya" (hoy mismo). Si no hay demanda pronosticada, no se
        # puede estimar cuándo se va a agotar.
        ya_hay_que_pedir = hay_alerta
        if not hay_inventario:
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan
        elif ya_hay_que_pedir:
            fecha_estimada_pedido = fecha_referencia
            dias_para_pedido = 0
        elif demanda_promedio > 0:
            dias_para_pedido = min(
                (inventario_actual - punto_reorden) / demanda_diaria,
                DIAS_PARA_PEDIDO_TECHO,
            )
            fecha_estimada_pedido = fecha_referencia + pd.Timedelta(days=dias_para_pedido)
        else:
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan

        filas.append({
            "producto_id": producto,
            "nombre_producto": segmento.get("nombre_producto"),
            "categoria": segmento.get("categoria"),
            "proveedor_principal": segmento.get("proveedor_principal"),
            "proveedor_alterno": segmento.get("proveedor_alterno"),
            "categoria_abc": segmento["categoria_abc"],
            "variabilidad": segmento["variabilidad"],
            "metodo_pronostico": segmento["metodo_campeon"],
            "demanda_promedio_mensual_pronosticada": round(demanda_promedio, 2),
            "lead_time_semanas": round(lead_time_meses * 4.345, 1),
            "lead_time_meses": round(lead_time_meses, 2),
            "stock_seguridad": round(stock_seguridad, 2),
            "punto_reorden": round(punto_reorden, 2),
            "fuente_sigma": fuente_sigma,
            # "medido" / "medido en cero" / "no disponible" — mismo criterio que el
            # pronóstico (ver pronostico_futuro.estado_error_historico). Sin backtest
            # no hay error medido: el estado es "no disponible", aunque el stock de
            # seguridad se calcule con el desvío del pronóstico.
            "estado_error_historico": estado_error_historico(
                mae_campeon if fuente_sigma == "MAE de backtest" else None
            ),
            "stock_seguridad_lognormal": round(stock_seguridad_lognormal, 2) if not pd.isna(stock_seguridad_lognormal) else np.nan,
            "punto_reorden_lognormal": round(punto_reorden_lognormal, 2) if not pd.isna(punto_reorden_lognormal) else np.nan,
            "inventario_actual": round(inventario_actual, 2) if hay_inventario else np.nan,
            "stock_al_llegar_pedido": round(stock_al_llegar_pedido, 2) if hay_inventario else np.nan,
            "riesgo_ruptura_durante_entrega": None if riesgo_ruptura_durante_entrega is None else ("SI" if riesgo_ruptura_durante_entrega else "NO"),
            "eoq": round(eoq, 0) if not pd.isna(eoq) else np.nan,
            "cantidad_ideal_pedido": round(cantidad_ideal, 2) if not pd.isna(cantidad_ideal) else np.nan,
            "regla_cantidad": regla_cantidad,
            "costo_unitario": round(costo_unitario_producto, 2) if not pd.isna(costo_unitario_producto) else np.nan,
            "valor_inventario_actual": round(inventario_actual * costo_unitario_producto, 2) if hay_inventario and not pd.isna(costo_unitario_producto) else np.nan,
            "margen_unitario": round(margen_unitario, 2) if not pd.isna(margen_unitario) else np.nan,
            "costo_ruptura_estimado": round(costo_ruptura_estimado, 2) if not pd.isna(costo_ruptura_estimado) else np.nan,
            "fecha_estimada_pedido": fecha_estimada_pedido.strftime("%Y-%m-%d") if fecha_estimada_pedido is not None else None,
            "dias_para_pedido": round(dias_para_pedido) if not pd.isna(dias_para_pedido) else None,
            "ordenar": "SI" if ya_hay_que_pedir else "NO",
            # Datos internos para asignar el presupuesto (no se exponen).
            "_tiene_costo": tiene_costo,
            "_costo": costo_unitario_producto,
            "_cantidad_ideal": cantidad_ideal,
            "_dias": dias_para_pedido,
            "_demanda_promedio": demanda_promedio,
            "_lead_time_meses": lead_time_meses,
            "_hay_alerta": hay_alerta,
        })

    _asignar_presupuesto(filas, working_capital_budget)

    reorder_table = []
    for fila in filas:
        asignada = fila.pop("_asignada")
        limite = fila.pop("_limite")
        ideal = fila.pop("_cantidad_ideal")
        tiene_costo = fila.pop("_tiene_costo")
        costo_exacto = fila.pop("_costo")
        demanda_promedio = fila.pop("_demanda_promedio")
        lead_time_meses = fila.pop("_lead_time_meses")
        fila.pop("_dias")
        fila.pop("_hay_alerta")

        costo_estimado_pedido = (
            asignada * costo_exacto
            if tiene_costo and not pd.isna(asignada) else np.nan
        )
        sin_cubrir = (
            ideal - asignada if tiene_costo and not pd.isna(ideal) and not pd.isna(asignada) else np.nan
        )
        if pd.isna(ideal) or ideal <= 0:
            estado_presupuesto = None
        elif asignada >= ideal:
            estado_presupuesto = "completo"
        elif asignada > 0:
            estado_presupuesto = "parcial"
        else:
            estado_presupuesto = "sin presupuesto"

        # Cobertura después de la entrega: meses de demanda que alcanza el
        # stock que queda al llegar el pedido, más la cantidad asignada.
        if not pd.isna(fila["stock_al_llegar_pedido"]) and not pd.isna(asignada) and demanda_promedio > 0:
            cobertura_despues_entrega = (fila["stock_al_llegar_pedido"] + asignada) / demanda_promedio
        else:
            cobertura_despues_entrega = np.nan

        fila.update({
            "limite_presupuesto_unidades": limite,
            "cantidad_sugerida_pedido": round(asignada, 2) if not pd.isna(asignada) else np.nan,
            "cantidad_sin_cubrir": round(sin_cubrir, 2) if not pd.isna(sin_cubrir) else np.nan,
            "estado_presupuesto": estado_presupuesto,
            "costo_estimado_pedido": round(costo_estimado_pedido, 2) if not pd.isna(costo_estimado_pedido) else np.nan,
            "cobertura_despues_entrega_meses": round(cobertura_despues_entrega, 2) if not pd.isna(cobertura_despues_entrega) else np.nan,
        })
        reorder_table.append(fila)

    reorder_df = pd.DataFrame(reorder_table)
    # Reconstruir el DataFrame desde una lista de diccionarios puede
    # volver a convertir a NaN los None que ya veníamos cuidando desde
    # segmentacion.py (ver utils.nan_a_none) — se corrige de nuevo acá,
    # justo antes de devolver la tabla final. Se incluyen también los
    # campos NUMÉRICOS que pueden faltar (costo_unitario, eoq, etc.):
    # esos ya se guardaban como np.nan (no None) más arriba porque son
    # números, pero el mismo problema de fondo aplica — un DataFrame de
    # float64 tampoco puede contener None, solo NaN — y en los templates
    # se comparan con "is not none" para decidir si mostrar "—".
    reorder_df = nan_a_none(reorder_df, [
        "nombre_producto", "categoria", "proveedor_principal", "proveedor_alterno",
        "fecha_estimada_pedido", "inventario_actual", "eoq", "limite_presupuesto_unidades",
        "cantidad_sugerida_pedido", "costo_unitario", "costo_estimado_pedido",
        "valor_inventario_actual", "dias_para_pedido", "margen_unitario", "costo_ruptura_estimado",
        "stock_seguridad_lognormal", "punto_reorden_lognormal",
        "cantidad_ideal_pedido", "cantidad_sin_cubrir", "estado_presupuesto",
        "stock_al_llegar_pedido", "riesgo_ruptura_durante_entrega", "cobertura_despues_entrega_meses",
        "regla_cantidad", "fuente_sigma",
    ])

    return reorder_df, working_capital_budget, presupuesto_por_producto


def _asignar_presupuesto(filas, presupuesto_total):
    """
    Reparte presupuesto_total entre las cantidades ideales por prioridad
    (ver la política al inicio de calcular_tabla_reorden). Modifica "filas"
    en place agregando "_asignada" y "_limite" (unidades que alcanzaron con
    el dinero que quedaba en el momento de decidir).
    """
    # Prioridad (de mayor a menor): alerta de pedido; riesgo de agotarse
    # antes de que llegue el pedido; clase ABC (A antes que B y C); menos
    # días para llegar al punto de reorden; y, como último criterio fijo,
    # el ID del producto. Sin los criterios de riesgo y ABC, todos los
    # urgentes empatan en días y el desempate alfabético decidía quién
    # recibía el dinero.
    orden_abc = {"A": 0, "B": 1, "C": 2}
    orden = sorted(
        range(len(filas)),
        key=lambda i: (
            0 if filas[i]["ordenar"] == "SI" else 1,
            0 if filas[i]["riesgo_ruptura_durante_entrega"] == "SI" else 1,
            orden_abc.get(filas[i]["categoria_abc"], 3),
            filas[i]["_dias"] if not pd.isna(filas[i]["_dias"]) else float("inf"),
            filas[i]["producto_id"],
        ),
    )
    restante = float(presupuesto_total)
    for i in orden:
        fila = filas[i]
        ideal = fila["_cantidad_ideal"]
        costo = fila["_costo"]

        if pd.isna(ideal):
            fila["_asignada"] = np.nan
            fila["_limite"] = np.nan
            continue

        if not fila["_tiene_costo"]:
            # Sin costo no se puede valorar la compra: no consume presupuesto
            # y se conserva la cantidad ideal.
            fila["_asignada"] = ideal
            fila["_limite"] = np.nan
            continue

        unidades_disponibles = int(restante // costo)
        asignada = min(ideal, unidades_disponibles)
        restante -= asignada * costo
        fila["_asignada"] = asignada
        fila["_limite"] = unidades_disponibles
