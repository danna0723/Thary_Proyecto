import numpy as np
import pandas as pd

from app.services.prediccion.pronostico_futuro import MAE_POR_METODO_COL
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

    reorder_table = []
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
        # seguridad sin ningún valor.
        metodo_campeon = segmento["metodo_campeon"]
        mae_campeon = segmento.get(MAE_POR_METODO_COL.get(metodo_campeon))
        if mae_campeon is not None and not pd.isna(mae_campeon):
            desviacion_demanda = mae_campeon
        else:
            desviacion_demanda = stats_pronostico.at[producto, "desviacion_pronostico_std"]

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

        if not pd.isna(costo_unitario_producto) and costo_unitario_producto > 0:
            limite_presupuesto = int(presupuesto_por_producto / costo_unitario_producto)
        else:
            limite_presupuesto = np.nan

        if not pd.isna(eoq):
            cantidad_sugerida = eoq
        elif not pd.isna(inventario_actual):
            cantidad_sugerida = max(0, punto_reorden - inventario_actual)
        else:
            cantidad_sugerida = np.nan

        # Ejemplo 1: cantidad a pedir acotada por el presupuesto de
        # capital de trabajo (min(EOQ, límite por presupuesto)). Se aplica
        # sobre CUALQUIER método usado arriba (EOQ o la diferencia contra
        # el punto de reorden), no solo cuando hay EOQ: muchos catálogos
        # reales (como el de la clínica) no traen costo_pedido ni
        # porcentaje_costo_mantenimiento, así que el EOQ nunca se puede
        # calcular — pero sí se conoce el costo_unitario, y el presupuesto
        # tiene que respetarse igual.
        if not pd.isna(cantidad_sugerida) and not pd.isna(limite_presupuesto):
            cantidad_sugerida = min(cantidad_sugerida, limite_presupuesto)

        # Costo estimado de hacer ese pedido (cantidad * costo unitario).
        # No viene de ningún ejemplo de referencia — se agrega para que el
        # usuario vea en dinero, no solo en unidades, cuánto implica cada
        # pedido sugerido, y pueda contrastarlo contra su presupuesto.
        if not pd.isna(cantidad_sugerida) and not pd.isna(costo_unitario_producto):
            costo_estimado_pedido = cantidad_sugerida * costo_unitario_producto
        else:
            costo_estimado_pedido = np.nan

        # Valor de lo que ya hay en stock (inventario actual * costo
        # unitario) — para la pantalla de "Estado del inventario", donde
        # interesa cuánto capital está inmovilizado hoy en el depósito,
        # no cuánto costaría reponerlo.
        if not pd.isna(inventario_actual) and not pd.isna(costo_unitario_producto):
            valor_inventario_actual = inventario_actual * costo_unitario_producto
        else:
            valor_inventario_actual = np.nan

        # Costo de ruptura de inventario (stockout): el margen que se
        # dejaría de ganar si el producto se agota y no hay stock
        # disponible durante el tiempo que tarda en llegar el próximo
        # pedido (lead time). Es la tercera categoría de costo de
        # inventario junto con el costo de pedido (EOQ, arriba) y el
        # costo de mantenimiento (porcentaje_costo_mantenimiento, usado
        # en el EOQ) — ver el desglose "Costos del inventario" en la
        # pantalla de Presupuesto. Requiere el precio de venta (columna
        # opcional Precio/Price): sin eso no hay margen que calcular.
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

        # Fecha estimada en la que el inventario va a llegar al punto de
        # reorden, asumiendo que se consume al ritmo promedio pronosticado
        # (una aproximación lineal — no sabe de picos puntuales dentro del
        # mes, pero da una fecha concreta en vez de solo un umbral de
        # cantidad). Si ya está en o por debajo del punto de reorden, la
        # fecha es "ya" (hoy mismo). Si no hay demanda pronosticada, no se
        # puede estimar cuándo se va a agotar.
        ya_hay_que_pedir = not pd.isna(inventario_actual) and inventario_actual <= punto_reorden
        if pd.isna(inventario_actual):
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan
        elif ya_hay_que_pedir:
            fecha_estimada_pedido = fecha_referencia
            dias_para_pedido = 0
        elif demanda_promedio > 0:
            demanda_diaria = demanda_promedio / DIAS_POR_MES
            dias_para_pedido = min(
                (inventario_actual - punto_reorden) / demanda_diaria,
                DIAS_PARA_PEDIDO_TECHO,
            )
            fecha_estimada_pedido = fecha_referencia + pd.Timedelta(days=dias_para_pedido)
        else:
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan

        reorder_table.append({
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
            "stock_seguridad_lognormal": round(stock_seguridad_lognormal, 2) if not pd.isna(stock_seguridad_lognormal) else np.nan,
            "punto_reorden_lognormal": round(punto_reorden_lognormal, 2) if not pd.isna(punto_reorden_lognormal) else np.nan,
            "inventario_actual": round(inventario_actual, 2) if not pd.isna(inventario_actual) else np.nan,
            "eoq": round(eoq, 0) if not pd.isna(eoq) else np.nan,
            "limite_presupuesto_unidades": limite_presupuesto,
            "cantidad_sugerida_pedido": round(cantidad_sugerida, 2) if not pd.isna(cantidad_sugerida) else np.nan,
            "costo_unitario": round(costo_unitario_producto, 2) if not pd.isna(costo_unitario_producto) else np.nan,
            "costo_estimado_pedido": round(costo_estimado_pedido, 2) if not pd.isna(costo_estimado_pedido) else np.nan,
            "valor_inventario_actual": round(valor_inventario_actual, 2) if not pd.isna(valor_inventario_actual) else np.nan,
            "margen_unitario": round(margen_unitario, 2) if not pd.isna(margen_unitario) else np.nan,
            "costo_ruptura_estimado": round(costo_ruptura_estimado, 2) if not pd.isna(costo_ruptura_estimado) else np.nan,
            "fecha_estimada_pedido": fecha_estimada_pedido.strftime("%Y-%m-%d") if fecha_estimada_pedido is not None else None,
            "dias_para_pedido": round(dias_para_pedido) if not pd.isna(dias_para_pedido) else None,
            "ordenar": "SI" if ya_hay_que_pedir else "NO"
        })

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
    ])

    return reorder_df, working_capital_budget, presupuesto_por_producto
