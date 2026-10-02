import os

import numpy as np
import pandas as pd

from app.services.prediccion.carga_datos import cargar_y_limpiar, agregar_mensual
from app.services.prediccion.segmentacion import calcular_comportamiento_producto
from app.services.prediccion.features import construir_features
from app.services.prediccion.regresion_lineal import entrenar_regresion_lineal
from app.services.prediccion.modelos import (
    elegir_campeon_por_producto,
    entrenar_xgboost,
    prediccion_baseline_movil,
    benchmarking_modelos,
    construir_tabla_predicciones,
    tamano_test_backtest,
)
from app.services.prediccion.pronostico_futuro import (
    generar_pronostico_futuro,
    HORIZONTE_MESES_DEFAULT,
    HORIZONTE_MESES_MIN,
    HORIZONTE_MESES_MAX,
)
from app.services.prediccion.reposicion import calcular_tabla_reorden
from app.services.prediccion.validacion import validacion_cruzada_temporal
from app.services.prediccion.graficas import (
    graficar_segmentacion,
    graficar_importancia_features,
    graficar_comparacion_metricas,
    graficar_dispersion_modelos,
)
from app.services.prediccion.exportacion import guardar_resultados


def ejecutar_sistema(
    ruta_archivo, carpeta_resultados=None, presupuesto_capital_trabajo=None, horizonte_meses=None
):
    """
    Orquestador del pipeline completo. Cada paso vive en su propio
    módulo bajo app/services/prediccion/ — ver ahí los comentarios de
    metodología (qué parte corresponde a cada uno de los 4 ejemplos de
    referencia del proyecto).

    presupuesto_capital_trabajo: monto que el usuario ingresó en el
    formulario de subida para acotar cuánto se sugiere pedir por
    producto (ver reposicion.py). Si es None, se usa un valor por
    defecto.

    horizonte_meses: a cuántos meses hacia adelante pronosticar, elegido
    por el usuario en el formulario de subida. Si es None o queda fuera
    del rango permitido, se acota a [HORIZONTE_MESES_MIN,
    HORIZONTE_MESES_MAX] en vez de fallar — el mismo criterio que ya se
    usa para el presupuesto (un valor inválido no debería bloquear la
    subida, cae a un valor por defecto razonable).
    """
    if carpeta_resultados is None:
        carpeta_resultados = "resultados"
    os.makedirs(carpeta_resultados, exist_ok=True)

    if horizonte_meses is None:
        horizonte_meses = HORIZONTE_MESES_DEFAULT
    horizonte_meses = max(HORIZONTE_MESES_MIN, min(HORIZONTE_MESES_MAX, int(horizonte_meses)))

    # 1. Carga, limpieza y agregación mensual (carga_datos.py)
    df = cargar_y_limpiar(ruta_archivo)
    df_mensual = agregar_mensual(df)

    # 2. Segmentación de productos: ABC, variabilidad, etiquetas
    #    (segmentacion.py)
    producto_comportamiento = calcular_comportamiento_producto(df, df_mensual)
    ruta_segmentacion_grafica = graficar_segmentacion(producto_comportamiento, carpeta_resultados)

    # 3. Feature engineering: fecha cíclica, lags, medias móviles,
    #    estadísticas por producto (features.py)
    df_mensual, df_model, feature_cols = construir_features(df_mensual, producto_comportamiento)

    meses_ordenados = sorted(df_model["fecha"].unique())
    if len(meses_ordenados) < 4:
        raise ValueError(
            "El archivo necesita al menos 4 meses de historial por producto (se pierden los "
            "primeros 3 meses al calcular tendencias). Sube un archivo con más meses de datos."
        )

    # Ejemplo 1 / Ejemplo 3: split de entrenamiento/prueba por fecha (no
    # aleatorio), igual que el corte por semana en ambos notebooks de
    # referencia — necesario en series de tiempo para no entrenar con
    # datos "del futuro" respecto al período de prueba.
    #
    # EXTENSIÓN PROPIA: el tamaño del test ya no es siempre 3 — con
    # archivos largos (ver tamano_test_backtest en modelos.py) se usa un
    # test más grande, para que la prueba de Diebold-Mariano de
    # elegir_campeon_por_producto tenga más poder estadístico.
    tamano_test = tamano_test_backtest(len(meses_ordenados))
    meses_test = meses_ordenados[-tamano_test:]
    meses_antes_del_test = meses_ordenados[:-tamano_test]

    # EXTENSIÓN PROPIA (corrección de fuga de datos en la selección de
    # modelo) — el método campeón de cada producto ya NO se elige
    # mirando el mismo período que después se usa para reportar
    # mape_final/wape_final. Se separa un período de VALIDACIÓN,
    # anterior y distinto al de test, usado solo para decidir qué
    # método conviene en cada producto (elegir_campeon_por_producto,
    # más abajo) — el test queda intacto, sin participar en ninguna
    # decisión, para que esas métricas finales sean una medida honesta
    # de qué tan bien generaliza el sistema a datos que nunca vio, ni
    # para entrenar ni para elegir método (ver el comentario largo
    # junto a elegir_campeon_por_producto en modelos.py).
    #
    # Mismo tamaño que el test, por la misma razón que ese tamaño ya
    # varía con tamano_test_backtest: la prueba de Diebold-Mariano
    # necesita suficientes meses para tener poder estadístico, sin
    # importar si se aplica sobre test o sobre validación.
    tamano_val = tamano_test
    if len(meses_antes_del_test) > tamano_val:
        meses_val = meses_antes_del_test[-tamano_val:]
        meses_train = meses_antes_del_test[:-tamano_val]
        seleccion_metodo_con_fuga = False
    else:
        # Historial muy corto: no alcanza para separar un período de
        # validación sin dejar el entrenamiento vacío. El sistema sigue
        # funcionando, cayendo al criterio anterior (elegir método con
        # el propio test) — pero en ese caso mape_final/wape_final
        # vuelven a ser una cota optimista, no una medida de
        # generalización genuina. Se deja explícito en el resultado en
        # vez de ocultarlo.
        meses_val = meses_test
        meses_train = meses_antes_del_test
        seleccion_metodo_con_fuga = True

    train_df = df_model[df_model["fecha"].isin(meses_train)].copy()
    val_df = df_model[df_model["fecha"].isin(meses_val)].copy()
    test_df = df_model[df_model["fecha"].isin(meses_test)].copy()

    if len(train_df) == 0:
        raise ValueError("No quedó historial suficiente para entrenar el modelo. Sube un archivo con más meses de datos.")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["demanda"]
    y_test = test_df["demanda"]

    # Ejemplo 3 (Kaggle Grupo Bimbo): log1p del target antes de entrenar
    # XGBoost; se revierte con expm1 al predecir (dentro de entrenar_xgboost).
    y_train_log = np.log1p(y_train)

    # 4. Modelos: Regresión Lineal (regresion_lineal.py — Ejemplo 4),
    #    XGBoost (modelos.py — Ejemplos 1 y 3), Media Móvil (modelos.py — Ejemplo 1)
    #    Entrenados UNA sola vez con meses_train — se reutilizan (sin
    #    volver a entrenar) tanto para predecir sobre validación (para
    #    elegir método) como sobre test (para reportar), igual que
    #    pronostico_futuro.py ya reutiliza estos mismos modelos para el
    #    pronóstico recursivo.
    modelo_lr, pred_lr, features_lr = entrenar_regresion_lineal(X_train, y_train, X_test, feature_cols)
    model, pred_xgb = entrenar_xgboost(X_train, y_train_log, X_test)
    pred_baseline_3 = prediccion_baseline_movil(test_df)

    pred_lr_val = np.maximum(modelo_lr.predict(X_val[features_lr]), 0)
    pred_xgb_val = np.maximum(np.expm1(model.predict(X_val)), 0)
    pred_baseline_val = prediccion_baseline_movil(val_df)

    tabla_val = val_df[["producto_id", "demanda"]].copy()
    tabla_val["prediccion_media_movil"] = pred_baseline_val
    tabla_val["prediccion_regresion_lineal"] = pred_lr_val
    tabla_val["prediccion_xgboost"] = pred_xgb_val
    # El segundo valor (MAE por método sobre la validación) no se usa
    # acá — el rango de pronóstico futuro usa el MAE sobre TEST, que
    # construir_tabla_predicciones calcula más abajo sin fuga.
    metodo_campeon, _ = elegir_campeon_por_producto(tabla_val)

    resultados = benchmarking_modelos(y_test, pred_baseline_3, pred_lr, pred_xgb)
    resultados_prediccion, mape_final, wape_final, metodo_campeon, errores_por_producto = construir_tabla_predicciones(
        test_df, pred_baseline_3, pred_lr, pred_xgb, metodo_campeon
    )

    # El campeón por producto (decidido arriba con el período de
    # validación) se agrega a producto_comportamiento para que
    # pronostico_futuro.py y reposicion.py lo lean igual que cualquier
    # otra columna de segmentación. Los productos que por algún motivo
    # no llegaron a val_df (catálogos muy chicos) quedan con Media
    # Móvil como respaldo seguro.
    producto_comportamiento["metodo_campeon"] = (
        producto_comportamiento["producto_id"].map(metodo_campeon).fillna("Media móvil")
    )

    # EXTENSIÓN PROPIA — MAE de CADA método en el backtest, por producto
    # (no solo el del campeón), para que pronóstico futuro pueda armar un
    # rango [mínimo, máximo] alrededor de cada predicción puntual, según
    # qué tanto se equivocó en el pasado el método que termine usándose
    # para ese producto. Ver pronostico_futuro.py.
    errores_renombrados = errores_por_producto.rename(columns={
        "Media móvil": "mae_media_movil",
        "Regresión Lineal": "mae_regresion_lineal",
        "XGBoost": "mae_xgboost",
    })
    producto_comportamiento = producto_comportamiento.merge(
        errores_renombrados, left_on="producto_id", right_index=True, how="left"
    )

    ruta_importancia = graficar_importancia_features(model, feature_cols, carpeta_resultados)

    # 5. Validación cruzada temporal (validacion.py): repite el
    #    entrenamiento de los tres modelos contra varios splits por
    #    fecha en vez de uno solo, aplicando la MISMA política de
    #    selección por producto decidida arriba (metodo_campeon) —
    #    para confirmar que esa decisión, tomada con un período de
    #    validación, sigue funcionando en otros períodos históricos. No
    #    reemplaza el split de producción de arriba — es una validación
    #    adicional, informativa, que se muestra en el panel técnico.
    detalle_validacion_cruzada, resumen_validacion_cruzada = validacion_cruzada_temporal(
        df_model, feature_cols, metodo_campeon
    )

    # 6. Pronóstico recursivo de los próximos N meses, elegidos por el
    #    usuario (pronostico_futuro.py)
    (
        pronostico_futuro_df,
        pronostico_pivot,
        meses_pronosticados,
        meses_pronosticados_legibles,
    ) = generar_pronostico_futuro(
        df_mensual, producto_comportamiento, model, feature_cols, modelo_lr, features_lr,
        horizonte_meses=horizonte_meses,
    )

    # 7. Puntos de reorden y EOQ (reposicion.py — Ejemplos 1 y 2)
    reorder_df, presupuesto_usado, presupuesto_por_producto = calcular_tabla_reorden(
        df_model, df_mensual, producto_comportamiento, pronostico_futuro_df,
        presupuesto_capital_trabajo=presupuesto_capital_trabajo
    )

    # Costo total estimado de los pedidos urgentes (los que ya llegaron
    # a su punto de reorden), para contrastarlo contra el presupuesto en
    # el dashboard. Productos sin costo_unitario simplemente no suman.
    costos_urgentes = reorder_df.loc[reorder_df["ordenar"] == "SI", "costo_estimado_pedido"]
    costo_total_pedidos_urgentes = costos_urgentes.sum(skipna=True) if not costos_urgentes.empty else 0.0

    # Valor total del inventario que hay HOY en stock (para la pantalla
    # de "Estado del inventario"), y cuántos productos no tienen dato de
    # inventario en el CSV (para que quede claro que esos no se están
    # contando, en vez de asumir silenciosamente que están en cero).
    valor_total_inventario = reorder_df["valor_inventario_actual"].sum(skipna=True)
    productos_sin_dato_inventario = int(reorder_df["inventario_actual"].isna().sum())
    productos_en_alerta = int((reorder_df["ordenar"] == "SI").sum())

    # 8. Gráficas y exportación de artefactos (graficas.py / exportacion.py)
    graficas_metricas = graficar_comparacion_metricas(resultados, carpeta_resultados)
    ruta_dispersion_modelos = graficar_dispersion_modelos(resultados_prediccion, carpeta_resultados)
    rutas = guardar_resultados(
        carpeta_resultados, model, feature_cols, resultados_prediccion,
        resultados, reorder_df, producto_comportamiento, pronostico_futuro_df,
        detalle_validacion_cruzada, resumen_validacion_cruzada
    )
    rutas["importancia"] = ruta_importancia
    rutas["segmentacion_grafica"] = ruta_segmentacion_grafica
    rutas["graficas"] = graficas_metricas
    rutas["dispersion_modelos"] = ruta_dispersion_modelos

    return {
        "metricas": resultados.to_dict(orient="records"),
        "predicciones": resultados_prediccion.to_dict(orient="records"),
        "reorder": reorder_df.to_dict(orient="records"),
        "segmentacion": producto_comportamiento.to_dict(orient="records"),
        "mape_final": None if pd.isna(mape_final) else round(float(mape_final), 2),
        "wape_final": None if pd.isna(wape_final) else round(float(wape_final), 2),
        "validacion_cruzada_detalle": detalle_validacion_cruzada,
        "validacion_cruzada_resumen": resumen_validacion_cruzada,
        "pronostico_futuro": pronostico_futuro_df.to_dict(orient="records"),
        "pronostico_pivot": pronostico_pivot,
        "meses_pronosticados": meses_pronosticados,
        "meses_pronosticados_legibles": meses_pronosticados_legibles,
        "horizonte_meses": horizonte_meses,
        "features_regresion_lineal": features_lr,
        "presupuesto_capital_trabajo": round(float(presupuesto_usado), 2),
        "presupuesto_por_producto": round(float(presupuesto_por_producto), 2),
        "costo_total_pedidos_urgentes": round(float(costo_total_pedidos_urgentes), 2),
        "valor_total_inventario": None if pd.isna(valor_total_inventario) else round(float(valor_total_inventario), 2),
        "productos_sin_dato_inventario": productos_sin_dato_inventario,
        "productos_en_alerta": productos_en_alerta,
        "seleccion_metodo_con_fuga": seleccion_metodo_con_fuga,
        "archivos": rutas,
    }
