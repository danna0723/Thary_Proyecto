import os
import json


# Infraestructura propia (guardado de artefactos en disco para el
# dashboard) — no corresponde a la metodología de ningún ejemplo de
# referencia en particular.
def guardar_resultados(
    carpeta_resultados, model, feature_cols, resultados_prediccion,
    resultados, reorder_df, producto_comportamiento, pronostico_futuro_df,
    validacion_cruzada_detalle, validacion_cruzada_resumen
):
    """Guarda en disco los artefactos del pipeline y devuelve el dict de rutas."""
    ruta_modelo = os.path.join(carpeta_resultados, "modelo_xgboost_demanda.json")
    model.save_model(ruta_modelo)

    ruta_features = os.path.join(carpeta_resultados, "feature_columns.json")
    with open(ruta_features, "w") as f:
        json.dump(feature_cols, f)

    ruta_predicciones = os.path.join(carpeta_resultados, "predicciones.csv")
    resultados_prediccion.to_csv(ruta_predicciones, index=False)

    ruta_evaluacion = os.path.join(carpeta_resultados, "evaluacion_modelos.csv")
    resultados.to_csv(ruta_evaluacion, index=False)

    ruta_reorder = os.path.join(carpeta_resultados, "reorder_table.csv")
    reorder_df.to_csv(ruta_reorder, index=False)

    ruta_segmentacion = os.path.join(carpeta_resultados, "segmentacion_productos.csv")
    producto_comportamiento.to_csv(ruta_segmentacion, index=False)

    ruta_pronostico_futuro = os.path.join(carpeta_resultados, "pronostico_futuro.csv")
    pronostico_futuro_df.to_csv(ruta_pronostico_futuro, index=False)

    ruta_validacion_cruzada = os.path.join(carpeta_resultados, "validacion_cruzada.json")
    with open(ruta_validacion_cruzada, "w", encoding="utf-8") as f:
        json.dump(
            {"detalle_por_fold": validacion_cruzada_detalle, "resumen": validacion_cruzada_resumen},
            f, ensure_ascii=False, indent=2
        )

    return {
        "modelo": ruta_modelo,
        "features": ruta_features,
        "predicciones": ruta_predicciones,
        "evaluacion": ruta_evaluacion,
        "reorder": ruta_reorder,
        "segmentacion": ruta_segmentacion,
        "pronostico_futuro": ruta_pronostico_futuro,
        "validacion_cruzada": ruta_validacion_cruzada,
    }
