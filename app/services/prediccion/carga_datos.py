import numpy as np
import pandas as pd


# EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de referencia.
# Los 4 ejemplos ya reciben datos limpios en un formato fijo conocido
# (un CSV/notebook específico); acá esto es infraestructura necesaria
# para que el sistema acepte el formato genérico largo de CUALQUIER
# inventario que suba el usuario (nombres de columna variables,
# encabezados en inglés o español, meses incompletos, etc.).
ETIQUETAS_COLUMNAS_OBLIGATORIAS = {
    "fecha": "Fecha (o Date, Sale_Date)",
    "producto_id": "Producto (o Product, Product_ID)",
    "demanda": "Demanda (o Demand, Sales, Quantity_Sold)",
}


def cargar_y_limpiar(ruta_archivo):
    """
    Lee el CSV subido por el usuario, normaliza nombres de columnas a
    variantes conocidas (inglés/español), valida las columnas
    obligatorias y descarta el último mes si viene incompleto.

    Cada problema posible levanta un ValueError con un mensaje pensado
    para mostrarse tal cual al usuario (ver procesar() en
    app/routes/main.py, que lo manda directo a flash()) — no un mensaje
    técnico de pandas, sino qué le pasa al archivo y qué tiene que
    corregir.
    """
    try:
        df = pd.read_csv(ruta_archivo)
    except pd.errors.EmptyDataError:
        raise ValueError("El archivo está vacío. Sube un CSV con al menos una fila de datos.")
    except pd.errors.ParserError:
        raise ValueError(
            "No se pudo leer el archivo como CSV. Verifica que sea un archivo de texto "
            "separado por comas (por ejemplo, no un Excel renombrado a .csv)."
        )
    except UnicodeDecodeError:
        raise ValueError(
            "El archivo tiene una codificación de caracteres que el sistema no reconoce. "
            "Guárdalo como CSV con codificación UTF-8 e intenta de nuevo."
        )

    if df.empty:
        raise ValueError("El archivo no tiene filas de datos, solo encabezados.")

    df = df.rename(columns={
        "Sale_Date": "fecha", "Date": "fecha", "date": "fecha", "Fecha": "fecha",
        "Product_ID": "producto_id", "Product": "producto_id", "Producto": "producto_id",
        "Quantity_Sold": "demanda", "Sales": "demanda", "Demand": "demanda", "Demanda": "demanda",
        "Unit_Price": "precio", "Price": "precio", "Price_per_unit": "precio",
        "Inventario_Actual": "inventario", "Inventory": "inventario", "Stock": "inventario",
        "Store": "tienda_id", "Store_ID": "tienda_id",
        "Category": "categoria", "Product_Category": "categoria",
        "Product_Name": "nombre_producto", "Nombre_Producto": "nombre_producto", "Nombre": "nombre_producto",
        "Lead_time_Semana": "lead_time_semanas",
        "Costo_pedido": "costo_pedido",
        "Costo_unitario": "costo_unitario",
        "Porcentaje_costo_mantenimiento": "porcentaje_costo_mantenimiento",
        "Supplier": "proveedor_principal", "Proveedor": "proveedor_principal",
        "Proveedor_Principal": "proveedor_principal",
        "Alternate_Supplier": "proveedor_alterno", "Proveedor_Alterno": "proveedor_alterno",
        "Proveedor_Alternativo": "proveedor_alterno",
    })

    columnas_faltantes = [c for c in ETIQUETAS_COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if columnas_faltantes:
        faltan = ", ".join(ETIQUETAS_COLUMNAS_OBLIGATORIAS[c] for c in columnas_faltantes)
        raise ValueError(
            f"Al archivo le falta al menos una columna obligatoria: {faltan}. "
            "Agrégala al CSV (con ese nombre exacto u otro de los aceptados) y vuelve a subirlo."
        )

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["demanda"] = pd.to_numeric(df["demanda"], errors="coerce")
    filas_antes_de_limpiar = len(df)
    df = df.dropna(subset=["fecha", "producto_id", "demanda"])
    if df.empty:
        raise ValueError(
            "Ninguna de las filas del archivo tiene a la vez una fecha, un producto y una "
            "demanda válidos. Revisa que la columna de fecha tenga fechas reales (no texto) "
            "y que la columna de demanda tenga números."
        )
    if len(df) < filas_antes_de_limpiar:
        print(
            f"Se descartaron {filas_antes_de_limpiar - len(df)} fila(s) sin fecha, "
            "producto o demanda válidos."
        )
    df["demanda"] = np.maximum(df["demanda"], 0)
    df = df.sort_values(["producto_id", "fecha"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Descartar el último mes si está incompleto (ej. el CSV corta
    # el día 1 de un mes nuevo, dejando solo 1 día de ese mes en vez
    # de todo el mes). Si no se descarta, distorsiona la evaluación
    # Y el pronóstico futuro (que usa el último mes como lag_1).
    # ------------------------------------------------------------
    ultimo_periodo_crudo = df["fecha"].max().to_period("M")
    fin_ultimo_periodo_crudo = ultimo_periodo_crudo.end_time
    dias_faltantes = (fin_ultimo_periodo_crudo - df["fecha"].max()).days

    if dias_faltantes > 5:
        filas_antes = len(df)
        df = df[df["fecha"].dt.to_period("M") != ultimo_periodo_crudo].copy()
        print(
            f"Se descartó el mes {ultimo_periodo_crudo} por estar incompleto "
            f"(le faltaban {dias_faltantes} días). Filas removidas: {filas_antes - len(df)}."
        )
        if df.empty:
            raise ValueError(
                "El archivo solo tiene datos de un mes, y ese mes está incompleto "
                "(no llega hasta fin de mes). Sube un archivo con más de un mes de historial."
            )

    return df


def agregar_mensual(df):
    """
    Agrega las transacciones (posiblemente diarias) a nivel producto-mes,
    completa el calendario para que cada producto tenga una fila por cada
    mes del rango (demanda 0 donde no hubo ventas), y rellena hacia
    adelante/atrás los atributos que no cambian mes a mes (precio,
    categoría, lead time, etc.).
    """
    df = df.copy()
    df["mes_periodo"] = df["fecha"].dt.to_period("M").dt.to_timestamp()

    agregaciones = {"demanda": "sum"}
    if "precio" in df.columns:
        agregaciones["precio"] = "mean"
    if "inventario" in df.columns:
        agregaciones["inventario"] = "mean"
    if "costo_unitario" in df.columns:
        agregaciones["costo_unitario"] = "mean"
    if "categoria" in df.columns:
        agregaciones["categoria"] = "first"
    for col in ["lead_time_semanas", "costo_pedido", "porcentaje_costo_mantenimiento"]:
        if col in df.columns:
            agregaciones[col] = "first"

    df_mensual = (
        df.groupby(["producto_id", "mes_periodo"])
        .agg(agregaciones)
        .reset_index()
        .rename(columns={"mes_periodo": "fecha"})
    )

    productos = df_mensual["producto_id"].unique()
    meses_completos = pd.date_range(df_mensual["fecha"].min(), df_mensual["fecha"].max(), freq="MS")

    calendario_producto_mes = (
        pd.MultiIndex.from_product([productos, meses_completos], names=["producto_id", "fecha"])
        .to_frame(index=False)
    )

    df_mensual = calendario_producto_mes.merge(df_mensual, on=["producto_id", "fecha"], how="left")
    df_mensual["demanda"] = df_mensual["demanda"].fillna(0)

    columnas_ffill = [
        c for c in
        ["precio", "costo_unitario", "inventario", "categoria",
         "lead_time_semanas", "costo_pedido", "porcentaje_costo_mantenimiento"]
        if c in df_mensual.columns
    ]
    for col in columnas_ffill:
        df_mensual[col] = df_mensual.groupby("producto_id")[col].transform(lambda x: x.ffill().bfill())

    df_mensual = df_mensual.sort_values(["producto_id", "fecha"]).reset_index(drop=True)
    return df_mensual
