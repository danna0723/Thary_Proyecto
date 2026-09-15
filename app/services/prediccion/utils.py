def nan_a_none(df, columnas):
    """
    Reemplaza NaN por None de Python en las columnas de texto indicadas,
    de forma que sobrevivan a to_dict(orient="records") (y de ahí a los
    templates, donde se usa "{% if fila.categoria %}" para distinguir
    "sin dato" de un valor real).

    Necesario por un detalle de pandas >= 3.0: su nuevo dtype "str" por
    defecto para columnas de texto NO puede almacenar None — cualquier
    None que se le asigne se normaliza de nuevo a NaN. Pasar la columna
    por dtype "object" antes de asignar None evita esa conversión. Hay
    que aplicarlo justo antes de to_dict(): un merge o una reconstrucción
    posterior del DataFrame (p. ej. pd.DataFrame(lista_de_dicts)) puede
    volver a inferir dtype "str" y deshacer el arreglo.
    """
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df
