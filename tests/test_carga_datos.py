"""
Pruebas de los mensajes de error de cargar_y_limpiar() (carga_datos.py)
— confirman que un CSV inválido no revienta con un error técnico de
pandas, sino con un mensaje que le dice al usuario qué está mal en el
archivo y qué tiene que cambiar. Cada prueba simula un archivo mal
formado distinto y verifica el mensaje que vería el usuario en la
pantalla de carga (ver procesar() en app/routes/main.py, que manda este
mismo texto a flash()).
"""

import pytest

from app.services.prediccion.carga_datos import cargar_y_limpiar


def escribir_csv(tmp_path, contenido, nombre="archivo.csv"):
    ruta = tmp_path / nombre
    ruta.write_text(contenido, encoding="utf-8")
    return str(ruta)


def test_archivo_vacio_da_mensaje_claro(tmp_path):
    ruta = escribir_csv(tmp_path, "")
    with pytest.raises(ValueError, match="vacío"):
        cargar_y_limpiar(ruta)


def test_archivo_solo_con_encabezados_da_mensaje_claro(tmp_path):
    ruta = escribir_csv(tmp_path, "Fecha,Producto,Demanda\n")
    with pytest.raises(ValueError, match="solo encabezados"):
        cargar_y_limpiar(ruta)


def test_falta_columna_obligatoria_la_nombra_en_el_mensaje(tmp_path):
    # Falta "Demanda" — el mensaje tiene que mencionarla, no solo decir
    # "falta una columna" en general.
    ruta = escribir_csv(tmp_path, "Fecha,Producto\n2026-01-01,P1\n")
    with pytest.raises(ValueError, match="Demanda"):
        cargar_y_limpiar(ruta)


def test_acepta_nombres_de_columna_en_ingles(tmp_path):
    """No debería fallar: Date/Product/Demand son sinónimos aceptados
    de Fecha/Producto/Demanda (ver ETIQUETAS_COLUMNAS_OBLIGATORIAS)."""
    ruta = escribir_csv(
        tmp_path,
        "Date,Product,Demand\n"
        "2026-01-15,P1,10\n2026-02-15,P1,12\n2026-03-15,P1,11\n2026-04-15,P1,13\n",
    )
    df = cargar_y_limpiar(ruta)
    assert not df.empty


def test_reconoce_la_columna_precio_en_espanol(tmp_path):
    """Bug encontrado en producción: la pantalla de carga documenta
    "Precio" como nombre de columna aceptado (ver index.html), pero el
    diccionario de renombres solo tenía las variantes en inglés
    (Unit_Price/Price/Price_per_unit) — un archivo con la columna
    "Precio" tal cual se quedaba sin ese dato silenciosamente, sin
    ningún error que avisara. Esto rompía el cálculo de costo de
    ruptura de inventario en reposicion.py, que depende del precio
    para estimar el margen por unidad."""
    ruta = escribir_csv(
        tmp_path,
        "Fecha,Producto,Demanda,Precio\n"
        "2026-01-15,P1,10,5000\n2026-02-15,P1,12,5000\n"
        "2026-03-15,P1,11,5000\n2026-04-15,P1,13,5000\n",
    )
    df = cargar_y_limpiar(ruta)
    assert "precio" in df.columns
    assert (df["precio"] == 5000).all()


def test_fechas_y_demanda_invalidas_en_todas_las_filas(tmp_path):
    """Todas las filas tienen basura en fecha y demanda -> después de
    limpiar no queda ninguna fila válida. Tiene que fallar con un
    mensaje que oriente a revisar esas dos columnas, no con un error
    interno de pandas al intentar calcular con un DataFrame vacío."""
    ruta = escribir_csv(
        tmp_path,
        "Fecha,Producto,Demanda\nno-es-fecha,P1,no-es-numero\notra-basura,P2,tampoco\n",
    )
    with pytest.raises(ValueError, match="fecha.*demanda|demanda.*fecha"):
        cargar_y_limpiar(ruta)


def test_algunas_filas_invalidas_se_descartan_pero_el_resto_se_procesa(tmp_path):
    """Si solo ALGUNAS filas son inválidas, esas se descartan y el
    archivo se procesa igual con las que sí sirven — no debería fallar
    solo porque una fila puntual tenga un dato mal cargado."""
    ruta = escribir_csv(
        tmp_path,
        "Fecha,Producto,Demanda\n"
        "2026-01-15,P1,10\n"
        "fecha-invalida,P1,999\n"  # esta fila se descarta por fecha inválida
        "2026-02-15,P1,12\n2026-03-15,P1,11\n2026-04-15,P1,13\n",  # abril: mes incompleto, también se descarta
    )
    df = cargar_y_limpiar(ruta)
    assert len(df) == 3  # quedan enero, febrero y marzo
