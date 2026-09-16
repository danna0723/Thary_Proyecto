import os
import csv
import io

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, send_from_directory, session, Response
)
from werkzeug.utils import secure_filename

from app.services.sistema_prediccion import ejecutar_sistema
from app.services.auth import admin_required, login_required, buscar_usuario
from app.services.pedidos import cargar_pedidos, registrar_pedido, quitar_pedido
from app.services.perfil_empresa import cargar_nombre_empresa
from app.services.resultados import guardar_resultado, cargar_ultima_corrida
from app.services.actividad import registrar as registrar_actividad, listar_propia, listar_empresa
from app.services.prediccion.reposicion import DIAS_POR_MES
from app.services.prediccion.pronostico_futuro import HORIZONTE_MESES_DEFAULT

main_bp = Blueprint("main", __name__)

EXTENSIONES_PERMITIDAS = {"csv"}


def extension_permitida(nombre_archivo):
    return (
        "." in nombre_archivo
        and nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS
    )


def carpeta_uploads(empresa_id):
    """Carpeta de CSV subidos de una empresa puntual. Cada empresa
    (cada admin que se registró) tiene la suya, separada del resto."""
    ruta = os.path.join(current_app.config["UPLOAD_FOLDER_BASE"], empresa_id)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def carpeta_resultados(empresa_id):
    """Carpeta donde ejecutar_sistema escribe el pronóstico, los
    gráficos y demás archivos de resultado de una empresa puntual."""
    ruta = os.path.join(current_app.config["RESULTADOS_FOLDER_BASE"], empresa_id)
    os.makedirs(ruta, exist_ok=True)
    return ruta


ARCHIVOS_RESULTADO = {
    "modelo": "modelo_xgboost_demanda.json",
    "features": "feature_columns.json",
    "predicciones": "predicciones.csv",
    "evaluacion": "evaluacion_modelos.csv",
    "reorder": "reorder_table.csv",
    "segmentacion": "segmentacion_productos.csv",
    "pronostico_futuro": "pronostico_futuro.csv",
    "validacion_cruzada": "validacion_cruzada.json",
    "importancia": "xgb_feature_importance.png",
    "segmentacion_grafica": "segmentacion_abc_variabilidad.png",
}


def guardar_ultimo_resultado(resultado, empresa_id):
    """Guarda el resultado del procesamiento en MySQL (tabla "corridas"
    y sus tablas hijas — ver app/services/resultados.py) para que
    /desarrollo y /dashboard puedan leerlo después, sin volver a correr
    el pipeline."""
    guardar_resultado(resultado, empresa_id)


def cargar_ultimo_resultado(empresa_id):
    """Devuelve el resultado de la corrida más reciente de esta
    empresa, o None si todavía no procesó ningún CSV. La parte tabular
    (predicciones, reorder, métricas, etc.) viene de MySQL; acá se le
    agrega "archivos", que son rutas a disco (gráficas, CSVs
    exportados) y por eso las arma esta función, no resultados.py."""
    resultado = cargar_ultima_corrida(empresa_id)
    if resultado is None:
        return None
    carpeta = carpeta_resultados(empresa_id)
    resultado["archivos"] = {
        clave: os.path.join(carpeta, nombre) for clave, nombre in ARCHIVOS_RESULTADO.items()
    }
    return resultado


def pantalla_sin_datos():
    """Qué mostrar cuando todavía no hay ningún CSV procesado. El admin
    puede ir a subir uno (/), pero un empleado no tiene acceso a esa
    pantalla — mandarlo ahí causaría un redirect infinito (admin_required
    lo devolvería al dashboard). En vez de eso, a un empleado se le
    muestra una pantalla de espera."""
    if session.get("rol") == "admin":
        flash("Todavía no has subido ningún archivo. Sube un CSV primero.")
        return redirect(url_for("main.index"))
    return render_template("sin_datos.html")


@main_bp.route("/inicio", methods=["GET"])
@login_required
def inicio():
    """Primera pantalla que ve cualquier usuario al entrar — un menú de
    tarjetas con todas las opciones del sistema, filtradas según el rol
    (un empleado no ve las tarjetas de administración)."""
    return render_template("inicio.html")


@main_bp.route("/", methods=["GET"])
@admin_required
def index():
    # nombre_empresa ya llega al template vía el context processor
    # global (app/__init__.py), que lo usa también en el sidebar.
    return render_template("index.html")


@main_bp.route("/procesar", methods=["POST"])
@admin_required
def procesar():
    archivo = request.files.get("archivo")

    if archivo is None or archivo.filename == "":
        flash("Debes seleccionar un archivo CSV antes de continuar.")
        return redirect(url_for("main.index"))

    if not extension_permitida(archivo.filename):
        flash("El archivo debe tener extensión .csv")
        return redirect(url_for("main.index"))

    nombre_seguro = secure_filename(archivo.filename)
    ruta_csv = os.path.join(carpeta_uploads(session["empresa_id"]), nombre_seguro)
    archivo.save(ruta_csv)

    # Presupuesto opcional: si el usuario lo deja vacío o pone algo
    # inválido, ejecutar_sistema cae a su valor por defecto en vez de
    # bloquear la subida por esto.
    presupuesto_texto = request.form.get("presupuesto", "").strip()
    presupuesto = None
    if presupuesto_texto:
        try:
            presupuesto = float(presupuesto_texto)
            if presupuesto <= 0:
                presupuesto = None
        except ValueError:
            presupuesto = None

    # Horizonte del pronóstico: mismo criterio que el presupuesto — un
    # valor vacío o inválido no bloquea la subida, ejecutar_sistema cae
    # a HORIZONTE_MESES_DEFAULT (y acota a [MIN, MAX]) en vez de fallar.
    horizonte_texto = request.form.get("horizonte_meses", "").strip()
    horizonte_meses = None
    if horizonte_texto:
        try:
            horizonte_meses = int(horizonte_texto)
        except ValueError:
            horizonte_meses = None

    try:
        resultado = ejecutar_sistema(
            ruta_csv, carpeta_resultados(session["empresa_id"]),
            presupuesto_capital_trabajo=presupuesto,
            horizonte_meses=horizonte_meses,
        )
    except ValueError as e:
        # Errores esperados y ya traducidos a un mensaje accionable —
        # ver carga_datos.py, features.py y sistema_prediccion.py — se
        # muestran tal cual, sin agregar nada técnico encima.
        flash(str(e))
        return redirect(url_for("main.index"))
    except Exception as e:
        # Cualquier otra falla no prevista: se avisa de forma clara que
        # el problema está en el archivo, y se agrega el detalle técnico
        # al final por si hace falta para diagnosticarlo (esta pantalla
        # es solo para el administrador).
        flash(
            "No se pudo procesar el archivo. Revisa que las columnas tengan el formato "
            f"esperado (ver la lista de columnas más abajo). Detalle técnico: {e}"
        )
        return redirect(url_for("main.index"))

    guardar_ultimo_resultado(resultado, session["empresa_id"])

    registrar_actividad(
        session["empresa_id"], session.get("username"), session.get("nombre") or session.get("username"),
        "archivo_subido", f"Subió el archivo \"{nombre_seguro}\" y generó un nuevo pronóstico.",
    )

    # Después de procesar, se manda directo al dashboard (la pantalla
    # que vería un usuario real). La pantalla técnica queda disponible
    # aparte, en /desarrollo, para quien la necesite.
    return redirect(url_for("main.dashboard"))


@main_bp.route("/desarrollo", methods=["GET"])
@admin_required
def desarrollo():
    """Pantalla técnica: comparación de modelos, métricas, importancia
    de características, segmentación. Pensada para desarrollo/validación,
    no para el usuario final del sistema."""

    resultado = cargar_ultimo_resultado(session["empresa_id"])

    if resultado is None:
        flash("Todavía no has subido ningún archivo. Sube un CSV primero.")
        return redirect(url_for("main.index"))

    return render_template("desarrollo.html", r=resultado)


@main_bp.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    """Pantalla real del sistema: solo la demanda pronosticada del
    próximo mes por producto, sin tecnicismos de modelos ni métricas."""

    resultado = cargar_ultimo_resultado(session["empresa_id"])

    if resultado is None:
        return pantalla_sin_datos()

    return render_template("dashboard.html", r=resultado)


@main_bp.route("/reporte-demanda", methods=["GET"])
@login_required
def reporte_demanda():
    """Descarga en CSV la misma tabla que se ve en el dashboard (demanda
    pronosticada por producto y mes) — disponible tanto para admin como
    para empleado, a diferencia de /descargar (que es solo para el
    admin y sirve los archivos técnicos del panel de desarrollo).

    Acepta ?desde=N&hasta=M (1-based, inclusive, sobre la lista de
    meses pronosticados) para descargar solo un rango de meses — el
    mismo filtro visual del dashboard arma este link. Sin esos
    parámetros, se descarga el horizonte completo como siempre."""
    resultado = cargar_ultimo_resultado(session["empresa_id"])
    if resultado is None:
        return pantalla_sin_datos()

    meses_legibles = resultado["meses_pronosticados_legibles"]
    total_meses_disponibles = len(meses_legibles)

    desde = request.args.get("desde", type=int) or 1
    hasta = request.args.get("hasta", type=int) or total_meses_disponibles
    desde, hasta = sorted((desde, hasta))
    desde = max(1, min(desde, total_meses_disponibles))
    hasta = max(1, min(hasta, total_meses_disponibles))
    # Índices 0-based para recortar las listas.
    i_desde, i_hasta = desde - 1, hasta - 1

    meses_filtrados = meses_legibles[i_desde:i_hasta + 1]
    total_meses = len(meses_filtrados)

    buffer = io.StringIO()
    escritor = csv.writer(buffer)

    nombre_empresa = cargar_nombre_empresa(session["empresa_id"])
    if nombre_empresa:
        escritor.writerow(["Empresa", nombre_empresa])
        escritor.writerow([])

    escritor.writerow(
        ["Producto", "Nombre", "Categoría", *meses_filtrados, f"Total {total_meses} mes(es)"]
    )
    for fila in resultado["pronostico_pivot"]:
        valores_filtrados = fila["valores"][i_desde:i_hasta + 1]
        escritor.writerow([
            fila["producto_id"],
            fila.get("nombre_producto") or "",
            fila.get("categoria") or "",
            *valores_filtrados,
            round(sum(valores_filtrados)),
        ])

    # "utf-8-sig" agrega el BOM que Excel necesita para mostrar bien los
    # acentos y la "ñ" al abrir el CSV directamente (sin el BOM, Excel
    # en Windows muestra los caracteres especiales mal codificados).
    contenido = buffer.getvalue().encode("utf-8-sig")
    return Response(
        contenido,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=reporte_demanda.csv"}
    )


@main_bp.route("/presupuesto", methods=["GET"])
@login_required
def presupuesto():
    """Pantalla de reposición, presupuesto y costos: qué pedir, cuándo,
    y cuánto cuesta — separada del dashboard de demanda para no mezclar
    "cuánto se va a vender" con "cuánto sale reponerlo"."""

    resultado = cargar_ultimo_resultado(session["empresa_id"])

    if resultado is None:
        return pantalla_sin_datos()

    pedidos = cargar_pedidos(session["empresa_id"])

    return render_template("presupuesto.html", r=resultado, pedidos=pedidos)


@main_bp.route("/calendario", methods=["GET"])
@login_required
def calendario():
    """Calendario de reposición: en qué fecha hay que pedir cada
    producto y cuánto, en un solo vistazo — la misma información de
    "Próximos pedidos" y las alertas urgentes de /presupuesto, pero
    organizada por fecha en vez de por tabla."""
    resultado = cargar_ultimo_resultado(session["empresa_id"])
    if resultado is None:
        return pantalla_sin_datos()

    # La "fecha estimada de pedido" no viene del pronóstico (que solo
    # llega hasta horizonte_meses) — es una extrapolación en línea recta
    # de inventario_actual / consumo diario promedio, así que para un
    # producto con mucho stock y poca venta puede dar una fecha muy
    # lejana, sin relación real con lo que el sistema pronosticó. Se
    # limita acá el calendario al mismo horizonte configurado al subir
    # el archivo, para no mostrar fechas más allá de lo que en verdad
    # se calculó con confianza.
    horizonte_meses = resultado.get("horizonte_meses") or HORIZONTE_MESES_DEFAULT
    limite_dias = horizonte_meses * DIAS_POR_MES

    eventos = [
        {
            "fecha": fila["fecha_estimada_pedido"],
            "producto_id": fila["producto_id"],
            "nombre_producto": fila.get("nombre_producto"),
            "categoria": fila.get("categoria"),
            "cantidad": fila.get("cantidad_sugerida_pedido"),
            "costo": fila.get("costo_estimado_pedido"),
            "proveedor": fila.get("proveedor_principal"),
            "urgente": fila["ordenar"] == "SI",
        }
        for fila in resultado["reorder"]
        if fila.get("fecha_estimada_pedido") and (fila.get("dias_para_pedido") or 0) <= limite_dias
    ]

    return render_template("calendario.html", r=resultado, eventos=eventos)


@main_bp.route("/inventario", methods=["GET"])
@login_required
def inventario():
    """Pantalla de estado actual del inventario: cuánto stock hay hoy
    por producto, cuáles están en alerta y cuánto capital representa,
    sin mezclarlo con el pronóstico ni con las sugerencias de pedido."""

    resultado = cargar_ultimo_resultado(session["empresa_id"])

    if resultado is None:
        return pantalla_sin_datos()

    return render_template("inventario.html", r=resultado)


@main_bp.route("/pedido", methods=["GET"])
@login_required
def pedido():
    """Pantalla de acción: de los productos en alerta, cuáles ya se
    pidieron (con proveedor elegido, quién y cuándo) y cuáles siguen
    pendientes de confirmar. El historial de pedidos vive aparte del
    análisis (pedidos.json) — subir un CSV nuevo no borra lo que ya se
    pidió."""

    resultado = cargar_ultimo_resultado(session["empresa_id"])
    if resultado is None:
        return pantalla_sin_datos()

    pedidos = cargar_pedidos(session["empresa_id"])
    alertas = [fila for fila in resultado["reorder"] if fila["ordenar"] == "SI"]

    pendientes = []
    confirmados = []
    for fila in alertas:
        pid = str(fila["producto_id"])
        if pid in pedidos:
            confirmados.append({**fila, **pedidos[pid]})
        else:
            pendientes.append(fila)

    pendientes.sort(key=lambda f: f.get("costo_estimado_pedido") or 0, reverse=True)
    confirmados.sort(key=lambda f: f.get("fecha_hora", ""), reverse=True)

    return render_template("pedido.html", r=resultado, pendientes=pendientes, confirmados=confirmados)


@main_bp.route("/pedido/confirmar", methods=["POST"])
@login_required
def pedido_confirmar():
    resultado = cargar_ultimo_resultado(session["empresa_id"])
    if resultado is None:
        flash("Todavía no has subido ningún archivo. Sube un CSV primero.")
        return redirect(url_for("main.index"))

    productos_por_id = {str(fila["producto_id"]): fila for fila in resultado["reorder"]}
    seleccionados = request.form.getlist("confirmar")

    if not seleccionados:
        flash("No seleccionaste ningún producto para confirmar.")
        return redirect(url_for("main.pedido"))

    usuario = session.get("nombre") or session.get("username") or "Desconocido"
    confirmados_ahora = 0
    for producto_id in seleccionados:
        fila = productos_por_id.get(producto_id)
        if fila is None:
            continue
        # La cantidad y el costo se toman del análisis vigente (fila),
        # no de lo que venga en el formulario: el usuario elige el
        # proveedor, pero no debería poder alterar cantidades ni costos
        # a mano.
        proveedor = request.form.get(f"proveedor_{producto_id}") or "Sin especificar"
        registrar_pedido(
            empresa_id=session["empresa_id"],
            producto_id=producto_id,
            proveedor=proveedor,
            cantidad=fila.get("cantidad_sugerida_pedido"),
            costo=fila.get("costo_estimado_pedido"),
            usuario=usuario,
        )
        confirmados_ahora += 1

    if confirmados_ahora:
        lista = ", ".join(seleccionados[:5]) + ("…" if len(seleccionados) > 5 else "")
        registrar_actividad(
            session["empresa_id"], session.get("username"), usuario,
            "pedido_confirmado", f"Confirmó {confirmados_ahora} pedido(s): {lista}",
        )

    flash(f"Se confirmaron {confirmados_ahora} pedido(s).")
    return redirect(url_for("main.pedido"))


@main_bp.route("/pedido/deshacer/<producto_id>", methods=["POST"])
@login_required
def pedido_deshacer(producto_id):
    quitar_pedido(session["empresa_id"], producto_id)
    registrar_actividad(
        session["empresa_id"], session.get("username"), session.get("nombre") or session.get("username"),
        "pedido_deshecho", f"Deshizo la confirmación del pedido de {producto_id}.",
    )
    flash("Se deshizo la confirmación del pedido.")
    return redirect(url_for("main.pedido"))


@main_bp.route("/descargar/<nombre_archivo>")
@admin_required
def descargar(nombre_archivo):
    return send_from_directory(
        carpeta_resultados(session["empresa_id"]),
        nombre_archivo,
        as_attachment=True
    )


@main_bp.route("/imagenes/<nombre_archivo>")
@admin_required
def imagenes(nombre_archivo):
    return send_from_directory(
        carpeta_resultados(session["empresa_id"]),
        nombre_archivo
    )


@main_bp.route("/perfil")
@login_required
def perfil():
    """Perfil personal: datos de la cuenta y los movimientos que este
    usuario hizo (confirmar/deshacer pedidos, o si es admin, sus
    subidas de archivo y las cuentas que creó)."""
    movimientos = listar_propia(session["empresa_id"], session["username"])
    usuario = buscar_usuario(session["username"])
    return render_template("perfil.html", movimientos=movimientos, usuario=usuario)


@main_bp.route("/actividad")
@admin_required
def actividad_empresa():
    """Solo para el admin: los movimientos de todos los usuarios de su
    empresa, no solo los propios — una bitácora/auditoría completa."""
    movimientos = listar_empresa(session["empresa_id"])
    return render_template("actividad.html", movimientos=movimientos)
