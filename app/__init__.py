import os
from flask import Flask, url_for, session, flash, redirect
from werkzeug.exceptions import RequestEntityTooLarge


def create_app():

    app = Flask(__name__)

    # Flask firma las cookies de sesión con esta clave (la sesión guarda el
    # rol y la empresa del usuario), así que quien la conozca puede
    # fabricar una sesión de administrador sin contraseña. Por eso NO
    # puede estar escrita en el código (que va a GitHub): se lee de la
    # variable de entorno SECRET_KEY, y si falta, la app se niega a
    # arrancar en vez de usar una clave conocida. Solo "python run.py"
    # (desarrollo local) define una clave de desarrollo — ver run.py.
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise RuntimeError(
            "Falta la variable de entorno SECRET_KEY. Defínela con un valor "
            "aleatorio largo (en Railway: pestaña Variables del servicio). "
            "Para generar uno: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    app.config["SECRET_KEY"] = secret_key

    # Límite de tamaño de subida (protección básica contra denegación de
    # servicio por archivos enormes): el CSV real de referencia pesa ~1 MB,
    # así que 10 MB da margen generoso para catálogos más grandes sin dejar
    # el límite abierto del todo.
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    from app.extensions import limiter
    limiter.init_app(app)

    def static_url(filename):
        """
        Como url_for('static', ...), pero agrega "?v=<fecha de
        modificación del archivo>" al final. Sin esto, el navegador
        puede seguir usando una copia vieja en caché de un CSS/JS
        aunque el archivo en el servidor ya haya cambiado — pasó
        justamente con style.css durante el desarrollo (se corregían
        los estilos pero el usuario seguía viendo los de antes). Al
        cambiar el archivo, cambia su fecha de modificación, cambia el
        "?v=", y el navegador lo trata como una URL nueva.
        """
        ruta = os.path.join(app.static_folder, filename)
        try:
            version = int(os.path.getmtime(ruta))
        except OSError:
            version = 0
        return url_for("static", filename=filename, v=version)

    app.jinja_env.globals["static_url"] = static_url

    def formato_moneda(valor):
        """
        Formatea un monto en pesos colombianos: separador de miles con
        punto y sin decimales (ej. 1234567 -> "1.234.567"), la
        convención local — al revés de "{:,.0f}" de Python, que separa
        los miles con coma (convención de EE. UU.).
        """
        return "{:,.0f}".format(valor).replace(",", ".")

    app.jinja_env.filters["moneda"] = formato_moneda

    # Sistema multiempresa: cada admin que se registra crea su propia
    # "empresa" (un id generado en el momento) con su propio inventario
    # — CSV subido, pronóstico, pedidos — completamente separado del de
    # cualquier otro admin. Estas carpetas siguen siendo archivos de
    # verdad (el CSV subido, gráficas .png, el modelo entrenado) — lo
    # único que se movió a MySQL es la parte tabular (usuarios, empresas,
    # pedidos, y el resultado de cada corrida del pipeline; ver
    # app/services/db.py).
    app.config["UPLOAD_FOLDER_BASE"] = os.path.join(os.getcwd(), "app", "uploads")
    app.config["RESULTADOS_FOLDER_BASE"] = os.path.join(os.getcwd(), "app", "resultados")
    os.makedirs(app.config["UPLOAD_FOLDER_BASE"], exist_ok=True)
    os.makedirs(app.config["RESULTADOS_FOLDER_BASE"], exist_ok=True)

    from app.services.db import crear_tablas
    crear_tablas()

    # Asistente de IA (Ollama corriendo en la misma máquina, sin mandar
    # nada a un servicio externo). Configurable por variable de entorno
    # por si el modelo o el puerto cambian en otra instalación.
    # DESACTIVADO temporalmente (en evaluación si se quita definitivamente).
    # app.config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    # app.config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "llama3.2")

    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    # from app.routes.asistente import asistente_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    # app.register_blueprint(asistente_bp)

    # Sin esto, un archivo que supera MAX_CONTENT_LENGTH (ver arriba)
    # muestra la página cruda de error 413 de Werkzeug, en inglés y sin
    # nada del estilo del sistema — acá se atrapa esa excepción y se
    # redirige a la pantalla de carga con el mismo tipo de aviso que ya
    # usan los demás errores del formulario (ver procesar() en
    # app/routes/main.py).
    @app.errorhandler(RequestEntityTooLarge)
    def archivo_demasiado_grande(error):
        limite_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        flash(f"El archivo supera el límite de {limite_mb} MB. Sube un CSV más pequeño.")
        return redirect(url_for("main.index"))

    @app.context_processor
    def inyectar_nombre_empresa():
        """Disponible como "nombre_empresa" en cualquier template que
        extienda base.html (se usa en el sidebar) sin que cada ruta
        tenga que pasarlo a mano — login/registro no tienen empresa_id
        en sesión todavía, así que ahí queda None."""
        if "empresa_id" not in session:
            return {"nombre_empresa": None}
        from app.services.perfil_empresa import cargar_nombre_empresa
        return {"nombre_empresa": cargar_nombre_empresa(session["empresa_id"])}

    return app
