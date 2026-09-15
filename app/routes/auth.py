from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from app.services.auth import (
    admin_required,
    crear_usuario,
    listar_usuarios,
    verificar_credenciales,
)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identificador = request.form.get("identificador", "").strip()
        password = request.form.get("password", "")

        usuario = verificar_credenciales(identificador, password)
        if usuario is None:
            flash("Usuario o contraseña incorrectos.")
            return redirect(url_for("auth.login"))

        session["username"] = usuario["username"]
        session["nombre"] = usuario["nombre"]
        session["rol"] = usuario.get("rol", "empleado")
        session["empresa_id"] = usuario["empresa_id"]

        # El admin arranca en "Subir archivo" (su tarea principal); un
        # empleado no tiene acceso a esa pantalla, así que arranca en
        # el dashboard.
        destino = "main.index" if session["rol"] == "admin" else "main.dashboard"
        return redirect(url_for(destino))

    return render_template("login.html")


@auth_bp.route("/registro", methods=["GET", "POST"])
def registro():
    """
    Registro público: cualquiera puede crear una cuenta acá, y siempre
    queda como admin de una empresa (inventario) nueva y propia — no se
    une a una empresa existente. Las cuentas de empleados, en cambio,
    las crea cada admin desde /usuarios/nuevo para su propia empresa
    (ver más abajo), no un registro público.
    """
    if request.method == "POST":
        nombre = request.form.get("nombre", "")
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")

        usuario, error = crear_usuario(nombre, username, email, password, rol="admin")
        if error:
            flash(error)
            return redirect(url_for("auth.registro"))

        session["username"] = usuario["username"]
        session["nombre"] = usuario["nombre"]
        session["rol"] = usuario["rol"]
        session["empresa_id"] = usuario["empresa_id"]
        flash(f"Cuenta de administrador creada. ¡Bienvenido/a, {usuario['nombre']}!")
        return redirect(url_for("main.index"))

    return render_template("registro.html")


@auth_bp.route("/usuarios/nuevo", methods=["GET", "POST"])
@admin_required
def usuarios_nuevo():
    """Pantalla del admin para crear cuentas de empleados (o de otro
    admin) dentro de su propia empresa. No es un registro público."""
    if request.method == "POST":
        nombre = request.form.get("nombre", "")
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        rol = request.form.get("rol", "empleado")

        usuario, error = crear_usuario(nombre, username, email, password, rol=rol, empresa_id=session["empresa_id"])
        if error:
            flash(error)
        else:
            flash(f"Cuenta creada para {usuario['nombre']} ({usuario['rol']}).")
        return redirect(url_for("auth.usuarios_nuevo"))

    return render_template("usuarios.html", usuarios=listar_usuarios(session["empresa_id"]))


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Sesión cerrada.")
    return redirect(url_for("auth.login"))
