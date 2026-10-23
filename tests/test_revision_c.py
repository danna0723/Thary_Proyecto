"""
Pruebas de la corrección C de la revisión (contradicciones entre requisitos,
escenarios y comportamiento).

- C2: el límite de intentos por minuto es global solo con un proceso de
  Gunicorn, así que el Procfile y el README deben decir lo mismo.
- C3: la cantidad sugerida muestra la regla usada y el texto de ayuda no
  presenta EOQ como salida general.
- C4: la política "un admin puede crear otros admins en su empresa" queda
  escrita en la pantalla de usuarios.
"""

from pathlib import Path

from app.routes import auth as auth_routes  # noqa: F401  (registra los límites)
from app.extensions import limiter  # noqa: F401

RAIZ = Path(__file__).resolve().parent.parent


def _leer(ruta):
    return (RAIZ / ruta).read_text(encoding="utf-8")


class TestLimiteDeIntentosC2:
    def test_procfile_usa_un_solo_proceso_con_hilos(self):
        linea = [l for l in _leer("Procfile").splitlines() if l.startswith("web:")][0]
        assert "--workers 1" in linea
        assert "--threads" in linea
        # Ningún worker extra: el contador en memoria tiene que ser único.
        assert "--workers 2" not in linea

    def test_readme_coincide_con_el_procfile(self):
        readme = _leer("README.md")
        assert "--workers 1 --threads 4" in readme
        assert "--workers 2" not in readme
        assert "por IP" in readme

    def test_login_y_registro_tienen_el_limite(self):
        codigo = _leer("app/routes/auth.py")
        assert codigo.count('@limiter.limit("10 per minute")') == 2


class TestCantidadSugeridaC3:
    def test_tabla_de_presupuesto_muestra_la_regla_por_producto(self):
        plantilla = _leer("app/templates/presupuesto.html")
        assert "fila.regla_cantidad" in plantilla

    def test_ayuda_no_presenta_eoq_como_salida_general(self):
        plantilla = _leer("app/templates/presupuesto.html")
        assert "depende de la regla indicada debajo" in plantilla
        assert "solo si hay costo de pedido y de mantenimiento" in plantilla

    def test_reglas_posibles_son_las_que_calcula_reposicion(self):
        reposicion = _leer("app/services/prediccion/reposicion.py")
        for regla in ('regla_cantidad = "EOQ"',
                      'regla_cantidad = "punto de reorden - inventario"',
                      'regla_cantidad = "un mes de demanda',
                      'regla_cantidad = "sin necesidad de pedido"'):
            assert regla in reposicion


class TestPoliticaDeAdminesC4:
    def test_pantalla_de_usuarios_declara_que_admin_puede_crear_admin(self):
        plantilla = _leer("app/templates/usuarios.html")
        assert "Un admin puede crear otras cuentas de admin dentro de su misma empresa" in plantilla

    def test_la_ruta_permite_elegir_rol_solo_dentro_de_la_empresa(self):
        codigo = _leer("app/routes/auth.py")
        assert 'rol=rol, empresa_id=session["empresa_id"]' in codigo
