import os

# Solo al ejecutar "python run.py" (desarrollo local) se usa una clave de
# desarrollo por defecto. Con gunicorn ("run:app", como en Railway) este
# archivo se importa como módulo, así que __name__ no es "__main__" y la
# variable SECRET_KEY tiene que venir del entorno (ver app/__init__.py).
if __name__ == "__main__":
    os.environ.setdefault("SECRET_KEY", "clave-solo-para-desarrollo-local")

from app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
