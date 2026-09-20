from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Instancia a nivel de módulo (no dentro de create_app) para que las
# rutas puedan importarla y decorar endpoints puntuales con @limiter.limit(...)
# sin import circular. Se conecta a la app real con limiter.init_app(app).
# Sin storage explícito usa memoria del propio proceso — suficiente para
# esta escala (una sola instancia), pero con más de un worker de Gunicorn
# el límite es por worker, no global (ver Procfile).
limiter = Limiter(key_func=get_remote_address, default_limits=[])
