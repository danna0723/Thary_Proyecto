from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Instancia a nivel de módulo (no dentro de create_app) para que las
# rutas puedan importarla y decorar endpoints puntuales con @limiter.limit(...)
# sin import circular. Se conecta a la app real con limiter.init_app(app).
# Sin storage explícito usa memoria del propio proceso. El límite de 10
# intentos por minuto por IP es global solo con un único proceso de
# Gunicorn: el Procfile usa --workers 1 --threads 4. Con más workers, cada
# uno tendría su propio contador.
limiter = Limiter(key_func=get_remote_address, default_limits=[])
