# Thary — Sistema de apoyo a la decisión para pronóstico de demanda de inventario

Sistema web que predice la demanda mensual de productos a partir del historial de consumo (archivo CSV), y a partir de ese pronóstico genera recomendaciones de reposición: punto de reorden, cantidad económica de pedido (EOQ), alertas de desabastecimiento y costo de ruptura estimado.

- **Demo desplegada:** [https://web-production-6ac7e5.up.railway.app](https://web-production-6ac7e5.up.railway.app)
- **Stack:** Python, Flask, XGBoost, scikit-learn, pandas, MySQL

## Requisitos

- Python 3.12 o superior
- Un servidor MySQL accesible (local o remoto)

## Instalación

1. Clonar el repositorio y entrar a la carpeta:

   ```bash
   git clone https://github.com/danna0723/Thary_Proyecto.git
   cd Thary_Proyecto
   ```

2. Crear un entorno virtual e instalar las dependencias:

   ```bash
   python -m venv venv
   venv\Scripts\activate       # En Windows
   # source venv/bin/activate  # En Linux/Mac
   pip install -r requirements.txt
   ```

3. Crear en MySQL un usuario con permisos sobre la base de datos del sistema (cambia `<tu_contraseña>` por una propia):

   ```sql
   CREATE USER 'thary_app'@'localhost' IDENTIFIED BY '<tu_contraseña>';
   GRANT ALL PRIVILEGES ON sistema_inventario.* TO 'thary_app'@'localhost';
   FLUSH PRIVILEGES;
   ```

   Al iniciar, la aplicación crea automáticamente la base de datos (si el usuario tiene permiso) y todas las tablas necesarias. No hace falta crearlas a mano.

4. Configurar las variables de entorno de conexión a MySQL. Si no se definen, el sistema usa por defecto `127.0.0.1`, puerto `3307`, usuario `thary_app`, base de datos `sistema_inventario` y contraseña vacía.

   Solo hace falta definir las que sean distintas a esos valores por defecto; por ejemplo, si MySQL corre en el puerto estándar 3306, hay que definir `DB_PORT`. En PowerShell (Windows):

   ```powershell
   $env:DB_PORT = "3306"
   $env:DB_PASSWORD = "<tu_contraseña>"
   ```

   En bash (Linux/Mac):

   ```bash
   export DB_PORT=3306
   export DB_PASSWORD="<tu_contraseña>"
   ```

   Estas variables solo duran mientras la terminal esté abierta. En modo de desarrollo (`python run.py`) no hace falta definir `SECRET_KEY`: se usa automáticamente una clave de desarrollo.

5. Ejecutar la aplicación:

   ```bash
   python run.py
   ```

6. Abrir en el navegador: [http://localhost:5000](http://localhost:5000)

## Ejecutar las pruebas

```bash
pytest tests/ -q
```

## Despliegue en producción

El sistema está pensado para desplegarse en [Railway](https://railway.app), con dos servicios: la aplicación web (conectada a este repositorio) y una instancia de MySQL.

Variables de entorno que hay que configurar en el servicio web, además de las de conexión a MySQL:

- `SECRET_KEY`: una cadena aleatoria larga, usada para firmar las sesiones de los usuarios. Sin esta variable, la aplicación no arranca. Se puede generar con:

  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```

El arranque en producción usa Gunicorn, configurado en el archivo `Procfile`:

```
web: gunicorn run:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
```

Se usa un solo proceso (`--workers 1`) con hilos. El límite de intentos de login y registro (10 por minuto, por IP) se guarda en memoria del proceso: con varios workers cada uno tendría su propio contador y el límite no sería global.

Railway detecta automáticamente la versión de Python (`.python-version`) e instala las dependencias de `requirements.txt`.

## Estructura del proyecto

```
app/
    routes/       # Rutas de Flask (auth.py, main.py)
    services/
        auth.py, db.py, pedidos.py, actividad.py
        prediccion/               # Pipeline de predicción
            carga_datos.py        # 1. Carga y limpieza
            segmentacion.py       # 2. Segmentación ABC/variabilidad
            features.py           # 3. Construcción de variables
            modelos.py            # 4. Entrenamiento y selección
            regresion_lineal.py   # 4.1. Modelo de Regresión Lineal
            validacion.py         # 5. Validación cruzada temporal
            pronostico_futuro.py  # 6. Pronóstico recursivo
            reposicion.py         # 7. Puntos de reorden y EOQ
            graficas.py           # 8. Generación de gráficos
            exportacion.py        # 9. Exportación de resultados
    templates/    # Vistas HTML
tests/            # Pruebas automatizadas (pytest)
```
