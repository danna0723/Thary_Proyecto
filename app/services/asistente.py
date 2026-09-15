import json
import urllib.error
import urllib.request

from flask import current_app

SYSTEM_PROMPT = """Eres el asistente virtual del "Sistema de Predicción de Demanda e Inventario",
una herramienta interna para pronosticar ventas y gestionar reposición de stock.

Pantallas del sistema:
- Dashboard: pronóstico de demanda del próximo mes por producto.
- Estado del inventario: stock actual por producto y cuáles están en alerta.
- Presupuesto y costos: qué conviene reponer, cuánto cuesta y cómo se reparte
  el presupuesto de capital de trabajo disponible.
- Pedido: productos en alerta pendientes de confirmar, y el historial de
  pedidos ya confirmados (proveedor, cantidad, costo, quién y cuándo).
- Panel técnico (solo para el administrador): comparación de modelos de
  predicción, métricas, importancia de variables y segmentación.
- Usuarios (solo para el administrador): alta de cuentas de empleados.

Roles: el "administrador" sube el archivo CSV de ventas y accede al panel
técnico y a la gestión de usuarios. El "empleado" ve demanda, inventario,
presupuesto y pedidos, y puede confirmar pedidos, pero no sube archivos.

Responde siempre en español neutro (usa "tú", nunca "vos"), de forma breve y
clara, ayudando a las personas que usan el sistema a entender qué significa
cada pantalla y cómo usarla. No tienes acceso a los datos cargados en este
momento (números de stock, montos, pronósticos concretos): si te preguntan
por un valor específico, indícales en qué pantalla del sistema pueden
encontrarlo en vez de inventar una cifra."""


class AsistenteNoDisponibleError(Exception):
    """Ollama no está corriendo, no tiene el modelo, o no respondió a tiempo."""


def preguntar_asistente(mensaje, historial=None):
    """
    Le manda la pregunta (más los últimos turnos de la conversación, si
    los hay) al modelo local de Ollama y devuelve el texto de la
    respuesta. Usa la API HTTP de Ollama (http://localhost:11434), que
    corre en la misma máquina — no se manda nada a un servicio externo.
    """
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes.extend(historial or [])
    mensajes.append({"role": "user", "content": mensaje})

    payload = json.dumps({
        "model": current_app.config["OLLAMA_MODEL"],
        "messages": mensajes,
        "stream": False,
    }).encode("utf-8")

    peticion = urllib.request.Request(
        current_app.config["OLLAMA_URL"],
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(peticion, timeout=120) as respuesta:
            datos = json.loads(respuesta.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise AsistenteNoDisponibleError(
            "No se pudo conectar con Ollama. ¿Está corriendo (\"ollama serve\")?"
        ) from e
    except json.JSONDecodeError as e:
        raise AsistenteNoDisponibleError("Ollama devolvió una respuesta inválida.") from e

    try:
        return datos["message"]["content"].strip()
    except (KeyError, TypeError) as e:
        raise AsistenteNoDisponibleError("Ollama devolvió una respuesta inválida.") from e
