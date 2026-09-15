from flask import Blueprint, jsonify, request

from app.services.asistente import AsistenteNoDisponibleError, preguntar_asistente
from app.services.auth import login_required

asistente_bp = Blueprint("asistente", __name__)

MENSAJE_MAX = 1000
HISTORIAL_MAX_TURNOS = 10
TURNO_MAX = 2000


def _limpiar_historial(historial_crudo):
    """Se queda solo con lo que hace falta para mandarle a Ollama: los
    últimos turnos, cada uno con role "user"/"assistant" y contenido de
    texto. Lo que venga del cliente en un formato inesperado se descarta
    en vez de mandarlo tal cual al modelo."""
    limpio = []
    for turno in (historial_crudo or [])[-HISTORIAL_MAX_TURNOS:]:
        if not isinstance(turno, dict):
            continue
        role = turno.get("role")
        contenido = turno.get("content")
        if role in ("user", "assistant") and isinstance(contenido, str) and contenido.strip():
            limpio.append({"role": role, "content": contenido.strip()[:TURNO_MAX]})
    return limpio


@asistente_bp.route("/asistente/mensaje", methods=["POST"])
@login_required
def mensaje():
    datos = request.get_json(silent=True) or {}
    texto = (datos.get("mensaje") or "").strip()

    if not texto:
        return jsonify({"error": "Escribe una pregunta."}), 400
    if len(texto) > MENSAJE_MAX:
        return jsonify({"error": "El mensaje es demasiado largo."}), 400

    historial = _limpiar_historial(datos.get("historial"))

    try:
        respuesta = preguntar_asistente(texto, historial)
    except AsistenteNoDisponibleError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({"respuesta": respuesta})
