(function () {
    "use strict";

    var widget = document.getElementById("asistente-widget");
    if (!widget) return;

    var toggleBtn = document.getElementById("asistente-toggle");
    var cerrarBtn = document.getElementById("asistente-cerrar");
    var panel = document.getElementById("asistente-panel");
    var mensajesEl = document.getElementById("asistente-mensajes");
    var form = document.getElementById("asistente-form");
    var input = document.getElementById("asistente-input");

    // Historial en memoria de la pestaña actual, nada más — no se
    // guarda en el servidor ni sobrevive a un F5. Se manda junto con
    // cada pregunta para que el modelo tenga contexto de la conversación.
    var historial = [];
    var enviando = false;

    function abrirPanel() {
        panel.hidden = false;
        toggleBtn.classList.add("activo");
        input.focus();
    }

    function cerrarPanel() {
        panel.hidden = true;
        toggleBtn.classList.remove("activo");
    }

    toggleBtn.addEventListener("click", function () {
        if (panel.hidden) abrirPanel(); else cerrarPanel();
    });
    cerrarBtn.addEventListener("click", cerrarPanel);

    function agregarMensaje(texto, clase) {
        var burbuja = document.createElement("div");
        burbuja.className = "asistente-msg " + clase;
        burbuja.textContent = texto;
        mensajesEl.appendChild(burbuja);
        mensajesEl.scrollTop = mensajesEl.scrollHeight;
        return burbuja;
    }

    form.addEventListener("submit", function (e) {
        e.preventDefault();
        if (enviando) return;

        var texto = input.value.trim();
        if (!texto) return;

        agregarMensaje(texto, "asistente-msg-user");
        input.value = "";
        enviando = true;

        var indicador = agregarMensaje("Pensando...", "asistente-msg-bot asistente-msg-pensando");

        fetch("/asistente/mensaje", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mensaje: texto, historial: historial })
        })
            .then(function (resp) {
                return resp.json().then(function (datos) {
                    return { ok: resp.ok, datos: datos };
                });
            })
            .then(function (resultado) {
                indicador.remove();
                if (resultado.ok) {
                    agregarMensaje(resultado.datos.respuesta, "asistente-msg-bot");
                    historial.push({ role: "user", content: texto });
                    historial.push({ role: "assistant", content: resultado.datos.respuesta });
                    if (historial.length > 20) historial = historial.slice(-20);
                } else {
                    agregarMensaje(resultado.datos.error || "Ocurrió un error.", "asistente-msg-bot asistente-msg-error");
                }
            })
            .catch(function () {
                indicador.remove();
                agregarMensaje("No se pudo contactar al servidor.", "asistente-msg-bot asistente-msg-error");
            })
            .finally(function () {
                enviando = false;
            });
    });
})();
