// Interacciones de la pantalla "Subir archivo": arrastrar-y-soltar con
// vista previa del CSV elegido, validación de extensión antes de
// llegar al servidor, y un aviso de "procesando" mientras corre el
// pipeline (la petición es un POST normal y tarda varios segundos en
// responder — sin esto, la pantalla se queda en blanco sin avisar nada).
document.addEventListener("DOMContentLoaded", function () {
    var dropzone = document.getElementById("dropzone");
    var input = document.getElementById("archivo");
    if (!dropzone || !input) return;

    var estadoVacio = document.getElementById("dropzone-vacio");
    var estadoArchivo = document.getElementById("dropzone-archivo");
    var nombreEl = document.getElementById("archivo-nombre");
    var tamanoEl = document.getElementById("archivo-tamano");
    var errorEl = document.getElementById("dropzone-error");
    var quitarBtn = document.getElementById("dropzone-quitar");
    var form = document.getElementById("form-subir");
    var overlay = document.getElementById("upload-overlay");
    var btnSubir = document.getElementById("btn-subir");

    function formatoTamano(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function esCsv(archivo) {
        return /\.csv$/i.test(archivo.name);
    }

    function mostrarArchivo(archivo) {
        errorEl.hidden = true;
        dropzone.classList.remove("dropzone-error-borde");
        if (!esCsv(archivo)) {
            errorEl.hidden = false;
            dropzone.classList.add("dropzone-error-borde");
            limpiar();
            return;
        }
        nombreEl.textContent = archivo.name;
        tamanoEl.textContent = formatoTamano(archivo.size);
        estadoVacio.hidden = true;
        estadoArchivo.hidden = false;
        dropzone.classList.add("dropzone-con-archivo");
    }

    function limpiar() {
        input.value = "";
        estadoVacio.hidden = false;
        estadoArchivo.hidden = true;
        dropzone.classList.remove("dropzone-con-archivo");
    }

    input.addEventListener("change", function () {
        if (input.files && input.files.length) mostrarArchivo(input.files[0]);
    });

    quitarBtn.addEventListener("click", function (e) {
        e.preventDefault();
        limpiar();
    });

    ["dragenter", "dragover"].forEach(function (evento) {
        dropzone.addEventListener(evento, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dropzone-activo");
        });
    });
    ["dragleave", "drop"].forEach(function (evento) {
        dropzone.addEventListener(evento, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dropzone-activo");
        });
    });
    dropzone.addEventListener("drop", function (e) {
        var archivos = e.dataTransfer.files;
        if (archivos && archivos.length) {
            input.files = archivos;
            mostrarArchivo(archivos[0]);
        }
    });

    // Al enviar: si no hay archivo o no es .csv, se bloquea acá mismo
    // (mensaje inmediato, sin esperar la respuesta del servidor). Si
    // está todo bien, se deja seguir el envío normal del formulario y
    // solo se muestra el aviso de "procesando" — la navegación a la
    // página siguiente la sigue haciendo el propio navegador.
    form.addEventListener("submit", function (e) {
        if (!input.files || !input.files.length) {
            e.preventDefault();
            errorEl.hidden = false;
            errorEl.innerHTML = '<i class="fa fa-exclamation-triangle"></i> Selecciona un archivo CSV antes de continuar.';
            dropzone.classList.add("dropzone-error-borde");
            dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
            return;
        }
        if (!esCsv(input.files[0])) {
            e.preventDefault();
            errorEl.hidden = false;
            dropzone.classList.add("dropzone-error-borde");
            return;
        }
        btnSubir.disabled = true;
        overlay.hidden = false;
    });

    // Tarjeta "Columnas que reconoce el archivo": arranca escondida y la
    // columna de subida centrada (mx-auto) para aprovechar el espacio;
    // "Más información" la muestra y descentra la columna de subida, el
    // botón "x" adentro de la tarjeta la vuelve a esconder.
    var colSubir = document.getElementById("col-subir");
    var colColumnas = document.getElementById("col-columnas");
    var btnMostrarColumnas = document.getElementById("btn-mostrar-columnas");
    var btnCerrarColumnas = document.getElementById("btn-cerrar-columnas");
    if (colSubir && colColumnas && btnMostrarColumnas && btnCerrarColumnas) {
        var DURACION_ANIMACION_MS = 400;
        colSubir.classList.add("col-subir-centrado");
        btnMostrarColumnas.addEventListener("click", function () {
            colColumnas.hidden = false;
            colSubir.classList.remove("col-subir-centrado");
            // Fuerza un reflow entre sacar "hidden" y agregar la clase que
            // dispara la transición — si no, el navegador aplica los dos
            // cambios juntos y no hay animación de entrada.
            void colColumnas.offsetWidth;
            colColumnas.classList.add("columnas-visible");
        });
        btnCerrarColumnas.addEventListener("click", function () {
            colColumnas.classList.remove("columnas-visible");
            colSubir.classList.add("col-subir-centrado");
            setTimeout(function () {
                colColumnas.hidden = true;
            }, DURACION_ANIMACION_MS);
        });
    }
});
