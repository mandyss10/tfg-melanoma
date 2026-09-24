"""Demo interactiva: foto de una lesion -> sospecha de malignidad + Grad-CAM.

Uso (desde la raiz del proyecto):
    streamlit run app/app.py

Busca los modelos en results/<experimento>/best.pth, que es donde deja los
pesos src.train. Desde el movil se abre la URL "Network URL" que imprime
Streamlit, con el ordenador y el movil en la misma wifi.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.explain import explain_array  # noqa: E402
from src.models.classifier import load_checkpoint  # noqa: E402

# TFG_RESULTS permite apuntar a otra carpeta de modelos sin mover ficheros.
RESULTS = Path(os.environ.get("TFG_RESULTS", RAIZ / "results"))

# Orden en que se presentan: es el orden del diseno experimental.
ORDEN = ["finetune", "aug", "baseline"]
DESCRIPCION = {
    "baseline": "solo dermatoscopio",
    "aug": "dermatoscopio + degradaciones de movil simuladas",
    "finetune": "aug + ajuste con fotos reales de movil (propuesta)",
}


def modelos_disponibles() -> dict[str, Path]:
    encontrados = {p.parent.name: p for p in sorted(RESULTS.glob("*/best.pth"))}
    return dict(
        sorted(encontrados.items(), key=lambda kv: ORDEN.index(kv[0]) if kv[0] in ORDEN else 99)
    )


@st.cache_resource
def cargar(ruta: str):
    return load_checkpoint(ruta, device="cpu")


def recorte_central(rgb: np.ndarray, zoom: float) -> np.ndarray:
    """Recorte cuadrado centrado.

    Las fotos de PAD-UFES-20 son primeros planos de la lesion. Una foto de movil
    normal deja el lunar pequeno en el centro, y al reducirla a 224x224 el modelo
    apenas lo veria; el recorte lo acerca a las condiciones de entrenamiento.
    """
    h, w = rgb.shape[:2]
    lado = int(min(h, w) / zoom)
    y0, x0 = (h - lado) // 2, (w - lado) // 2
    return rgb[y0 : y0 + lado, x0 : x0 + lado]


def leer_imagen(fichero) -> np.ndarray | None:
    datos = np.frombuffer(fichero.getvalue(), np.uint8)
    bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Lesiones cutaneas - demo TFG", layout="wide")
st.title("Lesiones cutáneas en fotografía de móvil")
st.warning(
    "**Proyecto académico, no es una herramienta de diagnóstico.** El modelo marca "
    "como sospechosas muchas lesiones benignas y puede fallar con lesiones malignas. "
    "Ante cualquier lesión que cambie, sangre o preocupe, consulta a un dermatólogo."
)

modelos = modelos_disponibles()
if not modelos:
    st.error(
        f"No hay modelos en {RESULTS}. Descarga la carpeta results/ de Kaggle y "
        "colócala en la raíz del proyecto (debe quedar results/finetune/best.pth)."
    )
    st.stop()

with st.sidebar:
    st.header("Ajustes")
    nombre = st.selectbox(
        "Modelo",
        list(modelos),
        format_func=lambda n: f"{n} ({DESCRIPCION.get(n, '')})",
    )
    comparar = st.checkbox("Comparar con los demás modelos", value=len(modelos) > 1)
    zoom = st.slider(
        "Zoom al centro", 1.0, 4.0, 1.0, 0.25,
        help="Acerca la lesión si en la foto sale pequeña. Debe ocupar buena parte del recuadro.",
    )
    st.caption(
        "Consejos: lesión centrada, a unos 10 cm, con buena luz y sin flash, "
        "y la cámara enfocada."
    )

pestana_camara, pestana_archivo = st.tabs(["Hacer foto", "Subir imagen"])
with pestana_camara:
    foto = st.camera_input("Foto de la lesión")
with pestana_archivo:
    subida = st.file_uploader("Imagen de la lesión", type=["jpg", "jpeg", "png"])

fichero = foto or subida
if fichero is None:
    st.info("Haz una foto o sube una imagen para analizarla.")
    st.stop()

rgb = leer_imagen(fichero)
if rgb is None:
    st.error("No se pudo leer la imagen.")
    st.stop()
rgb = recorte_central(rgb, zoom)

model, meta = cargar(str(modelos[nombre]))
umbral = float(meta.get("threshold", 0.5))
with st.spinner("Analizando..."):
    prob, cam = explain_array(rgb, model, meta)

col_img, col_cam, col_res = st.columns([1, 1, 1])
col_img.image(rgb, caption="Imagen analizada", width="stretch")
col_cam.image(cam, caption="Grad-CAM: dónde se ha fijado el modelo", width="stretch")

with col_res:
    st.subheader("Resultado")
    if prob >= umbral:
        st.error("**Sospechosa**: el modelo la asocia a lesiones malignas.")
    else:
        st.success("**No sospechosa** para el modelo.")
    st.metric("Puntuación de malignidad", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(
        f"Umbral de decisión: {umbral:.0%}. Se fijó en validación para detectar ~95% "
        "de las lesiones malignas, a costa de muchas falsas alarmas. La puntuación "
        "no es una probabilidad real de tener cáncer."
    )
    st.caption(
        "Maligno = melanoma, carcinoma basocelular o carcinoma epidermoide. "
        "Si el mapa de calor se enciende fuera de la lesión, el resultado no es fiable."
    )

if comparar and len(modelos) > 1:
    st.subheader("Los tres modelos ante la misma foto")
    filas = []
    for otro, ruta in modelos.items():
        m, mt = cargar(str(ruta))
        p, _ = explain_array(rgb, m, mt) if otro != nombre else (prob, None)
        u = float(mt.get("threshold", 0.5))
        filas.append(
            {
                "modelo": otro,
                "entrenamiento": DESCRIPCION.get(otro, ""),
                "puntuación": f"{p:.0%}",
                "umbral": f"{u:.0%}",
                "resultado": "sospechosa" if p >= u else "no sospechosa",
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
