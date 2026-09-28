"""Preprocesado clasico de vision por computador para lesiones cutaneas.

Dos tecnicas estandar en la literatura dermatologica, pensadas para atacar el
cambio de dominio desde la imagen en lugar de desde el modelo:

    eliminacion de pelo   DullRazor (Lee et al., 1997): el vello tapa bordes y
                          textura, y aparece mucho mas en fotos de movil que en
                          dermatoscopia (donde el gel y la presion lo aplastan).
    constancia de color   Shades of Gray (Finlayson y Trezzi, 2004): estima el
                          color de la luz y lo neutraliza. Es la fuente de
                          variacion mas evidente entre fotos de movil, tomadas
                          sin luz controlada.

Se aplican ANTES de la augmentation, tanto en entrenamiento como en evaluacion,
y se activan desde la seccion `preprocess` de la configuracion. Asi se pueden
medir como experimento: mismo modelo con y sin preprocesado.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

# Por encima de este lado se reduce la imagen antes de preprocesar. Las fotos de
# PAD-UFES-20 llegan a >1000 px y el inpainting escala con el area; al modelo le
# llegan a 224 px de todos modos.
MAX_LADO = 512


def limitar_tamano(rgb: np.ndarray, max_lado: int = MAX_LADO) -> np.ndarray:
    h, w = rgb.shape[:2]
    escala = max_lado / max(h, w)
    if escala >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)


def mascara_pelo(rgb: np.ndarray, kernel: int = 17, umbral: int = 10) -> np.ndarray:
    """Mascara binaria del vello (DullRazor).

    El pelo es una estructura oscura, fina y alargada. El filtro black-hat
    (cierre morfologico menos la imagen) resalta justo eso: elementos oscuros
    mas estrechos que el elemento estructurante. La lesion, mucho mas ancha, no
    responde. Se escala el kernel con la imagen para que funcione a cualquier
    resolucion.
    """
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = max(3, int(kernel * max(rgb.shape[:2]) / MAX_LADO) | 1)
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, elemento)
    _, mascara = cv2.threshold(blackhat, umbral, 255, cv2.THRESH_BINARY)
    # Dilatacion ligera: el borde del pelo tambien esta contaminado.
    return cv2.dilate(mascara, np.ones((3, 3), np.uint8), iterations=1)


def quitar_pelo(rgb: np.ndarray, kernel: int = 17, umbral: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (imagen sin pelo, mascara). Rellena el vello por inpainting."""
    mascara = mascara_pelo(rgb, kernel, umbral)
    limpia = cv2.inpaint(rgb, mascara, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return limpia, mascara


def shades_of_gray(rgb: np.ndarray, p: int = 6) -> np.ndarray:
    """Constancia de color Shades of Gray.

    Supone que la media de orden p de cada canal deberia ser gris; la desviacion
    es el color de la iluminacion y se corrige canal a canal. p=1 es Gray World
    y p->inf es Max-RGB; p=6 es el valor habitual en dermatologia (fue parte de
    las soluciones ganadoras de ISIC 2019).
    """
    img = rgb.astype(np.float32)
    iluminante = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    iluminante /= np.linalg.norm(iluminante) + 1e-8
    corregida = img / (iluminante * np.sqrt(3) + 1e-8)
    return np.clip(corregida, 0, 255).astype(np.uint8)


def preprocesar(rgb: np.ndarray, pelo: bool = False, color: bool = False) -> np.ndarray:
    if not (pelo or color):
        return rgb
    out = limitar_tamano(rgb)
    if pelo:
        out, _ = quitar_pelo(out)
    if color:
        out = shades_of_gray(out)
    return out


def pasos(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Todas las etapas intermedias, para las figuras de la memoria."""
    reducida = limitar_tamano(rgb)
    sin_pelo, mascara = quitar_pelo(reducida)
    return {
        "original": reducida,
        "mascara de pelo": mascara,
        "sin pelo": sin_pelo,
        "sin pelo + color": shades_of_gray(sin_pelo),
    }


class Preprocesado(A.ImageOnlyTransform):
    """Adaptador para usarlo como primer paso de un pipeline de albumentations."""

    def __init__(self, pelo: bool = False, color: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.pelo = pelo
        self.color = color

    def apply(self, img, **params):
        return preprocesar(img, pelo=self.pelo, color=self.color)

    def get_transform_init_args_names(self):
        return ("pelo", "color")
