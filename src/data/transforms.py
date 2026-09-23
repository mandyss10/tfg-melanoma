"""Transformaciones de imagen.

El punto central del trabajo esta aqui: `mobile_domain_transform` simula sobre
imagenes de dermatoscopio las degradaciones tipicas de una foto de movil, para
reducir el domain gap sin necesidad de miles de fotos caseras etiquetadas.

Cada degradacion replica un fenomeno fisico concreto observable en PAD-UFES-20:

    iluminacion   la foto casera no tiene luz controlada ni difusor
    reflejos      el flash produce brillos especulares sobre piel grasa
    desenfoque    el enfoque automatico falla a corta distancia
    ruido ISO     sensores pequenos con poca luz
    compresion    JPEG agresivo de las camaras de movil
    encuadre      distancia y angulo variables, lesion descentrada
    balance color temperatura de color no calibrada
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Estadisticas de ImageNet: los backbones preentrenados las esperan.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize(size: int) -> list:
    return [
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def baseline_transform(size: int = 224, train: bool = True) -> A.Compose:
    """Augmentation estandar de clasificacion de imagen.

    Es la linea base contra la que se compara: solo transformaciones
    geometricas y de color suaves, sin simular ningun dominio.
    """
    if not train:
        return A.Compose(_normalize(size))

    return A.Compose(
        [
            A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def mobile_domain_transform(size: int = 224, train: bool = True, strength: float = 1.0) -> A.Compose:
    """Augmentation que simula las condiciones de una camara de movil.

    Args:
        size: lado de la imagen de salida.
        train: si es False devuelve solo resize + normalizacion.
        strength: escala global de la intensidad de las degradaciones. Permite
            hacer un barrido (0.5 / 1.0 / 1.5) y estudiar si existe un punto a
            partir del cual degradar mas empeora el resultado.

    La intensidad se controla con `strength` en vez de fijarla, porque una
    augmentation demasiado agresiva destruye la textura de la lesion (que es
    justo la senal diagnostica) y hace caer el rendimiento en ambos dominios.
    """
    if not train:
        return A.Compose(_normalize(size))

    s = strength

    return A.Compose(
        [
            # --- encuadre: distancia y angulo variables, lesion descentrada ---
            A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                rotate=(-45, 45),
                shear=(-10, 10),
                translate_percent=(-0.1, 0.1),
                p=0.7,
            ),

            # --- iluminacion no controlada ---
            A.RandomBrightnessContrast(
                brightness_limit=0.35 * s, contrast_limit=0.35 * s, p=0.8
            ),
            # Sombra proyectada por la propia mano o el telefono
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.3 * s),

            # --- balance de blancos sin calibrar ---
            A.ColorJitter(
                brightness=0.0, contrast=0.0, saturation=0.3 * s, hue=0.06 * s, p=0.7
            ),

            # --- reflejos especulares del flash ---
            # RandomSunFlare simula un brillo circular saturado, que es
            # visualmente equivalente al reflejo del flash sobre piel grasa.
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                src_radius=int(60 * s),
                num_flare_circles_range=(1, 3),
                p=0.25 * s,
            ),

            # --- optica: desenfoque por autofoco fallido o pulso ---
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 7)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Defocus(radius=(2, 5)),
                ],
                p=0.4 * s,
            ),

            # --- sensor: ruido ISO con poca luz ---
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.05, 0.15)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ],
                p=0.4 * s,
            ),

            # --- compresion JPEG del pipeline de la camara ---
            A.ImageCompression(quality_range=(40, 90), p=0.5 * s),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_transform(name: str, size: int = 224, train: bool = True, **kwargs) -> A.Compose:
    """Selector por nombre, para poder fijarlo desde un fichero de configuracion."""
    builders = {
        "baseline": baseline_transform,
        "mobile": mobile_domain_transform,
    }
    if name not in builders:
        raise ValueError(f"transform desconocida: {name!r}. Opciones: {list(builders)}")
    if name == "baseline":
        kwargs.pop("strength", None)
    return builders[name](size=size, train=train, **kwargs)
