"""Deteccion automatica de los datasets montados en Kaggle.

En Kaggle los datasets se montan en `/kaggle/input/<slug>/`, y el slug depende
de que copia concreta del dataset se haya adjuntado al notebook. Hay varias
subidas de ISIC y de PAD-UFES-20, con estructuras distintas, asi que fijar la
ruta en el codigo lo rompe en cuanto se cambia de fuente.

Este modulo localiza los datasets por su CONTENIDO (que columnas tiene el CSV)
en lugar de por su nombre, que es estable frente a cambios de slug.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Firmas de cada dataset: columnas que debe tener su CSV de metadatos.
FIRMA_PAD = {"img_id", "diagnostic"}
FIRMA_ISIC = {"image_name", "target"}


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists()


def _csvs(root: Path, max_depth: int = 4) -> list[Path]:
    """CSV hasta cierta profundidad; evita recorrer arboles de imagenes enteros."""
    encontrados = []
    for depth in range(max_depth):
        patron = "/".join(["*"] * depth + ["*.csv"]) if depth else "*.csv"
        encontrados.extend(root.glob(patron))
    return encontrados


def _tiene_firma(csv_path: Path, firma: set[str]) -> bool:
    try:
        # nrows=0 lee solo la cabecera: instantaneo aunque el CSV sea enorme.
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return False
    return firma.issubset(cols)


def find_dataset(firma: set[str], search_root: Path | None = None) -> tuple[Path, str] | None:
    """Busca un dataset cuyo CSV contenga todas las columnas de `firma`.

    Devuelve (directorio_raiz, nombre_del_csv) o None.
    """
    root = search_root or KAGGLE_INPUT
    if not root.exists():
        return None

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for csv_path in _csvs(dataset_dir):
            if _tiene_firma(csv_path, firma):
                return csv_path.parent, csv_path.name
    return None


def autodetect() -> dict:
    """Localiza ISIC y PAD-UFES-20 entre los datasets adjuntos.

    Devuelve un diccionario listo para inyectar en la seccion `data` de una
    configuracion. Las claves ausentes indican que ese dataset no esta adjunto.
    """
    resultado: dict = {}

    pad = find_dataset(FIRMA_PAD)
    if pad:
        resultado["pad_root"] = str(pad[0])
        resultado["pad_metadata"] = pad[1]

    isic = find_dataset(FIRMA_ISIC)
    if isic:
        resultado["isic_root"] = str(isic[0])
        resultado["isic_metadata"] = isic[1]

    return resultado


def report() -> str:
    """Resumen de lo que hay montado. Primera celda a ejecutar en el notebook."""
    if not is_kaggle():
        return "No se esta ejecutando en Kaggle (/kaggle/input no existe)."

    lineas = ["Datasets adjuntos:"]
    for d in sorted(KAGGLE_INPUT.iterdir()):
        if d.is_dir():
            n_img = sum(1 for _ in d.rglob("*.jpg")) + sum(1 for _ in d.rglob("*.png"))
            lineas.append(f"  {d.name:45} {n_img:7d} imagenes")

    det = autodetect()
    lineas.append("")
    lineas.append("Deteccion automatica:")
    for clave in ("isic_root", "pad_root"):
        valor = det.get(clave)
        lineas.append(f"  {clave:12} {valor if valor else 'NO ENCONTRADO'}")

    if "pad_root" not in det:
        lineas.append("")
        lineas.append("  AVISO: sin PAD-UFES-20 no hay dominio movil y el trabajo")
        lineas.append("  no se puede evaluar. Adjunta el dataset antes de entrenar.")

    return "\n".join(lineas)
