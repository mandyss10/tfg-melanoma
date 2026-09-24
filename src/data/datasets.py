"""Carga de datasets de lesiones cutaneas.

ISIC y PAD-UFES-20 tienen esquemas de etiquetado distintos. Este modulo los
normaliza a un unico formato tabular con tres columnas:

    path    ruta absoluta a la imagen
    label   0 = benigno, 1 = maligno
    domain  'dermoscopy' o 'mobile'

Trabajar siempre contra ese DataFrame permite que el resto del codigo
(entrenamiento, evaluacion, splits) sea agnostico del origen de los datos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# PAD-UFES-20 etiqueta con el diagnostico clinico abreviado.
#   BCC  carcinoma basocelular      maligno
#   SCC  carcinoma epidermoide      maligno
#   MEL  melanoma                   maligno
#   ACK  queratosis actinica        benigno (premaligno, ver nota)
#   NEV  nevus                      benigno
#   SEK  queratosis seborreica      benigno
#
# Nota: ACK es una lesion premaligna. Se clasifica como benigna porque no
# requiere escision urgente, pero conviene declararlo en la memoria: mover ACK
# al grupo maligno cambia la prevalencia y por tanto las metricas.
PAD_MALIGNANT = {"BCC", "SCC", "MEL"}
PAD_BENIGN = {"ACK", "NEV", "SEK"}

# ISIC usa otro vocabulario (y cambia entre ediciones: 2019 usa codigos, 2020
# nombres completos). Se aplica la MISMA definicion de maligno que en PAD-UFES-20
# (MEL + BCC + SCC); si no, el modelo aprenderia una tarea y se evaluaria en otra.
# La columna `target` de ISIC no sirve para esto: solo marca melanoma.
ISIC_MALIGNANT = {
    "MEL", "MELANOMA",
    "BCC", "BASAL CELL CARCINOMA",
    "SCC", "SQUAMOUS CELL CARCINOMA",
}
ISIC_BENIGN = {
    "NV", "NEVUS",
    "AK", "ACTINIC KERATOSIS",            # premaligna, como ACK en PAD-UFES-20
    "BKL", "SEBORRHEIC KERATOSIS", "PIGMENTED BENIGN KERATOSIS",
    "LICHENOID KERATOSIS", "SOLAR LENTIGO", "LENTIGO NOS",
    "DF", "DERMATOFIBROMA",
    "VASC", "VASCULAR LESION",
    "CAFE-AU-LAIT MACULE", "ATYPICAL MELANOCYTIC PROLIFERATION",
}
# Sin diagnostico concreto: se recurre a la columna `target`. En ISIC 2020 son
# ~27.000 imagenes benignas sin subtipo.
ISIC_SIN_DIAGNOSTICO = {"UNKNOWN", "UNK", "NAN", ""}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DomainStats:
    """Resumen de un conjunto, para volcarlo en la memoria sin recalcularlo."""

    n: int
    n_benign: int
    n_malignant: int
    domain: str

    @property
    def prevalence(self) -> float:
        return self.n_malignant / self.n if self.n else 0.0

    def __str__(self) -> str:
        return (
            f"{self.domain:11} n={self.n:6d}  "
            f"benigno={self.n_benign:6d}  maligno={self.n_malignant:6d}  "
            f"prevalencia={self.prevalence:.1%}"
        )


def _check_dir(root: Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Consulta data/README.md para las instrucciones "
            "de descarga."
        )
    return root


def load_pad_ufes(root: str | Path, metadata_name: str = "metadata.csv") -> pd.DataFrame:
    """Carga PAD-UFES-20 (fotografia de movil).

    Espera la estructura oficial del dataset: un `metadata.csv` con las columnas
    `img_id` y `diagnostic`, y las imagenes en el mismo arbol.
    """
    root = _check_dir(root)
    meta_path = root / metadata_name
    if not meta_path.exists():
        raise FileNotFoundError(f"No se encuentra {meta_path}")

    df = pd.read_csv(meta_path)
    for col in ("img_id", "diagnostic"):
        if col not in df.columns:
            raise ValueError(
                f"{meta_path} no tiene la columna {col!r}. Columnas: {list(df.columns)}"
            )

    # Las imagenes pueden estar repartidas en subcarpetas (imgs_part_1, ...).
    index = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}

    diag = df["diagnostic"].str.upper().str.strip()
    desconocidas = set(diag) - PAD_MALIGNANT - PAD_BENIGN
    if desconocidas:
        raise ValueError(f"Diagnosticos no contemplados en PAD-UFES-20: {desconocidas}")

    # patient_id es imprescindible: PAD-UFES-20 tiene 2.298 lesiones de solo
    # 1.373 pacientes, asi que dividir por imagen mete al mismo paciente en
    # train y en test. El modelo aprenderia a reconocer su piel, no la lesion.
    if "patient_id" in df.columns:
        patient = df["patient_id"].astype(str)
    else:
        print("[PAD-UFES-20] aviso: sin columna patient_id; se usa img_id como grupo")
        patient = df["img_id"].astype(str)

    out = pd.DataFrame(
        {
            "path": df["img_id"].map(lambda n: index.get(n)),
            "label": diag.isin(PAD_MALIGNANT).astype(int),
            "domain": "mobile",
            "diagnostic": diag,
            "patient_id": patient,
        }
    )

    faltan = out["path"].isna().sum()
    if faltan:
        print(f"[PAD-UFES-20] aviso: {faltan} imagenes del CSV no estan en disco; se descartan")
        out = out.dropna(subset=["path"])

    return out.reset_index(drop=True)


def load_isic(
    root: str | Path,
    metadata_name: str = "metadata.csv",
    label_column: str = "target",
    image_column: str = "image_name",
    patient_column: str = "patient_id",
    diagnosis_column: str = "diagnosis",
) -> pd.DataFrame:
    """Carga un volcado de ISIC (dermatoscopia).

    ISIC ha cambiado de esquema entre ediciones, asi que los nombres de columna
    son parametros. Para ISIC 2020 los valores por defecto ya son correctos.
    """
    root = _check_dir(root)
    meta_path = root / metadata_name
    if not meta_path.exists():
        raise FileNotFoundError(f"No se encuentra {meta_path}")

    df = pd.read_csv(meta_path)
    for col in (image_column, label_column):
        if col not in df.columns:
            raise ValueError(
                f"{meta_path} no tiene la columna {col!r}. Columnas: {list(df.columns)}"
            )

    index = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}

    if diagnosis_column in df.columns:
        diag = df[diagnosis_column].astype(str).str.upper().str.strip()
        desconocidas = set(diag) - ISIC_MALIGNANT - ISIC_BENIGN - ISIC_SIN_DIAGNOSTICO
        if desconocidas:
            raise ValueError(
                f"Diagnosticos no contemplados en ISIC: {desconocidas}. "
                "Anadelos a ISIC_MALIGNANT o ISIC_BENIGN en src/data/datasets.py"
            )
        sin_diag = diag.isin(ISIC_SIN_DIAGNOSTICO)
        label = diag.isin(ISIC_MALIGNANT).astype(int)
        label[sin_diag] = df.loc[sin_diag, label_column].astype(int)
    else:
        print(f"[ISIC] aviso: sin columna {diagnosis_column!r}; la etiqueta es "
              f"{label_column!r}, que en ISIC suele marcar SOLO melanoma")
        diag = pd.Series("DESCONOCIDO", index=df.index)
        label = df[label_column].astype(int)

    # ISIC 2020 tambien tiene varias imagenes por paciente (~33.000 imagenes de
    # ~2.000 pacientes), asi que el grupo importa igual que en PAD-UFES-20.
    # ISIC 2019 casi no trae patient_id: los huecos se sustituyen por la propia
    # imagen, porque si no todos acabarian en un unico grupo "nan".
    imagen = df[image_column].astype(str)
    if patient_column in df.columns:
        patient = df[patient_column].astype(str).str.strip()
        vacio = df[patient_column].isna() | patient.isin({"", "nan", "-1", "None"})
        patient = patient.where(~vacio, "img_" + imagen)
        if vacio.any():
            print(f"[ISIC] aviso: {vacio.sum()} imagenes sin {patient_column!r}; "
                  "se usa la imagen como grupo")
    else:
        print(f"[ISIC] aviso: sin columna {patient_column!r}; se usa la imagen como grupo")
        patient = "img_" + imagen

    out = pd.DataFrame(
        {
            "path": df[image_column].map(lambda n: index.get(Path(str(n)).stem)),
            "label": label.values,
            "domain": "dermoscopy",
            "diagnostic": diag.values,
            "patient_id": patient.values,
        }
    )

    faltan = out["path"].isna().sum()
    if faltan:
        print(f"[ISIC] aviso: {faltan} imagenes del CSV no estan en disco; se descartan")
        out = out.dropna(subset=["path"])

    return out.reset_index(drop=True)


def describe(df: pd.DataFrame) -> DomainStats:
    return DomainStats(
        n=len(df),
        n_benign=int((df["label"] == 0).sum()),
        n_malignant=int((df["label"] == 1).sum()),
        domain="/".join(sorted(df["domain"].unique())),
    )


class SkinLesionDataset(Dataset):
    """Dataset de PyTorch sobre el DataFrame normalizado."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        faltan = {"path", "label"} - set(frame.columns)
        if faltan:
            raise ValueError(f"El DataFrame no tiene las columnas {faltan}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]

        # cv2 lee en BGR; albumentations y los backbones esperan RGB.
        image = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"No se pudo leer la imagen {row['path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, np.int64(row["label"])
