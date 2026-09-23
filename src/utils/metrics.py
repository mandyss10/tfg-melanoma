"""Metricas de clasificacion orientadas a cribado dermatologico.

En este problema el umbral de decision NO es 0.5. Un falso negativo (declarar
benigna una lesion maligna) retrasa el diagnostico de un melanoma; un falso
positivo solo provoca una consulta innecesaria. El coste es asimetrico, asi que
el protocolo es:

    1. fijar una sensibilidad objetivo (p.ej. 0.95)
    2. elegir el umbral mas alto que la alcanza SOBRE EL CONJUNTO DE VALIDACION
    3. reportar la especificidad resultante sobre el conjunto de TEST

Reportar accuracy con umbral 0.5 sobre un conjunto desbalanceado es enganoso:
un modelo que prediga siempre "benigno" sobre ISIC 2020 (prevalencia ~1.8%)
alcanza un 98% de accuracy y una sensibilidad de 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    auc: float
    average_precision: float
    accuracy: float
    sensitivity: float   # recall de la clase maligna (TPR)
    specificity: float   # TNR
    precision: float
    threshold: float
    n: int
    n_positive: int
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AUC={self.auc:.4f}  AP={self.average_precision:.4f}  "
            f"sens={self.sensitivity:.4f}  spec={self.specificity:.4f}  "
            f"acc={self.accuracy:.4f}  (umbral={self.threshold:.4f}, n={self.n})"
        )


def threshold_for_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, target_sensitivity: float = 0.95
) -> float:
    """Umbral mas alto que alcanza al menos `target_sensitivity`.

    Se elige el mas alto de entre los validos porque, a igualdad de
    sensibilidad, maximiza la especificidad.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.sum() == 0:
        raise ValueError("No hay casos positivos: no se puede calibrar sensibilidad")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    validos = thresholds[tpr >= target_sensitivity]
    if len(validos) == 0:
        # Ningun umbral alcanza el objetivo: devolver el mas permisivo posible.
        return float(thresholds.min())
    return float(validos.max())


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    """Metricas completas para un umbral dado.

    `y_score` son probabilidades de la clase maligna, no logits.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(f"Formas distintas: {y_true.shape} vs {y_score.shape}")

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # AUC y AP no estan definidos si solo hay una clase presente.
    una_sola_clase = len(np.unique(y_true)) < 2
    auc = float("nan") if una_sola_clase else roc_auc_score(y_true, y_score)
    ap = float("nan") if una_sola_clase else average_precision_score(y_true, y_score)

    return ClassificationMetrics(
        auc=float(auc),
        average_precision=float(ap),
        accuracy=float((tp + tn) / max(len(y_true), 1)),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        specificity=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        precision=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        threshold=float(threshold),
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    threshold: float = 0.5,
    seed: int = 0,
) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap sobre las muestras.

    El tribunal preguntara si la diferencia entre modelos es significativa. Con
    conjuntos de test de unos cientos de imagenes las diferencias de AUC de
    menos de 0.02 suelen caer dentro del ruido: conviene poder ensenarlo.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    valores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # remuestreo degenerado, se descarta
        m = compute_metrics(y_true[idx], y_score[idx], threshold)
        valores.append(getattr(m, metric))

    if not valores:
        return (float("nan"), float("nan"))

    lo = float(np.percentile(valores, 100 * alpha / 2))
    hi = float(np.percentile(valores, 100 * (1 - alpha / 2)))
    return lo, hi
