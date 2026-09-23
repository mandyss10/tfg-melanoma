"""Division de datos en train / val / test.

Todo el modulo existe por una razon: **no puede haber un mismo paciente en dos
particiones**. PAD-UFES-20 tiene 2.298 lesiones de 1.373 pacientes e ISIC 2020
unas 33.000 imagenes de ~2.000 pacientes, de modo que dividir por imagen mete
al mismo paciente en entrenamiento y en test.

El efecto es una sobreestimacion del rendimiento: el modelo puede identificar la
piel, el vello o la iluminacion caracteristicos de ese paciente en lugar de la
morfologia de la lesion. Es un error clasico en trabajos de imagen medica y lo
primero que se pregunta en una defensa.

`stratified_group_split` divide por grupo (paciente) intentando ademas que la
proporcion de casos malignos sea parecida en ambos lados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_group_split(
    frame: pd.DataFrame,
    test_size: float,
    seed: int = 42,
    group_col: str = "patient_id",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide manteniendo cada grupo entero en un solo lado.

    Estrategia: se separan los grupos que contienen algun caso maligno de los
    que no, y se reparte cada conjunto de forma independiente respetando
    `test_size`. Asi la prevalencia queda parecida en ambos lados sin romper
    nunca un grupo.

    Devuelve (train, test).
    """
    if group_col not in frame.columns:
        raise ValueError(
            f"Falta la columna {group_col!r}: sin ella no se puede garantizar "
            "que no haya fuga entre particiones"
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size debe estar en (0, 1), recibido {test_size}")

    rng = np.random.default_rng(seed)

    # Un grupo se considera "positivo" si contiene al menos un caso maligno.
    por_grupo = frame.groupby(group_col)[label_col].max()
    grupos_pos = por_grupo[por_grupo == 1].index.to_numpy()
    grupos_neg = por_grupo[por_grupo == 0].index.to_numpy()

    test_groups = []
    for grupos in (grupos_pos, grupos_neg):
        if len(grupos) == 0:
            continue
        mezclados = rng.permutation(grupos)
        # Al menos 1 grupo a test si existe alguno de ese estrato.
        n_test = max(1, int(round(len(mezclados) * test_size)))
        n_test = min(n_test, len(mezclados) - 1) if len(mezclados) > 1 else len(mezclados)
        test_groups.extend(mezclados[:n_test])

    mascara_test = frame[group_col].isin(set(test_groups))
    train = frame[~mascara_test].reset_index(drop=True)
    test = frame[mascara_test].reset_index(drop=True)

    assert_no_leakage(train, test, group_col=group_col)
    return train, test


def assert_no_leakage(*frames: pd.DataFrame, group_col: str = "patient_id") -> None:
    """Falla si algun grupo aparece en mas de una particion.

    Se llama despues de cada division. Es barato y convierte un error silencioso
    (metricas infladas que solo se detectan al defender) en un fallo ruidoso.
    """
    vistos: dict[str, int] = {}
    for i, f in enumerate(frames):
        if group_col not in f.columns:
            continue
        for g in f[group_col].unique():
            if g in vistos and vistos[g] != i:
                raise AssertionError(
                    f"FUGA DE DATOS: el grupo {g!r} aparece en las particiones "
                    f"{vistos[g]} y {i}"
                )
            vistos[g] = i


def split_report(frames: dict[str, pd.DataFrame], group_col: str = "patient_id") -> str:
    """Resumen legible de las particiones, para la memoria."""
    lineas = [
        f"{'particion':10} {'imagenes':>9} {'pacientes':>10} {'malignos':>9} {'prevalencia':>12}"
    ]
    for nombre, f in frames.items():
        n_pac = f[group_col].nunique() if group_col in f.columns else 0
        n_mal = int((f["label"] == 1).sum())
        prev = n_mal / len(f) if len(f) else 0.0
        lineas.append(
            f"{nombre:10} {len(f):9d} {n_pac:10d} {n_mal:9d} {prev:11.1%}"
        )
    return "\n".join(lineas)
