"""Figuras de analisis de imagen para la memoria.

Genera en results/figures/:

    dominios.png        ejemplos de ISIC (dermatoscopio) frente a PAD-UFES-20
                        (movil), benignos y malignos: el cambio de dominio a ojo.
    estadisticas.png    el mismo cambio de dominio medido: brillo, saturacion,
                        contraste y nitidez de cada dataset (+ estadisticas.csv).
    augmentation.png    una imagen de dermatoscopio y varias versiones con las
                        degradaciones de movil simuladas del experimento 'aug'.
    preprocesado.png    paso a paso de la eliminacion de pelo y la constancia de
                        color sobre las imagenes con mas vello de la muestra.
    curvas.png          (con --runs) evolucion de la AUC y la perdida por epoca.

Uso en Kaggle (datasets adjuntos, deteccion automatica):
    python -m src.visualize --runs /kaggle/working/results/baseline ...

En local:
    python -m src.visualize --isic-root data/raw/isic --pad-root data/raw/pad-ufes-20
    python -m src.visualize --solo-curvas --runs results/baseline results/aug ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.datasets import load_isic, load_pad_ufes  # noqa: E402
from src.data.preprocess import limitar_tamano, mascara_pelo, pasos  # noqa: E402
from src.data.transforms import build_transform  # noqa: E402
from src.explain import denormalize  # noqa: E402

NOMBRE_DOMINIO = {"dermoscopy": "ISIC (dermatoscopio)", "mobile": "PAD-UFES-20 (movil)"}
COLOR_DOMINIO = {"dermoscopy": "#1f77b4", "mobile": "#d62728"}


def leer(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"No se pudo leer {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cuadrada(rgb: np.ndarray, lado: int = 256) -> np.ndarray:
    """Recorte central cuadrado y redimensionado, para mosaicos homogeneos."""
    h, w = rgb.shape[:2]
    m = min(h, w)
    y0, x0 = (h - m) // 2, (w - m) // 2
    return cv2.resize(rgb[y0 : y0 + m, x0 : x0 + m], (lado, lado), interpolation=cv2.INTER_AREA)


def guardar(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    return out


# --- 1. Ejemplos de cada dominio ---------------------------------------------


def fig_dominios(datos: pd.DataFrame, out: Path, por_fila: int = 6, seed: int = 42) -> Path:
    filas = [
        (dom, lab)
        for dom in ("dermoscopy", "mobile")
        for lab in (0, 1)
    ]
    fig, axes = plt.subplots(len(filas), por_fila, figsize=(2.2 * por_fila, 2.5 * len(filas)))
    for i, (dom, lab) in enumerate(filas):
        sub = datos[(datos.domain == dom) & (datos.label == lab)]
        muestra = sub.sample(min(por_fila, len(sub)), random_state=seed)
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j < len(muestra):
                fila = muestra.iloc[j]
                ax.imshow(cuadrada(leer(fila.path)))
                ax.set_title(fila.diagnostic, fontsize=9)
        axes[i][0].text(
            -0.08, 0.5,
            f"{NOMBRE_DOMINIO[dom]}\n{'maligno' if lab else 'benigno'}",
            transform=axes[i][0].transAxes, ha="right", va="center", fontsize=10,
        )
    fig.suptitle("Cambio de dominio: dermatoscopio frente a fotografia de movil", fontsize=13)
    return guardar(fig, out)


# --- 2. Estadisticas de imagen por dominio -----------------------------------


def medidas(rgb: np.ndarray) -> dict[str, float]:
    """Descriptores globales que capturan las diferencias fisicas entre dominios.

    Se calculan a una resolucion comun (256x256) porque la nitidez depende del
    tamano: sin normalizar, las fotos grandes de PAD parecerian mas nitidas.
    """
    img = cuadrada(rgb, 256)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return {
        "brillo": float(hsv[..., 2].mean() / 255),
        "saturacion": float(hsv[..., 1].mean() / 255),
        "contraste": float(lab[..., 0].std() / 255),
        "nitidez (log var. Laplaciano)": float(np.log1p(cv2.Laplacian(gris, cv2.CV_64F).var())),
        "fraccion de pelo": float((mascara_pelo(limitar_tamano(rgb)) > 0).mean()),
    }


def fig_estadisticas(datos: pd.DataFrame, out: Path, n: int = 300, seed: int = 42) -> Path:
    filas = []
    for dom, sub in datos.groupby("domain"):
        for path in sub.sample(min(n, len(sub)), random_state=seed).path:
            filas.append({"domain": dom, **medidas(leer(path))})
    tabla = pd.DataFrame(filas)

    resumen = tabla.groupby("domain").agg(["mean", "std"]).round(4)
    resumen.to_csv(out.with_suffix(".csv"))
    print(resumen.T.to_string())

    variables = [c for c in tabla.columns if c != "domain"]
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.4))
    for ax, var in zip(axes, variables):
        for dom, sub in tabla.groupby("domain"):
            ax.hist(sub[var], bins=30, alpha=0.55, density=True,
                    color=COLOR_DOMINIO[dom], label=NOMBRE_DOMINIO[dom])
        ax.set_title(var, fontsize=10)
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Distribucion de propiedades de imagen por dominio (n={n} por dominio)")
    return guardar(fig, out)


# --- 3. Augmentation que simula el movil -------------------------------------


def fig_augmentation(datos: pd.DataFrame, out: Path, n: int = 7, seed: int = 42) -> Path:
    ejemplo = datos[(datos.domain == "dermoscopy") & (datos.label == 1)].sample(1, random_state=seed)
    rgb = leer(ejemplo.path.iloc[0])
    tf = build_transform("mobile", size=256, train=True, strength=1.0)

    fig, axes = plt.subplots(2, (n + 2) // 2, figsize=(2.6 * ((n + 2) // 2), 5.6))
    axes = axes.ravel()
    axes[0].imshow(cuadrada(rgb))
    axes[0].set_title("original", fontsize=10)
    np.random.seed(seed)
    for ax in axes[1:]:
        ax.imshow(denormalize(tf(image=rgb)["image"]))
        ax.set_title("simulada", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Degradaciones de movil simuladas sobre una imagen de dermatoscopio (experimento 'aug')")
    return guardar(fig, out)


# --- 4. Preprocesado paso a paso ---------------------------------------------


def fig_preprocesado(datos: pd.DataFrame, out: Path, n: int = 4, candidatas: int = 80, seed: int = 42) -> Path:
    """Muestra el preprocesado sobre las imagenes con MAS pelo de una muestra.

    Elegir al azar daria casi siempre imagenes sin vello, donde el paso no hace
    nada visible.
    """
    muestra = datos.sample(min(candidatas, len(datos)), random_state=seed).copy()
    muestra["pelo"] = [(mascara_pelo(limitar_tamano(leer(p))) > 0).mean() for p in muestra.path]
    elegidas = muestra.nlargest(n, "pelo")

    columnas = list(pasos(leer(elegidas.path.iloc[0])))
    fig, axes = plt.subplots(n, len(columnas), figsize=(3 * len(columnas), 3 * n))
    for i, (_, fila) in enumerate(elegidas.iterrows()):
        for j, (nombre, img) in enumerate(pasos(leer(fila.path)).items()):
            ax = axes[i][j]
            ax.imshow(cuadrada(img), cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
            if i == 0:
                ax.set_title(nombre, fontsize=11)
        axes[i][0].text(-0.05, 0.5, NOMBRE_DOMINIO[fila.domain].split(" ")[0],
                        transform=axes[i][0].transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Preprocesado: eliminacion de pelo (DullRazor) y constancia de color (Shades of Gray)")
    return guardar(fig, out)


# --- 5. Curvas de entrenamiento ----------------------------------------------


def fig_curvas(runs: list[str], out: Path) -> Path | None:
    historiales = {}
    for r in runs:
        h = Path(r) / "history.json"
        if h.exists():
            historiales[Path(r).name] = pd.DataFrame(json.loads(h.read_text(encoding="utf-8")))
    if not historiales:
        print("  (sin history.json: se omite curvas.png)")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for nombre, h in historiales.items():
        axes[0].plot(h.epoch, h.auc, marker="o", ms=3, label=nombre)
        axes[1].plot(h.epoch, h.val_loss, marker="o", ms=3, label=nombre)
        axes[2].plot(h.epoch, h.train_loss, marker="o", ms=3, label=nombre)
    for ax, titulo in zip(axes, ["AUC en validacion (movil)", "Perdida en validacion (movil)",
                                 "Perdida en entrenamiento"]):
        ax.set_title(titulo)
        ax.set_xlabel("epoca")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Evolucion del entrenamiento")
    return guardar(fig, out)


def cargar_datos(args) -> pd.DataFrame:
    if args.isic_root and args.pad_root:
        rutas = {"isic_root": args.isic_root, "pad_root": args.pad_root}
        # ISIC 2019/2020 traen train.csv; la estructura local de data/README.md
        # usa metadata.csv.
        for clave in ("isic", "pad"):
            raiz = Path(rutas[f"{clave}_root"])
            if not (raiz / "metadata.csv").exists() and (raiz / "train.csv").exists():
                rutas[f"{clave}_metadata"] = "train.csv"
    else:
        from src.utils.kaggle import autodetect

        rutas = autodetect(args.search_root)
        if "isic_root" not in rutas or "pad_root" not in rutas:
            raise SystemExit(f"No se encontraron los datasets. Detectado: {rutas}")
    isic = load_isic(rutas["isic_root"], metadata_name=rutas.get("isic_metadata", "metadata.csv"))
    pad = load_pad_ufes(rutas["pad_root"], metadata_name=rutas.get("pad_metadata", "metadata.csv"))
    return pd.concat([isic, pad], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--runs", nargs="*", default=[], help="experimentos para curvas.png")
    ap.add_argument("--n", type=int, default=300, help="imagenes por dominio en estadisticas")
    ap.add_argument("--isic-root")
    ap.add_argument("--pad-root")
    ap.add_argument("--search-root", help="donde buscar los datasets (por defecto /kaggle/input)")
    ap.add_argument("--solo-curvas", action="store_true",
                    help="solo curvas.png; no necesita los datasets (util en local)")
    args = ap.parse_args()

    out = Path(args.out)
    if args.solo_curvas:
        fig_curvas(args.runs, out / "curvas.png")
        return

    datos = cargar_datos(args)
    print(f"{len(datos)} imagenes cargadas. Figuras:")

    fig_dominios(datos, out / "dominios.png")
    fig_augmentation(datos, out / "augmentation.png")
    fig_preprocesado(datos, out / "preprocesado.png")
    fig_estadisticas(datos, out / "estadisticas.png", n=args.n)
    fig_curvas(args.runs, out / "curvas.png")


if __name__ == "__main__":
    main()
