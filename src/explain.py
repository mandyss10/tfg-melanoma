"""Explicabilidad con Grad-CAM.

Genera mapas de calor que muestran en que region de la imagen se apoyo el modelo
para decidir. Cumple dos funciones en el trabajo:

1. **Rigor.** Permite comprobar que el modelo mira la lesion y no un artefacto.
   Es un fallo documentado en dermatologia computacional: modelos que aprenden a
   detectar las reglas milimetradas o los parches adhesivos que aparecen en las
   fotos clinicas de casos malignos, en vez de la morfologia de la lesion. Si el
   mapa de calor se enciende en una esquina, hay un problema aunque el AUC sea
   alto.

2. **Defensa.** Visualmente es lo mas convincente que se puede ensenar: "el
   modelo se fija en el borde irregular, que es lo que mira un dermatologo".

Uso:
    python -m src.explain --run results/aug --n 8
    python -m src.explain --run results/aug --only-errors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # backend sin ventana: necesario en Kaggle y en servidores
import matplotlib.pyplot as plt  # noqa: E402

from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget  # noqa: E402

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform  # noqa: E402
from src.models.classifier import load_checkpoint  # noqa: E402


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Deshace la normalizacion de ImageNet para poder mostrar la imagen."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def overlay_cam(imagen: np.ndarray, mapa: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone el mapa de calor sobre la imagen original."""
    heat = cv2.applyColorMap(np.uint8(255 * mapa), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mezcla = (1 - alpha) * imagen + alpha * heat
    return np.clip(mezcla, 0, 1)


def explain_run(
    run_dir: str | Path,
    n: int = 8,
    only_errors: bool = False,
    device: str = "cpu",
    out_name: str | None = None,
) -> Path:
    """Genera una figura con los mapas de calor de varias imagenes del test.

    Args:
        only_errors: si es True, selecciona solo casos mal clasificados. Los
            errores son mucho mas informativos que los aciertos: ensenan si el
            modelo falla por mirar donde no debe.
    """
    run_dir = Path(run_dir)
    model, meta = load_checkpoint(run_dir / "best.pth", device=device)
    test = pd.read_csv(run_dir / "test_split.csv")

    size = meta.get("config", {}).get("model", {}).get("image_size", 224)
    threshold = meta.get("threshold", 0.5)
    tf = build_transform("baseline", size=size, train=False)

    scores_path = run_dir / "test_scores.npy"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Falta {scores_path}. Ejecuta antes: python -m src.evaluate --run {run_dir}"
        )
    scores = np.load(scores_path)
    test = test.assign(score=scores, pred=(scores >= threshold).astype(int))

    if only_errors:
        seleccion = test[test["pred"] != test["label"]]
        if seleccion.empty:
            print("No hay errores en el conjunto de test.")
            seleccion = test
        titulo = "Errores de clasificacion"
    else:
        # Mezcla equilibrada de ambas clases, de mayor a menor confianza.
        malignos = test[test["label"] == 1].nlargest(n // 2, "score")
        benignos = test[test["label"] == 0].nsmallest(n - n // 2, "score")
        seleccion = pd.concat([malignos, benignos])
        titulo = "Casos representativos"

    seleccion = seleccion.head(n)
    if seleccion.empty:
        raise ValueError("No hay imagenes que explicar")

    cam = GradCAM(model=model, target_layers=[model.target_layer_for_cam])

    filas = 2
    cols = int(np.ceil(len(seleccion) / filas))
    fig, axes = plt.subplots(filas, cols, figsize=(3.2 * cols, 6.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, seleccion.iterrows()):
        bgr = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = tf(image=rgb)["image"].unsqueeze(0).to(device)

        mapa = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
        vista = overlay_cam(denormalize(tensor[0]), mapa)

        real = "maligno" if row["label"] == 1 else "benigno"
        pred = "maligno" if row["pred"] == 1 else "benigno"
        acierto = row["label"] == row["pred"]

        ax.imshow(vista)
        ax.set_title(
            f"real: {real}\npred: {pred} ({row['score']:.2f})",
            fontsize=9,
            color="green" if acierto else "red",
        )
        ax.axis("off")

    for ax in axes[len(seleccion):]:
        ax.axis("off")

    fig.suptitle(f"Grad-CAM - {run_dir.name} - {titulo}", fontsize=13)
    fig.tight_layout()

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    nombre = out_name or f"gradcam_{run_dir.name}{'_errores' if only_errors else ''}.png"
    destino = out_dir / nombre
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada en {destino}")
    return destino


def explain_image(
    image_path: str | Path, checkpoint: str | Path, device: str = "cpu"
) -> tuple[float, np.ndarray]:
    """Explica una imagen suelta. Lo usa la app de demo."""
    model, meta = load_checkpoint(checkpoint, device=device)
    size = meta.get("config", {}).get("model", {}).get("image_size", 224)
    tf = build_transform("baseline", size=size, train=False)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"No se pudo leer {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = tf(image=rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor))[0])

    cam = GradCAM(model=model, target_layers=[model.target_layer_for_cam])
    mapa = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
    return prob, overlay_cam(denormalize(tensor[0]), mapa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--only-errors", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    explain_run(args.run, n=args.n, only_errors=args.only_errors, device=args.device)


if __name__ == "__main__":
    main()
