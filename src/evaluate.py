"""Evaluacion de un experimento sobre el conjunto de test reservado.

Uso:
    python -m src.evaluate --run results/aug
    python -m src.evaluate --compare results/baseline results/aug results/finetune

El umbral NO se recalibra sobre el test: se hereda el que se fijo en validacion
durante el entrenamiento. Calibrarlo sobre el test seria mirar la respuesta
antes de contestar, y sesga la sensibilidad al alza.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SkinLesionDataset, describe
from src.data.transforms import build_transform
from src.models.classifier import load_checkpoint
from src.utils.metrics import bootstrap_ci, compute_metrics


@torch.no_grad()
def predict(model, frame: pd.DataFrame, device: str, size: int = 224, batch_size: int = 32):
    """Puntuaciones del modelo sobre un DataFrame normalizado."""
    loader = DataLoader(
        SkinLesionDataset(frame, build_transform("baseline", size=size, train=False)),
        batch_size=batch_size,
        shuffle=False,
    )
    scores, targets = [], []
    for x, y in loader:
        scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(scores), np.concatenate(targets)


def evaluate_run(run_dir: str | Path, device: str = "cpu", n_boot: int = 1000) -> dict:
    """Evalua un experimento y deja los resultados junto a sus pesos."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.pth"
    test_path = run_dir / "test_split.csv"

    for p in (ckpt_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta {p}. Ejecuta primero src.train")

    model, meta = load_checkpoint(ckpt_path, device=device)
    test = pd.read_csv(test_path)
    size = meta.get("config", {}).get("model", {}).get("image_size", 224)

    # Umbral heredado de validacion: no se recalibra sobre el test.
    threshold = meta.get("threshold", 0.5)

    scores, targets = predict(model, test, device, size=size)
    metrics = compute_metrics(targets, scores, threshold)
    auc_lo, auc_hi = bootstrap_ci(targets, scores, "auc", n_boot=n_boot, threshold=threshold)
    sens_lo, sens_hi = bootstrap_ci(
        targets, scores, "sensitivity", n_boot=n_boot, threshold=threshold
    )

    resultado = {
        "run": run_dir.name,
        "epoch": meta.get("epoch"),
        "val_auc": meta.get("val_auc"),
        "threshold": threshold,
        **metrics.as_dict(),
        "auc_ci95": [auc_lo, auc_hi],
        "sensitivity_ci95": [sens_lo, sens_hi],
        "test_domain": describe(test).domain,
    }

    (run_dir / "test_metrics.json").write_text(
        json.dumps(resultado, indent=2), encoding="utf-8"
    )
    np.save(run_dir / "test_scores.npy", scores)

    return resultado


def print_report(res: dict) -> None:
    print(f"\n=== {res['run']} (test: {res['test_domain']}, n={res['n']}) ===")
    print(f"  AUC          {res['auc']:.4f}  IC95% [{res['auc_ci95'][0]:.4f}, {res['auc_ci95'][1]:.4f}]")
    print(f"  Sensibilidad {res['sensitivity']:.4f}  IC95% [{res['sensitivity_ci95'][0]:.4f}, {res['sensitivity_ci95'][1]:.4f}]")
    print(f"  Especificidad{res['specificity']:.4f}")
    print(f"  Accuracy     {res['accuracy']:.4f}   (umbral {res['threshold']:.4f}, heredado de validacion)")
    print(f"  Matriz       TP={res['tp']} FP={res['fp']} TN={res['tn']} FN={res['fn']}")
    if res["fn"]:
        print(f"  -> {res['fn']} lesiones malignas clasificadas como benignas")


def compare(run_dirs: list[str], device: str = "cpu", n_boot: int = 1000) -> pd.DataFrame:
    """Tabla comparativa: es la tabla de resultados de la memoria."""
    filas = []
    for d in run_dirs:
        res = evaluate_run(d, device=device, n_boot=n_boot)
        print_report(res)
        filas.append(
            {
                "experimento": res["run"],
                "AUC": res["auc"],
                "AUC_lo": res["auc_ci95"][0],
                "AUC_hi": res["auc_ci95"][1],
                "sensibilidad": res["sensitivity"],
                "especificidad": res["specificity"],
                "accuracy": res["accuracy"],
                "FN": res["fn"],
                "n": res["n"],
            }
        )

    df = pd.DataFrame(filas)

    print("\n" + "=" * 72)
    print("TABLA COMPARATIVA (test: fotografia de movil, identico en las tres filas)")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(df) > 1:
        base = df.iloc[0]
        mejor = df.iloc[df["AUC"].idxmax()]
        delta = mejor["AUC"] - base["AUC"]
        solapan = mejor["AUC_lo"] <= base["AUC_hi"]
        print(f"\nMejora de AUC sobre '{base['experimento']}': {delta:+.4f}")
        if solapan:
            print("AVISO: los intervalos de confianza se solapan. Con este tamano de")
            print("test la diferencia NO es estadisticamente concluyente; conviene")
            print("repetir con varias semillas antes de afirmar una mejora.")

    out = Path("results/metrics")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    print(f"\nTabla guardada en {out / 'comparison.csv'}")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="directorio de un experimento")
    ap.add_argument("--compare", nargs="+", help="varios directorios a comparar")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, device=args.device, n_boot=args.n_boot)
    elif args.run:
        print_report(evaluate_run(args.run, device=args.device, n_boot=args.n_boot))
    else:
        ap.error("indica --run o --compare")


if __name__ == "__main__":
    main()
