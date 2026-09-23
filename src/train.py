"""Entrenamiento de un experimento.

Uso:
    python -m src.train --config experiments/configs/baseline.yaml

El fichero de configuracion fija dataset, transformacion y backbone, de modo
que los tres experimentos del trabajo (baseline / aug / finetune) se diferencian
solo en datos, no en codigo. Eso es lo que hace la comparacion defendible: si
cambiase el codigo entre ramas, las diferencias no serian atribuibles a la
estrategia de dominio.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import SkinLesionDataset, describe, load_isic, load_pad_ufes
from src.data.transforms import build_transform
from src.models.classifier import (
    SkinLesionClassifier,
    pos_weight_from_labels,
    save_checkpoint,
)
from src.utils.metrics import compute_metrics, threshold_for_sensitivity


def set_seed(seed: int) -> None:
    """Semilla en las tres fuentes de aleatoriedad, para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_frames(cfg: dict) -> dict[str, pd.DataFrame]:
    """Construye los DataFrames de train / val / test segun la configuracion.

    El conjunto de TEST es siempre de dominio movil: todos los experimentos se
    juzgan sobre el escenario de uso real.
    """
    data = cfg["data"]
    seed = cfg.get("seed", 42)

    frames = []
    if data.get("isic_root"):
        frames.append(load_isic(data["isic_root"], **data.get("isic_kwargs", {})))
    if data.get("pad_root"):
        frames.append(load_pad_ufes(data["pad_root"]))
    if not frames:
        raise ValueError("La configuracion no declara ningun dataset")

    full = pd.concat(frames, ignore_index=True)

    mobile = full[full["domain"] == "mobile"]
    dermo = full[full["domain"] == "dermoscopy"]

    if len(mobile) == 0:
        raise ValueError(
            "No hay datos de dominio movil: sin ellos no se puede evaluar el "
            "objetivo del trabajo. Anade pad_root a la configuracion."
        )

    # El split de movil se hace SIEMPRE con la misma semilla en los tres
    # experimentos, de modo que comparten exactamente el mismo test.
    mob_trainval, mob_test = train_test_split(
        mobile,
        test_size=data.get("test_size", 0.3),
        stratify=mobile["label"],
        random_state=seed,
    )
    mob_train, mob_val = train_test_split(
        mob_trainval,
        test_size=data.get("val_size", 0.3),
        stratify=mob_trainval["label"],
        random_state=seed,
    )

    # Que parte del movil entra en entrenamiento depende del experimento:
    # baseline y aug no ven ninguna foto de movil; finetune si.
    usar_movil_en_train = data.get("use_mobile_in_train", False)

    train = dermo if not usar_movil_en_train else pd.concat([dermo, mob_train])
    if len(train) == 0:
        raise ValueError("El conjunto de entrenamiento esta vacio")

    return {
        "train": train.reset_index(drop=True),
        "val": mob_val.reset_index(drop=True),
        "test": mob_test.reset_index(drop=True),
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, scores, targets = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
            x = x.to(device)
            y = y.to(device).float()

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())

    return (
        total_loss / max(len(loader.dataset), 1),
        np.concatenate(scores),
        np.concatenate(targets),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    set_seed(cfg.get("seed", 42))
    device = args.device

    out_dir = Path(cfg.get("output_dir", "results")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== experimento: {name} | dispositivo: {device} ===")
    frames = build_frames(cfg)
    for split, df in frames.items():
        print(f"  {split:5} {describe(df)}")

    size = cfg["model"].get("image_size", 224)
    tf_train = build_transform(
        cfg["transform"]["name"],
        size=size,
        train=True,
        **cfg["transform"].get("kwargs", {}),
    )
    tf_eval = build_transform("baseline", size=size, train=False)

    n_workers = cfg["train"].get("num_workers", 0)
    loaders = {
        "train": DataLoader(
            SkinLesionDataset(frames["train"], tf_train),
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=n_workers,
            drop_last=True,
        ),
        "val": DataLoader(
            SkinLesionDataset(frames["val"], tf_eval),
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=n_workers,
        ),
    }

    model = SkinLesionClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    init_from = cfg["train"].get("init_from")
    if init_from:
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  pesos iniciales cargados de {init_from}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_from_labels(frames["train"]["label"]).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    objetivo_sens = cfg.get("target_sensitivity", 0.95)
    historial, mejor_auc, t0 = [], -1.0, time.time()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device, True
        )
        va_loss, va_scores, va_y = run_epoch(
            model, loaders["val"], criterion, optimizer, device, False
        )
        scheduler.step()

        th = threshold_for_sensitivity(va_y, va_scores, objetivo_sens)
        m = compute_metrics(va_y, va_scores, th)

        print(
            f"  epoca {epoch:3d}/{cfg['train']['epochs']}  "
            f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  {m}"
        )

        historial.append(
            {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **m.as_dict()}
        )

        # Se selecciona por AUC de validacion, que es independiente del umbral.
        if m.auc > mejor_auc:
            mejor_auc = m.auc
            save_checkpoint(
                model,
                out_dir / "best.pth",
                epoch=epoch,
                val_auc=m.auc,
                threshold=th,
                config=cfg,
            )

    (out_dir / "history.json").write_text(
        json.dumps(historial, indent=2), encoding="utf-8"
    )
    frames["test"].to_csv(out_dir / "test_split.csv", index=False)

    print(f"\nmejor AUC de validacion: {mejor_auc:.4f}  ({time.time() - t0:.0f}s)")
    print(f"pesos en {out_dir / 'best.pth'}")
    print(f"test reservado en {out_dir / 'test_split.csv'} -> evaluar con src.evaluate")


if __name__ == "__main__":
    main()
