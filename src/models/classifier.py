"""Clasificador binario de lesiones cutaneas.

Se usa un backbone estandar preentrenado en ImageNet via `timm`. El valor del
trabajo esta en la estrategia de datos y adaptacion de dominio, no en la
arquitectura, asi que conviene que esta sea una eleccion conservadora y
reproducible.

La cabeza tiene UNA salida (logit) en lugar de dos: para clasificacion binaria
`BCEWithLogitsLoss` permite ponderar la clase positiva con `pos_weight`, que es
la forma limpia de tratar el desbalance de ISIC.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn


class SkinLesionClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # num_classes=0 -> timm devuelve el vector de features tras el pooling.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits de forma (N,). Aplicar sigmoid para probabilidades."""
        return self.head(self.backbone(x)).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

    def freeze_backbone(self) -> None:
        """Congela el extractor: util en la primera fase del fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def target_layer_for_cam(self) -> nn.Module:
        """Ultima capa convolucional, que es donde Grad-CAM engancha."""
        convs = [m for m in self.backbone.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise RuntimeError(f"No se hallaron capas Conv2d en {self.backbone_name}")
        return convs[-1]


def pos_weight_from_labels(labels) -> torch.Tensor:
    """Peso de la clase positiva para compensar el desbalance.

    pos_weight = n_negativos / n_positivos. Con la prevalencia de ISIC (~2%)
    esto vale ~50, lo que evita que el modelo colapse a predecir siempre
    'benigno'.
    """
    import numpy as np

    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("No hay ejemplos positivos")
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def save_checkpoint(model: SkinLesionClassifier, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "backbone": model.backbone_name, **extra},
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[SkinLesionClassifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SkinLesionClassifier(backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
