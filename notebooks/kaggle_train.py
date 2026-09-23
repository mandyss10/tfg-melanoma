"""Script de entrenamiento para Kaggle.

Pegar en un notebook de Kaggle con GPU activada (Settings -> Accelerator -> GPU).
Se mantiene como .py para que git lo versione de forma legible; en Kaggle se
copia por celdas o se ejecuta entero.

ANTES DE EJECUTAR
-----------------
1. Activar GPU:  Settings -> Accelerator -> GPU T4 x2  (o P100)
2. Activar internet: Settings -> Internet -> On  (hace falta para pip y para los
   pesos preentrenados de timm)
3. Adjuntar los datasets: boton "+ Add Input" y buscar
     - ISIC 2020 (dermatoscopia)
     - PAD-UFES-20 (fotografia de movil)
   El slug exacto da igual: el autodetector los localiza por el contenido de su
   CSV, no por el nombre.

CUOTA
-----
Kaggle da 30 h de GPU por semana. Un experimento de 15 epocas sobre ISIC 2020
redimensionado a 256x256 ronda los 40-60 min en P100, asi que los tres
experimentos caben de sobra. Conviene NO dejar el notebook ocioso con la GPU
reservada: consume cuota igual.
"""

# =============================================================================
# CELDA 1 - Dependencias
# =============================================================================
# Kaggle ya trae torch, timm y sklearn. Solo falta albumentations reciente.
#
# !pip install -q "albumentations>=2.0" grad-cam


# =============================================================================
# CELDA 2 - Codigo del proyecto
# =============================================================================
# Dos opciones, segun donde tengas el codigo:
#
# (a) Repositorio en GitHub (recomendado, permite iterar sin resubir nada):
#         !git clone -q https://github.com/<tu-usuario>/<tu-repo>.git /kaggle/working/tfg
#
# (b) Subir la carpeta del proyecto como "Kaggle Dataset" privado y copiarla:
#         !cp -r /kaggle/input/<slug-de-tu-codigo> /kaggle/working/tfg
#
# import sys; sys.path.insert(0, "/kaggle/working/tfg")


# =============================================================================
# CELDA 3 - Comprobar el entorno ANTES de gastar cuota de GPU
# =============================================================================
def celda_comprobacion():
    import torch

    from src.utils.kaggle import report

    print(report())
    print()
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: sin GPU. Activala en Settings -> Accelerator antes de entrenar.")


# =============================================================================
# CELDA 4 - Construir las configuraciones con las rutas detectadas
# =============================================================================
def celda_configs(epochs: int = 15, batch_size: int = 64, image_size: int = 224):
    """Genera los tres YAML con las rutas reales de este notebook."""
    from pathlib import Path

    import yaml

    from src.utils.kaggle import autodetect

    detectado = autodetect()
    if "pad_root" not in detectado:
        raise RuntimeError(
            "PAD-UFES-20 no esta adjunto. Sin dominio movil no hay nada que "
            "evaluar: adjuntalo con '+ Add Input' antes de continuar."
        )
    if "isic_root" not in detectado:
        raise RuntimeError("ISIC no esta adjunto.")

    base_data = {
        **detectado,
        "test_size": 0.3,
        "val_size": 0.3,
    }
    base_train = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "num_workers": 2,
    }
    base_model = {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "dropout": 0.3,
        "image_size": image_size,
    }

    out = Path("/kaggle/working/configs")
    out.mkdir(parents=True, exist_ok=True)

    experimentos = {
        "baseline": {
            "transform": {"name": "baseline"},
            "data": {**base_data, "use_mobile_in_train": False},
            "train": dict(base_train),
        },
        "aug": {
            "transform": {"name": "mobile", "kwargs": {"strength": 1.0}},
            "data": {**base_data, "use_mobile_in_train": False},
            "train": dict(base_train),
        },
        "finetune": {
            "transform": {"name": "mobile", "kwargs": {"strength": 1.0}},
            "data": {**base_data, "use_mobile_in_train": True},
            "train": {
                **base_train,
                "epochs": max(1, epochs // 2),
                "lr": 5e-5,
                "init_from": "/kaggle/working/results/aug/best.pth",
            },
        },
    }

    rutas = {}
    for nombre, extra in experimentos.items():
        cfg = {
            "name": nombre,
            "seed": 42,
            "target_sensitivity": 0.95,
            "output_dir": "/kaggle/working/results",
            "model": dict(base_model),
            **extra,
        }
        p = out / f"{nombre}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        rutas[nombre] = str(p)
        print(f"  {nombre:9} -> {p}")

    return rutas


# =============================================================================
# CELDA 5 - Entrenar
# =============================================================================
# El orden importa: 'finetune' parte de los pesos de 'aug'.
#
# !python -m src.train --config /kaggle/working/configs/baseline.yaml
# !python -m src.train --config /kaggle/working/configs/aug.yaml
# !python -m src.train --config /kaggle/working/configs/finetune.yaml


# =============================================================================
# CELDA 6 - Evaluar y comparar
# =============================================================================
# !python -m src.evaluate --compare \
#     /kaggle/working/results/baseline \
#     /kaggle/working/results/aug \
#     /kaggle/working/results/finetune


# =============================================================================
# CELDA 7 - Descargar resultados antes de cerrar la sesion
# =============================================================================
# Kaggle borra /kaggle/working al cerrar si no se guarda la version del
# notebook. Empaquetar y descargar:
#
# !cd /kaggle/working && tar czf resultados.tar.gz results/ configs/
# Los .pth pesan ~16 MB cada uno, asi que el tar entero ronda los 50 MB.
