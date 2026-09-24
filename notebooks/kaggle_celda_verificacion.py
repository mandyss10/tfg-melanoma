# ==============================================================
# CELDA DE VERIFICACION - ejecutar ANTES de entrenar
# Comprueba que los datos reales se cargan y se dividen bien.
# Tarda ~1 min y no usa GPU.
# ==============================================================
import sys
sys.path.insert(0, "/kaggle/working/tfg")   # ajustar si el codigo esta en otra ruta

from src.utils.kaggle import autodetect
from src.data.datasets import load_isic, load_pad_ufes
from src.data.splits import stratified_group_split, split_report

det = autodetect()
print("Rutas detectadas:")
for k, v in det.items():
    print(f"  {k:14} {v}")
print()

isic = load_isic(det["isic_root"], metadata_name=det["isic_metadata"])
pad = load_pad_ufes(det["pad_root"], metadata_name=det["pad_metadata"])

print(f"ISIC  {len(isic):6d} imagenes  {isic.patient_id.nunique():5d} pacientes  "
      f"prevalencia {isic.label.mean():.2%}")
print(f"PAD   {len(pad):6d} imagenes  {pad.patient_id.nunique():5d} pacientes  "
      f"prevalencia {pad.label.mean():.2%}")
print()

print("Diagnosticos en ISIC (maligno = MEL, BCC, SCC, igual que en PAD):")
print(isic.groupby(["diagnostic", "label"]).size().to_string())
print()

print("Diagnosticos en PAD-UFES-20:")
print(pad.groupby(["diagnostic", "label"]).size().to_string())
print()

trval, test = stratified_group_split(pad, test_size=0.3, seed=42)
train, val = stratified_group_split(trval, test_size=0.3, seed=42)
print("Particiones del dominio movil:")
print(split_report({"train": train, "val": val, "test": test}))
print()

solapes = (
    len(set(train.patient_id) & set(test.patient_id))
    + len(set(val.patient_id) & set(test.patient_id))
    + len(set(train.patient_id) & set(val.patient_id))
)
print(f"pacientes compartidos entre particiones: {solapes}  (debe ser 0)")
