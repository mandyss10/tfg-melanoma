# Detección de melanoma en fotografías de móvil

TFG — clasificación de lesiones cutáneas (benigno / maligno) robusta al
cambio de dominio entre imágenes de **dermatoscopio** y **fotografías de
teléfono móvil**.

> **Aviso:** este es un proyecto académico de investigación. No es un producto
> sanitario ni una herramienta de diagnóstico. No debe usarse para tomar
> decisiones médicas.

## El problema

Los modelos entrenados con ISIC (dermatoscopio: iluminación controlada, mucho
zoom, sin reflejos, fondo uniforme) pierden rendimiento de forma acusada al
evaluarlos sobre fotografías tomadas con un móvil (iluminación variable, pelo,
reflejos especulares, ángulo, desenfoque, fondo de piel irregular).

Ese **domain gap** es el objeto del trabajo: no se propone una arquitectura
nueva, sino una estrategia de adaptación de dominio que permita a un modelo
entrenado con datos clínicos generalizar a fotografía casera.

## Hipótesis

Una combinación de (a) *data augmentation* que simule las degradaciones propias
de la cámara de móvil y (b) *fine-tuning* sobre un conjunto reducido de fotos
reales de móvil, recupera una parte sustancial de la caída de rendimiento
respecto a un modelo entrenado solo con dermatoscopia.

## Diseño experimental

Tres modelos, evaluados **siempre sobre el mismo conjunto de test de móvil**:

| Modelo | Entrenamiento | Propósito |
|---|---|---|
| `baseline` | ISIC, augmentation estándar | Línea base: mide la caída de dominio |
| `aug` | ISIC, augmentation que simula móvil | Aísla el efecto de la augmentation |
| `finetune` | `aug` + fine-tuning con PAD-UFES-20 | Propuesta completa |

La comparación entre las tres filas es la contribución del trabajo.

## Métricas

Se reportan AUC, accuracy, sensibilidad y especificidad. **La métrica principal
es la sensibilidad**: en cribado dermatológico un falso negativo (decir "es
benigno" ante un melanoma) tiene un coste clínico muy superior al de un falso
positivo. El umbral de decisión se calibra para fijar una sensibilidad objetivo
y se reporta la especificidad resultante.

## Datos

Ver `data/README.md` para las instrucciones de descarga. Ningún dataset se
versiona en el repositorio.

| Dataset | Dominio | Uso |
|---|---|---|
| ISIC Archive | Dermatoscopio | Entrenamiento base |
| PAD-UFES-20 | Cámara de móvil | Fine-tuning + test |
| Propias | Cámara de móvil | Test cualitativo en la defensa |

## Estructura

```
src/data/        carga de datasets y transformaciones
src/models/      arquitectura y checkpoints
src/utils/       métricas, semillas, logging
experiments/     configuraciones YAML de cada experimento
results/         métricas, figuras y pesos (no versionado)
app/             demo interactiva
notebooks/       exploración y cuadernos de Colab
docs/            memoria y notas
```

## Entorno

El entrenamiento está pensado para ejecutarse en **Google Colab** (GPU T4).
El desarrollo, la evaluación y la demo funcionan en local sobre CPU.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
