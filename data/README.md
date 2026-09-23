# Datos

Ningun dataset se versiona en el repositorio. Descargalos aqui siguiendo las
instrucciones y quedaran en las rutas que esperan los ficheros de configuracion.

## Estructura esperada

```
data/raw/
├── isic/
│   ├── metadata.csv          columnas: image_name, target
│   └── train/                *.jpg
└── pad-ufes-20/
    ├── metadata.csv          columnas: img_id, diagnostic
    └── imgs_part_1/          *.png
```

## ISIC Archive (dermatoscopia)

Base de entrenamiento. Portal: https://challenge.isic-archive.com/data/

La edicion **ISIC 2020** es la mas comoda para empezar: `train.csv` ya trae la
columna binaria `target`. Renombrala a `metadata.csv` o ajusta `isic_kwargs` en
la configuracion.

Aviso sobre el tamano: el conjunto completo de 2020 son ~33.000 imagenes
(unos 23 GB en resolucion original). Para desarrollar conviene bajar primero un
subconjunto, o usar las versiones redimensionadas a 256x256 que circulan en
Kaggle (~3 GB).

Prevalencia de melanoma en ISIC 2020: **1.8%**. Es un desbalance severo y es la
razon de que el codigo use `pos_weight` y calibre el umbral por sensibilidad.

## PAD-UFES-20 (fotografia de movil)

El dataset clave del trabajo: 2.298 lesiones de 1.373 pacientes, fotografiadas
con **camaras de telefono movil** en atencion primaria en Brasil.

- Articulo: Pacheco et al., *Data in Brief* (2020)
- Descarga: https://data.mendeley.com/datasets/zr7vgbcyr2
- Tamano: ~3 GB

Diagnosticos y mapeo binario que aplica `src/data/datasets.py`:

| Codigo | Diagnostico | Etiqueta |
|---|---|---|
| MEL | Melanoma | maligno |
| BCC | Carcinoma basocelular | maligno |
| SCC | Carcinoma epidermoide | maligno |
| ACK | Queratosis actinica | benigno* |
| NEV | Nevus | benigno |
| SEK | Queratosis seborreica | benigno |

\* ACK es una lesion **premaligna**. Se agrupa como benigna porque no exige
escision urgente, pero es una decision discutible que conviene declarar en la
memoria: moverla al grupo maligno cambia la prevalencia y las metricas. Un
analisis de sensibilidad con ambas agrupaciones es un buen apartado.

## Fotos propias

Para la demostracion en la defensa. Colocalas en `data/external/propias/` con un
`metadata.csv` de la misma forma que PAD-UFES-20.

Solo fotos tuyas o de personas que hayan dado consentimiento explicito. No se
versionan: son datos personales de salud. El `.gitignore` ya excluye la carpeta,
pero conviene comprobarlo antes de cada commit.

## Verificar la descarga

```bash
python -c "from src.data.datasets import load_isic, load_pad_ufes, describe; \
print(describe(load_isic('data/raw/isic'))); \
print(describe(load_pad_ufes('data/raw/pad-ufes-20')))"
```

Debe imprimir el recuento por clase y la prevalencia de cada dominio.
