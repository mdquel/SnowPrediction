# Máquina del experimento sat_v2 (paso 1)

Todas las ejecuciones de esta carpeta (brazos base, A y B; semillas 7, 42 y 123)
se entrenaron y evaluaron en la misma instancia, en serie.

## Hardware (Vast.ai)

| Dato | Valor | Fuente |
|---|---|---|
| Proveedor | Vast.ai, servidor en Corea del Sur | ficha de la oferta |
| Instancia | 56857030 | panel de Vast.ai |
| GPU | 1 × NVIDIA GeForce RTX 5070 Ti, 16 GB (16303 MiB) | `entorno.txt` |
| Driver de la GPU | 570.181 | `entorno.txt` |
| CUDA máximo del host | 12.8 | ficha de la oferta |
| CPU | Intel Xeon E5-2630 v4; 10 núcleos asignados (el servidor tiene 20 hilos) | ficha de la oferta; `entorno.txt` |
| RAM | 16 GB asignados | ficha de la oferta |
| Disco | SSD NVMe Samsung MZVLQ1T0HBLB, 150 GB asignados | ficha de la oferta |

## Software

| Dato | Valor |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA (PyTorch) | 12.8 |
| cuDNN | 9.19 (91900) |
| Código | commit `4ffea6e`, rama `feat/save-predictions` |

## Datos

- `images`, `masks` y CSV del split: versión 2 del dataset de Kaggle
  `martinezdequel/articulo-1-data`.
- `sat_v2`: subido desde el disco local (`sat_v2.zip`).
- Las tres carpetas y los dos CSV coinciden fichero a fichero (MD5) con
  `analysis/manifest_dataset_v4_ms_sx200.csv`, comprobado en la propia instancia
  antes de entrenar.

## Tiempos

- 181 minutos por ejecución (entrenamiento de 50 épocas + evaluación); detalle
  en `tiempos.csv`.
- La prueba `B_s7` empezó el 20-10-2026 a las 09:23 UTC; las otras 8 se
  lanzaron después, a las 13:47 UTC, y terminaron el 21-10-2026 a las 13:01 UTC.
  `entorno.txt` recoge la fecha del segundo lanzamiento porque el script lo
  reescribe en cada uno.

## Comparabilidad con resultados anteriores

- Las comparaciones entre brazos se hacen solo dentro de esta tanda.
- Los resultados anteriores ya redactados del split temporal no salieron de esta
  instancia: el entrenamiento de julio de `resunetpp_v3_sp00_s7` procede de la
  suite de Hernán (fichero idéntico en `repo-hernan`) y el de agosto
  (`_preds`), de la RTX 5070 Ti de Moisés.
