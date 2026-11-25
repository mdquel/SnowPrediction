# Pruebas exploratorias de climatología (fuera del protocolo de 2026)

Registrado el 25-11-2026, **antes de calcular nada**. Estas pruebas no cambian
ningún resultado del artículo ni lo registrado en `analysis/protocolo_test_2026.md`
para 2026.

## Pregunta

¿Mejora la combinación si cambiamos cómo se resume la climatología?

## Reglas

- Solo se mide en la **validación de 2024**. Nunca en 2025 ni en 2026.
- Hacia adelante: la climatología de 2024 sale solo de los vuelos de 2021–2023,
  como la clim_v1 (`data/patch_clim_channels.py`).
- Se usa el **nivel observado** de cada fecha, L(d), no el de la red B, para medir
  solo la calidad del dibujo.

## Variantes

Las cuatro se calculan con los 15 vuelos de 2021–2023, sobre la rejilla del DEM,
fecha a fecha (cada vuelo pesa lo mismo), con los mismos píxeles válidos que la
clim_v1 (finitos y > −100).

| Nombre | Climatología | Combinación |
|---|---|---|
| A. Referencia | media de los vuelos (la clim_v1 actual) | sumando: max(0, C − C̄(d) + L(d)) |
| B. Mediana | mediana de los vuelos en cada píxel | sumando |
| C. Control | media de los vuelos | multiplicando: max(0, C · L(d) / C̄(d)) |
| D. Dibujo | media de los vuelos, cada uno dividido antes por su nieve media | multiplicando |

- C̄(d) es la media de la climatología sobre los píxeles volados de la fecha d, y
  L(d) la media de la nieve observada sobre esos mismos píxeles, como en el
  apartado 8.13 del protocolo.
- C sirve para separar efectos: si D mejora a A, C dice si el mérito es del
  dibujo (D frente a C) o de combinar multiplicando (C frente a A).

## Control previo

La variante A tiene que reproducir el C̄(d) de
`analysis/niveles_por_fecha_clim_v2.csv` en las 7 fechas de 2024 (tolerancia
1e-4 m). Si no coincide, se para sin escribir resultados.

## Medidas (en 2024)

- R² y RMSE en los píxeles con nieve (HS > 0,01 m), y sesgo.
- Correlación media por tesela.
- R² por fecha, para ver si una mejora es general o de una sola fecha.

## Qué cuenta como mejora

Se compara B frente a A, y D frente a C. Cuenta como mejora si se cumplen las tres:

- R² al menos **+0,01** mejor;
- correlación por tesela no peor;
- mejor en **al menos 5 de las 7 fechas** de 2024.

## Pendiente para otra tanda

- **Climatología por épocas:** aparcada. La fecha de corte, mirada con la nieve
  media de cada vuelo de 2021–2024 (`analysis/niveles_por_fecha_clim_v2.csv`), es
  poco fiable con tan pocos vuelos, y 2024 tuvo el pico tarde.
- **Ponderar los vuelos por nivel de nieve parecido al del día:** sin fechas de
  corte y sin mirar al futuro (el nivel del día lo daría la red B).

## Resultados

(Se añaden aquí cuando se calculen, sin cambiar nada de lo anterior.)
