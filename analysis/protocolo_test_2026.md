# Protocolo de evaluación en la temporada 2025-26 (test independiente)

Fijado el 24-10-2026, **antes de abrir ni calcular nada sobre los valores de
nieve de 2026**. Lo único consultado de esos datos son los nombres de los
ficheros, sus metadatos (rejilla, coordenadas, valor sin dato) y sus huellas
MD5. Una vez comiteado este documento, no se cambia; cualquier desviación se
anota en el apartado 8 con su motivo y su fecha.

## 1. Por qué

El test de 2025 se ha mirado muchas veces mientras se tomaban decisiones
(qué brazo seguir, qué lectura hacer, la combinación exploratoria). La
temporada 2025-26 no ha influido en ninguna decisión, así que es la prueba
independiente de lo que se haya concluido con 2025.

## 2. Datos

- Mapas de HS de los 8 vuelos de la temporada 2025-26: 09-12-2025,
  23-02-2026, 04-03-2026, 12-03-2026, 24-03-2026, 09-04-2026, 07-05-2026 y
  13-05-2026 (envío de Jesús del 23-10-2026). Huellas MD5 en el apartado 9.
- Imágenes del producto de cobertura de nieve de Sentinel-2 de la misma
  temporada (las calcula el IPE-CSIC). Sin ellas no se evalúa.
- Se usan los 8 vuelos. Si dos mapas resultan idénticos (como el 7 y el 14
  de marzo de 2024), se conserva solo uno y se anota.

## 3. Preparación (antes de evaluar, sin mirar métricas)

1. Los mapas de 2026 vienen en otro formato que los de 2021-2025: la rejilla
   del modelo del terreno, EPSG:25830, sin dato = -9999 (-9998 en el
   13-05-2026). El generador de teselas se adapta para tratar ambos valores
   como «sin dato», sin cambiar nada más.
2. Se generan las teselas de 2026 con el mismo generador, los mismos 22
   canales, los canales de satélite v2 (`data/patch_sat_channels.py`) y la
   climatología (`data/patch_clim_channels.py`). Se generan 32 canales: los
   22 originales (0-21), los 8 de satélite v2 (22-29) y los 2 de
   climatología (30-31). Cada modelo elige los suyos con su configuración:
   la red B usa 26 (los originales sin el 5 ni el 14-16, que sustituyen los
   de satélite v2, más los 22-29) y la red con climatología, 28 (los de B
   más el 30 y el 31).
3. **Climatología de 2026: temporadas 2021, 2022 y 2023**, las mismas que la
   de 2025. Los modelos se entrenaron con esa climatología; cambiarla
   introduciría una segunda diferencia. Como análisis aparte, declarado como
   tal, se puede añadir la climatología con 2021-2025.
4. Controles que deben superarse antes de evaluar:
   - las teselas de 2026 reconstruyen el modelo del terreno en su posición
     (como `data/tile_georef.csv` para el resto);
   - el control de `patch_sat_channels.py` para las fechas de 2026;
   - ningún píxel con -9999 o -9998 llega a los objetivos ni a la pérdida;
   - comparación píxel a píxel entre fechas para descartar duplicados.
   Si algún control falla, se corrige la preparación y se repite; los
   controles no miran métricas de los modelos.

## 4. Modelos (congelados: no se reentrena nada)

Mejor época de cada entrenamiento, 3 semillas (7, 42, 123):

| Nombre | Qué es |
|---|---|
| **Combinación (principal)** | Climatología desplazada, por fecha, hasta la media que predice la red B del paso 1 sobre los píxeles volados (`scripts/clim_v1_hybrid.py`) |
| Red con climatología | Bclim del paso 2, tal cual |
| Red con climatología apagada | El mismo Bclim con la climatología a 0 |
| Red B | Brazo B del paso 1 |
| Climatología sola | Sin red |

**Modelo opcional: red de dos cabezas (paso 3).** Todavía no está entrenado
y depende de la valoración (esto lo comento con Hernán). Es la misma ResUNet++ con las mismas
28 entradas que la red con climatología, pero con dos salidas por píxel:
un peso w entre 0 y 1 (cuánto fiarse de la climatología) y una corrección
ΔHS. La predicción es HS = w · climatología + ΔHS, y donde no hay
climatología w vale 0, de modo que todo lo pone ΔHS. Es la versión
aprendida de la combinación explícita. Si se entrena antes de evaluar 2026,
con el mismo protocolo de entrenamiento y 3 semillas, se añade a la tabla
como un modelo más, con las mismas métricas y el criterio 5 del apartado 6.
Si no, no se evalúa en este test.

## 5. Métricas

Las mismas de 2025, con el mismo código:
- R² y SPAEF sobre los píxeles con nieve (HS > 0,01 m) y dato válido;
- dos lecturas: **tal cual** y **con el nivel corregido** por fecha
  (`scripts/clim_v1_level_corrected.py`);
- también con la última época, como lectura secundaria;
- por fecha, para ver si el resultado depende de la época de la temporada.

## 6. Criterios, fijados de antemano

Referencia de ruido: 0,0125 de R² (repetir el mismo entrenamiento).

1. **La combinación supera a la red B tal cual**: diferencia de medias de R²
   mayor que 0,0125 **y** mejora en las 3 semillas.
2. **La combinación supera a la red con climatología tal cual**: mismo
   criterio.
3. **La red con climatología supera a la red B tal cual**: mismo criterio.
4. **La red con la climatología apagada no empeora frente a B** más de
   0,0125 de media.
5. **Solo si se evalúa la red de dos cabezas:** supera a la combinación
   explícita tal cual, con el mismo criterio que el 1.

Cada criterio se informa como cumplido o no cumplido, tal cual sale.

## 7. Qué se publica

Todo lo anterior, se cumplan o no los criterios, con la tabla por semilla y
por fecha. Una sola evaluación: después de verla no se cambia ningún modelo,
dato ni criterio.

## 8. Desviaciones

### Enmienda 1 — 31-10-2026 (antes de abrir los datos de 2026)

Motivo: revisión de Hernán Díaz (correo del 27-10-2026). Se añade antes de
descargar o calcular nada sobre los valores de nieve de 2026.

**8.1. Métricas nuevas.** Se informan para todos los modelos del apartado 4,
con la mejor época y con la última, en 2025 y en 2026. Son descriptivas: los
criterios del apartado 6 no cambian.

1. Correlación de patrón por tesela: coeficiente de Pearson entre la HS
   observada y la predicha, sobre los mismos píxeles que el R² (con nieve,
   HS > 0,01 m, y dato válido). Se calcula en las mismas teselas que el SPAEF
   y se resume con la media sin ponderar y la mediana. No depende del nivel:
   es la medida del dibujo.
2. RMSE y MAE en metros, sobre los mismos píxeles que el R².
3. Métricas a 5 m: dentro de cada tesela, bloques de 5 × 5 píxeles (las
   últimas filas y columnas que no completan un bloque se descartan). El
   valor del bloque es la media de sus píxeles válidos; el bloque cuenta si
   tiene al menos 13 de 25 píxeles válidos. Sobre los bloques con nieve
   (media observada > 0,01 m) se calculan el R², el SPAEF, la correlación de
   patrón, el RMSE y el MAE, igual que a 1 m.

**8.2. Error de nivel.** Para cada modelo y fecha: media predicha menos media
observada sobre los píxeles con nieve de la fecha, en metros. Se informa por
fecha y como media de su valor absoluto.

**8.3. Presentación.** En dos pasos: primero las opciones aplicables, tal
cual; después la lectura con el nivel corregido, como diagnóstico, no como
opción real.

**8.4. Mapa del 13-05-2026.** Jesús Revuelto lo ha sustituido por una versión
corregida (correo del 27-10-2026). La huella del apartado 9 queda como
registro de la versión anterior; la nueva se añade aquí al descargarla,
consultando solo su MD5 y sus metadatos.

**8.5. Mapas de marzo de 2024.** Jesús confirma que los mapas del 7 y el
14 de marzo de 2024 tenían un error y los ha corregido. Están en validación,
así que intervienen en la elección de la mejor época. Si se regeneran sus
teselas y se reentrenan los modelos, se hará antes de abrir los datos de
2026, con el mismo protocolo de entrenamiento, y se declarará aquí.

## 9. Huellas MD5 de los mapas recibidos

```
8963d8affbc13994793267392c683efc  SD_20251209.tif
f435db60c7cd696880a52efeca615442  SD_20260223.tif
96dd53eae1b0da031c249591ec8730cf  SD_20260304.tif
f6519b3a2dd63ca253995c55ac6259c3  SD_20260312.tif
72a766228b9742e171aa289cd002e73b  SD_20260324.tif
8da9287b83761c503631573d4dbbf0cc  SD_20260409.tif
3a6025284ab89a50c2da6e9997296425  SD_20260507.tif
2b27f78f066c855d8503e8e68391da18  SD_20260513.tif
```

## 10. Pendiente de confirmar con Jesús (no cambia el protocolo)

- Con qué dron se volaron los vuelos de 2025-26.
- Si la referencia sin nieve sigue siendo la del 26-06-2021 y el procesado es
  el mismo que en 2021-2025.
- Por qué el 13-05-2026 usa -9998 como valor sin dato.
