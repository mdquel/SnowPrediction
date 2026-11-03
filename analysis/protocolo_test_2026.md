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

Descarga del 31-10-2026 (carpeta MapasOK, enlace de Jesús): llegan de nuevo
los 8 mapas de 2025-26. Solo se calculan las huellas MD5 y se leen los
metadatos; no se abre ningún valor de nieve.
- Los 7 mapas del 09-12-2025 al 07-05-2026 tienen la misma huella que en el
  apartado 9: no han cambiado.
- SD_20260513.tif, versión corregida:
  `b0a13a553262dfd939254856468abdb6`. La anterior era
  `2b27f78f066c855d8503e8e68391da18`.
- Metadatos de los 8: 3696 × 3257 píxeles, 1 m, EPSG:25830, origen
  (709451,87, 4736455,74), float64, sin dato = -9999. El 13-05-2026 ya no
  usa -9998, así que queda resuelta la pregunta del apartado 10.
- Para la evaluación se usa la versión corregida del 13-05-2026.
- La primera entrega (incluida la versión anterior del 13-05-2026) se ha
  borrado; su huella queda en el apartado 9. Los 8 mapas vigentes están en
  `Articulo 1/Data/izas/LiDAR/SnowDepth_2025_26/` (fuera de git), con un
  LEEME.md y sus huellas, comprobadas de nuevo tras moverlos el 31-10-2026.

**8.5. Mapas de marzo de 2024.** Jesús confirma que los mapas del 7 y el
14 de marzo de 2024 tenían un error y los ha corregido. Están en validación,
así que intervienen en la elección de la mejor época. Si se regeneran sus
teselas y se reentrenan los modelos, se hará antes de abrir los datos de
2026, con el mismo protocolo de entrenamiento, y se declarará aquí.

### Enmienda 2 — 3-11-2026 (antes de abrir los datos de 2026)

Motivo: revisión de Hernán Díaz (correo del 2-11-2026).

**8.6. Métricas a 10 m.** Igual que las de 5 m (8.1.3), con bloques de
10 × 10 píxeles: se descartan las 6 últimas filas y columnas de cada tesela
(256 = 25 × 10 + 6) y el bloque cuenta si tiene al menos 51 de 100 píxeles
válidos. Mismas métricas, descriptivas, en 2025 y en 2026.

**8.7. Mapas de marzo de 2024 (aplica 8.5).** Se regeneran las teselas del
7 y del 14 de marzo de 2024 con los mapas corregidos y se reentrenan, con
las mismas semillas y el mismo protocolo de entrenamiento, la red sin
climatología (B) y la red con climatología; la combinación se recalcula con
la nueva red B. Se informa la tabla antes y después en 2025. Los modelos
reentrenados sustituyen a los anteriores en el apartado 4 y son la
referencia del criterio 5 (red de dos cabezas). La climatología no cambia
(temporadas 2021-2023). Todo se hace antes de abrir los datos de 2026.

### Registro — 3-11-2026: imágenes de cobertura de nieve de 2025-26

**8.8.** Llegan las imágenes del producto de cobertura de nieve de la
temporada 2025-26 (enlace de Jesús Revuelto, 3-11-2026): 118 ficheros de 98
fechas, del 03-12-2025 al 28-06-2026. Solo se han consultado sus metadatos,
sus códigos y la fracción de píxeles observados; ninguna estadística de nieve.
- Mismo formato que las temporadas anteriores: rejilla de 10 m (941 × 1025),
  EPSG:25830, códigos 0 (sin observación), 10 (sin nieve) y 11 (nieve).
- Sin imágenes de octubre ni noviembre, como en todas las temporadas
  anteriores (empiezan el 1 de diciembre). Las ventanas de persistencia del
  vuelo del 09-12-2025 solo contienen tres fechas (3, 5 y 8 de diciembre).
- En 18 fechas hay dos o tres ficheros (sufijos «(1)» y «(2)»), que cubren
  zonas en parte distintas y coinciden en un 93-100 % donde se solapan. Se
  aplica la misma regla que en 2021-2025 (`build_sce_index` y
  `patch_sat_channels.py` solo leen `izas_AAAAMMDD.tif`): se usa el fichero
  sin sufijo. Pendiente de confirmar con Jesús qué son; si cambia la regla,
  se declarará aquí antes de abrir los datos de 2026.
- Huellas MD5 de los 118 ficheros: `analysis/sce_2025_26_md5.txt`.

### Registro — 3-11-2026: regeneración de las teselas de marzo de 2024 (aplica 8.7)

**8.9.** Se escribe antes de reentrenar y antes de ver ningún resultado.

*Mapas usados.* `SD_20240307.tif` (`ae0351e09c721e5c12ccd8d2c70658bc`) y
`SD_20240314.tif` (`ef7916863ee111fa66555b987f82ed68`), en
`Articulo 1/Data/izas/LiDAR/SnowDepth_2024_corregidos/`. Se convierten al
formato de los originales (sin dato -9999 a NaN) y se teselan con el mismo
código del generador (`data/regen_mapas_2024.py`, que importa
`data/generate_dataset_v4_ms_sx200.py`). Ejecutado en Vast.ai.

*Control de reproducción.* Antes de escribir nada se regeneran las 95
teselas del 13-02-2024 desde su mapa original y se comparan con las
existentes. Máscara (HS) idéntica y 14 de los 22 canales idénticos; difieren
por redondeo de float32 los canales derivados del MDE:

| Canales | Qué son | Diferencia máxima |
|---|---|---|
| 1-3, 18-20 | pendiente, northness, eastness (1 m y 5 m) | 1e-5 |
| 17 | MDE a 5 m (2000-2300 m) | 0,00195 m |
| 21 | TPI a 5 m | 0,0015 m |

Se atribuye a versiones distintas de las bibliotecas (instancia: opencv
4.14.0, numpy 2.5.4, GDAL 3.12.2; las originales no constan). Con opencv
5.0.0 el resultado es el mismo. Se acepta una tolerancia de 0,005 y el
control queda superado (0 teselas por encima).

*Resultado.* Teselas del 07-03-2024: 52 → 56; del 14-03-2024: 52 → 82
(antes las dos fechas usaban el mismo mapa). Validación: 431 → 465; total:
4892 → 4926. Entrenamiento (2021-2023) y test (2025) no cambian, ni las
estadísticas de normalización, que salen de entrenamiento. La posición de
las teselas nuevas coincide con el MDE (diferencia 0). Las teselas antiguas
se guardan en `_reemplazadas_mapas2024/` dentro del dataset, y los CSV
anteriores como `dataset_v4_ms_sx200_v1.csv` y `data/tile_georef_v1.csv`.
Resumen: `analysis/regen_mapas_2024.json`.

*Canales añadidos.* sat_v2 y clim_v1 de las 138 teselas nuevas con los
mismos scripts (`patch_sat_channels.py`, `patch_clim_channels.py`); controles
de clim_v1 superados (igualdad con el experimento B, 1e-6; causalidad).
Resúmenes: `analysis/sat_v2_canales_regen2024.json` y
`analysis/clim_v1_canales_regen2024.json`.

*No se actualiza* `dataset_v4_ms_sx200_spatial.csv`: sus filas del 7 y del
14 de marzo de 2024 quedan desfasadas. Ningún modelo del apartado 4 usa el
split espacial; si se repite un experimento espacial, se regenerará antes.

*Reentrenamiento.* `scripts/vast_run_regen2024.sh`: B y la red con
climatología, semillas 7, 42 y 123, mismas configuraciones salvo las
carpetas de salida (`results/regen2024/`). La combinación se recalcula con la
nueva B; la climatología sola se vuelve a evaluar como control (mismas
teselas de test: debe salir igual). Se informan las dos tablas, antes y
después, con la mejor época y con la última. Los modelos nuevos son los del
apartado 4 y la referencia del criterio 5, sea cual sea el resultado.

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
