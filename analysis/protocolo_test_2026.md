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

### Registro — 4-11-2026: resultado del reentrenamiento y respuesta de Jesús

**8.10. Resultado del reentrenamiento (8.9).** Test 2025, R² tal cual, media
de las 3 semillas (7 / 42 / 123). Fuentes:
`results/regen2024/resumen_antes_despues.json` y
`analysis/clim_v1_hibrido_regen2024.json`.

| Modelo | Mejor época, antes | Mejor época, después | Última época, antes | Última época, después |
|---|---|---|---|---|
| Red B | 0,196 | 0,231 (0,291 / 0,161 / 0,241) | 0,134 | 0,149 |
| Red con climatología | 0,257 | 0,259 (0,267 / 0,257 / 0,253) | 0,154 | 0,152 |
| Misma red, climatología apagada | 0,210 | 0,177 (0,191 / 0,243 / 0,099) | 0,143 | 0,143 |
| Combinación (nivel de B) | 0,385 | 0,400 (0,416 / 0,388 / 0,397) | 0,340 | 0,346 |

- La climatología sola sale idéntica (diferencia 0): control superado.
- Las configuraciones no usan planificador de la tasa de aprendizaje ni
  parada temprana; la validación solo elige la mejor época. Las diferencias
  en la última época son, por tanto, ruido de repetición del entrenamiento
  (de 0,000 a 0,015), del orden de la referencia de 0,0125.
- Criterios del apartado 6 aplicados a 2025 con los modelos nuevos (mejor
  época; descriptivo, la evaluación del protocolo es 2026):
  1 cumplido (+0,169, 3 de 3); 2 cumplido (+0,142, 3 de 3); 3 no cumplido
  (+0,028, pero 2 de 3: en la semilla 7 la red B supera a la red con
  climatología); 4 no cumplido (−0,054, sobre todo por la semilla 123).
  Con los modelos anteriores se cumplían los cuatro.
- Criterio 5: la referencia es la combinación con la nueva B, 0,400
  (0,416 / 0,388 / 0,397 por semilla).

**8.11. Respuesta de Jesús Revuelto (correo del 4-11-2026).**
- *Ficheros duplicados de cobertura de nieve (8.8):* son imágenes de teselas
  distintas del satélite del mismo día. Basta con una; la otra solo tendría
  sentido si la primera está muy nublada. Se mantiene la regla en todas las
  temporadas: el fichero sin sufijo. No se combinan para no cambiar los
  canales respecto a los del entrenamiento (en 2025 cambiaría el
  09-05-2025, que entra en la ventana del vuelo de test del 21-05-2025).
  Queda cerrado el pendiente del 8.8.
- *Imágenes de noviembre:* se pueden procesar desde el 1 de noviembre (antes
  no tiene sentido: el manto persistente empieza en diciembre). Pendiente de
  decidir con Hernán Díaz. Si se incorporan, se regenerarán sat_v2 y los
  modelos antes de abrir los datos de 2026, y se declarará aquí.
- *Mapas de marzo de 2024:* los del enlace de Jesús tienen las mismas huellas
  MD5 que los usados en 8.9. La misma carpeta trae los 8 mapas de 2025-26,
  con las huellas del 8.4 (solo se calcularon las huellas).
- *Meteorología:* los huecos de la serie son fallos de sensor. La serie de
  equivalente en agua de la nieve (SWE) empieza en 2024: no se usa, y la
  temporada 2025-26 no se analiza.

### Enmienda 3 — 10-11-2026 (antes de abrir los datos de 2026)

Motivo: respuesta de Hernán Díaz (correo del 10-11-2026).

**8.12. Imágenes de noviembre.** No se incorporan. La evaluación de 2026 se
hace con las imágenes de cobertura de nieve tal como están (desde el 1 de
diciembre), igual que en 2021-2025.

**8.13. Red de dos cabezas (criterio 5).** Diseño fijado antes de programarla
y de entrenarla.

*Salida.* Para cada fecha *d* y píxel *x*:

    HS = máx(0, L(d) + w · (C(x) − C̄(d)) + Δ)

- C(x): climatología del píxel (m). C̄(d): su media en la fecha.
- L(d): nivel de la fecha.
- w y Δ: una pareja de valores por tesela, que da la red. w está acotado a
  [0,5; 1,5] y Δ a [−0,5; +0,5] m.
- Al empezar, w = 1 y Δ = 0, de modo que la salida es exactamente la
  combinación.

*Nivel.* Las medias se calculan sobre todos los píxeles volados de la fecha,
como en la combinación (`scripts/clim_v1_hybrid.py`):
- en entrenamiento y validación, L(d) es la media observada de la fecha,
  calculada sobre su mosaico, sin contar dos veces el solape de las teselas;
- en el test, L(d) es la media de la predicción de la red B de la misma
  semilla.

Así la red solo aprende a corregir el dibujo; el nivel lo pone B.

*Climatología de entrenamiento.* La causal deja la temporada 2021 sin
climatología (958 de las 4241 teselas de entrenamiento), y con C = 0 la red
tendría que dibujar sola, que es la degeneración que se quiere evitar.
- Para las fechas de entrenamiento se usa una climatología sin la temporada
  propia: cada temporada recibe la media de las otras dos de entrenamiento.
  Se guarda en una carpeta nueva, `clim_v2`, sin tocar `clim_v1`.
- En validación y test, la climatología es la misma que en la combinación
  (2021-2023). Control: idéntica bit a bit a `clim_v1`.
- Limitación declarada: en entrenamiento la climatología es de dos
  temporadas, y en validación y test, de tres.

*Entrenamiento.*
- Mismas entradas que la red con climatología (28 canales), sin apagado
  aleatorio de la climatología.
- Misma pérdida, excluyendo los píxeles sin climatología.
- Mismos hiperparámetros: 50 épocas, semillas 7, 42 y 123.
- La mejor época se elige con la validación de 2024.

*Evaluación.*
- Mismas métricas, píxeles y teselas que el resto de modelos, con la mejor
  época y con la última.
- Se guardan w y Δ de cada tesela, para comprobar que no se quedan en los
  límites.
- Criterio 5: superar a la combinación en más de 0,0125 de R² de media y con
  cada una de las tres semillas, comparando cada semilla con la combinación
  de la misma semilla.
- En 2025 es descriptivo; la evaluación del criterio es la de 2026.

*Revisión.* Hernán Díaz revisa el código antes de los entrenamientos.
Cualquier cambio que salga de la revisión se declara aquí antes de entrenar.

**8.14. Implementación de la red de dos cabezas (antes de entrenar).** Detalles
fijados al programarla, con su comprobación en Vast.ai (10-11-2026):

- *Nivel por época.* En el test, la mejor época de la red usa el nivel de la
  mejor época de la red B de la misma semilla, y la última usa el de la
  última. Es lo mismo que hace la combinación en cada lectura.
- *Validación.* La pérdida de validación, que elige la mejor época, también
  excluye los píxeles sin climatología (el 0,3 % de los píxeles válidos de
  2024).
- *Arquitectura.* Codificador y cuello ASPP de ResUNet++ sin cambios. Como la
  red solo da w y Δ por tesela, no tiene decodificador. Cada cabeza es una
  capa oculta seguida de una salida, y el dropout del modelo (0,077) se
  aplica en esas capas ocultas. Los límites se imponen de forma suave, con
  una tangente hiperbólica. Tiene 21.051.826 parámetros.
- *Nivel observado.* Se calcula con los negativos de la HS observada a 0, que
  es como entra la nieve en la pérdida. Corrige `data/patch_clim_v2.py` antes
  de su primer uso.
- *Comprobaciones antes de entrenar.* Todas superadas:
  - `clim_v2`: las 685 teselas de validación y test son idénticas a
    `clim_v1`, ninguna fecha usa su propia temporada y ninguna tesela de
    entrenamiento queda sin climatología;
  - con la tabla de niveles se reproduce exactamente la combinación
    (diferencia 0);
  - sin entrenar, la red da la combinación en las 220 teselas de test, con
    diferencia ≤ 1,4·10⁻⁶ m, y w = 1 y Δ = 0 en todas, para las tres
    semillas;
  - ningún píxel sin climatología entra en la pérdida;
  - la pérdida de validación al arrancar es 0,1933, que es la referencia de
    la curva de entrenamiento;
  - un entrenamiento de prueba de 1 época (semilla 7) recorre todo el
    circuito. Sus resultados se descartan y no se usan para ninguna decisión.
- *Código:*
  - `models/resunetpp_dos_cabezas.py`, `training/dos_cabezas.py`;
  - `scripts/dos_cabezas_niveles.py`, `scripts/generate_dos_cabezas_configs.py`,
    `scripts/dos_cabezas_control_inicio.py`;
  - cambios en `main.py`, `data/dataset.py` y `models/unet.py`, que no afectan
    a los demás modelos.

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
