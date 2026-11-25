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

### Resultados (25-11-2026)

Ficheros: `analysis/exploratorio_climatologia.json` (cifras) y
`analysis/exploratorio_climatologia.log` (salida del script). Ejecutado en una
instancia Vast.ai (RTX 5070 Ti, sin usar la GPU) con las máscaras del pendrive,
que ya tienen corregido marzo de 2024; se excluyeron las de 2025.

**Control previo superado:** la variante A reproduce C̄(d) y L(d) de
`analysis/niveles_por_fecha_clim_v2.csv` en las 7 fechas de 2024.

| | Climatología | Combinación | R² | RMSE (m) | Sesgo (m) | Correlación por tesela |
|---|---|---|---|---|---|---|
| A | media | sumando | 0,679 | 0,460 | +0,020 | 0,663 |
| B | mediana | sumando | 0,637 | 0,489 | +0,032 | 0,651 |
| C | media | multiplicando | 0,561 | 0,538 | −0,021 | 0,663 |
| D | dibujo | multiplicando | 0,425 | 0,616 | −0,019 | 0,677 |

La correlación por tesela se calculó en las 465 teselas de validación.

**Según el criterio registrado, ninguna variante mejora:**

- B frente a A: R² −0,042, correlación −0,011, mejor en 2 de 7 fechas.
- D frente a C: R² −0,136, correlación +0,014, mejor en 3 de 7 fechas.
- Informativo, C frente a A (multiplicar frente a sumar): R² −0,118.

**Conclusión.** La climatología actual (media de los vuelos, combinación
sumando) es la mejor de las cuatro en 2024. La mediana es peor en casi todas
las fechas. La del dibujo tiene la correlación más alta (el dibujo en "veces"
se parece algo más al real), pero pierde al aplicarse multiplicando.

### Observación posterior: sumar o multiplicar según la nieve del día

Vista **al mirar los resultados**; no es una prueba registrada.

R² por fecha de 2024, con la climatología media (A sumando, C multiplicando):

| Fecha | Nieve media L(d) | A (sumando) | C (multiplicando) | Mejor |
|---|---|---|---|---|
| 25-01-2024 | 0,16 m | −0,760 | +0,446 | multiplicar |
| 13-02-2024 | 0,60 m | −0,276 | +0,015 | multiplicar |
| 07-03-2024 | 1,33 m | +0,712 | +0,228 | sumar |
| 14-03-2024 | 1,55 m | +0,570 | +0,264 | sumar |
| 05-04-2024 | 1,52 m | +0,659 | +0,209 | sumar |
| 08-05-2024 | 0,64 m | +0,425 | +0,433 | igual |
| 05-06-2024 | 0,16 m | +0,073 | +0,172 | multiplicar |

R² por fecha de las cuatro variantes:

| Fecha | A | B | C | D |
|---|---|---|---|---|
| 25-01-2024 | −0,760 | −1,335 | +0,446 | +0,518 |
| 13-02-2024 | −0,276 | −0,745 | +0,015 | −0,237 |
| 07-03-2024 | +0,712 | +0,675 | +0,228 | −0,152 |
| 14-03-2024 | +0,570 | +0,590 | +0,264 | +0,056 |
| 05-04-2024 | +0,659 | +0,660 | +0,209 | −0,316 |
| 08-05-2024 | +0,425 | +0,358 | +0,433 | +0,475 |
| 05-06-2024 | +0,073 | −0,207 | +0,172 | +0,232 |

**Qué se ve.** Sumar es claramente mejor en las fechas de mucha nieve (más de
1,3 m de media) y multiplicar en las de poca (0,16 m). Con un nivel intermedio
(0,64 m) dan lo mismo. El R² total favorece a sumar porque las fechas de mucha
nieve tienen más píxeles con nieve y más variación, y pesan más.

**Explicación física probable.**

- Con poca nieve, sumar resta lo mismo en todas partes: la climatología de un
  año medio (unos 0,7 m) baja en bloque hasta 0,16 m y casi todo el mapa queda a
  cero. Solo sobreviven los hoyos, y el dibujo se pierde. Multiplicar lo escala
  (×0,2) y conserva el dibujo.
- Con mucha nieve, multiplicar exagera el contraste (×2): los hoyos se disparan
  y las crestas apenas suben. En la realidad las crestas también ganan algo de
  nieve, y eso se parece más a sumar.

**Por qué no es una prueba.** La idea salió al mirar 2024, que es también donde
se midió, y son solo 7 fechas de un año. Para comprobarla habría que:

1. registrar antes una regla concreta, por ejemplo multiplicar por debajo de un
   nivel y sumar por encima, o una mezcla de las dos que dependa de L(d), con su
   umbral fijado de antemano;
2. elegir el umbral sin usar 2024 (con los años de entrenamiento);
3. medirla en años distintos de 2024 sin repetir el uso del test de 2025, por
   ejemplo con validación cruzada dentro de 2021–2023 o con 2026 si se registra
   antes de abrirlo.

**Por qué interesa.** El paso de la climatología al día ya no sería una suma
fija, sino que dependería de cuánta nieve hay. Encaja con lo que intentaba
aprender la red de dos cabezas con w (el contraste según el nivel), pero con
una regla física sencilla y sin entrenar nada.
