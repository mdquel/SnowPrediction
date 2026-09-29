r"""
Cota del dano del desfase sub-pixel entre topografia y nieve
============================================================

Que se encontro
---------------
check_georef_rounding.py mostro que el generador leyo el DEM TRUNCANDO el
origen fraccionario de cada vuelo, mientras que proyecto la nieve sobre la
rejilla fraccionaria exacta. En cada tile, el mapa de nieve queda
desplazado respecto a la topografia exactamente la parte fraccionaria del
origen: entre 0 y 1 pixel (hasta ~1 m) en cada eje, distinto en cada fecha.
Como varia entre fechas, la red no puede aprenderlo: actua como ruido de
posicion en el objetivo.

Que mide este script
--------------------
Una COTA SUPERIOR del dano. Se toma cada mapa de nieve real, se desplaza
la fraccion de su fecha con interpolacion bilineal, y se calcula el R2 entre
el mapa original y el desplazado. Ningun modelo, por perfecto que fuera,
podria superar ese R2 frente al objetivo tal como esta: es el techo que
impone el desfase.

Se reportan dos cifras:
    R2 fraccion   desplazamiento real de cada fecha
    R2 1 pixel    peor caso posible (desplazamiento de un pixel entero,
                  sin suavizado por interpolacion)

Cautela: la interpolacion bilineal suaviza un poco el mapa, lo que
empeora ligeramente el R2 de la primera cifra. Es una aproximacion, pero
basta para distinguir "irrelevante" de "grave".

Uso
---
    python scripts/check_subpixel_shift.py
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import rasterio

SNOW_DIR = os.path.join('Articulo 1', 'Data', 'izas', 'LiDAR', 'SnowDepth')
DEM_PATH = os.path.join('Articulo 1', 'Data', 'izas', 'LiDAR', 'Topografia',
                        'DEMbigIzas_1m.tif')
CSV_MAIN = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
SNOW_THRESHOLD = 0.01


def shift_bilinear(a, fr, fc):
    """Muestrea a en las posiciones (i - fr, j - fc), 0 <= fr, fc < 1.

    Devuelve el array desplazado sin la primera fila y columna (que
    necesitan un vecino fuera del tile), y el recorte correspondiente del
    original para compararlos pixel a pixel.
    """
    s = ((1 - fr) * (1 - fc) * a[1:, 1:] + (1 - fr) * fc * a[1:, :-1]
         + fr * (1 - fc) * a[:-1, 1:] + fr * fc * a[:-1, :-1])
    return s, a[1:, 1:]


def acc_r2(acc, obs, sim):
    acc['n'] += obs.size
    acc['so'] += obs.sum()
    acc['so2'] += (obs ** 2).sum()
    acc['sr'] += ((obs - sim) ** 2).sum()


def r2_of(acc):
    ss_tot = acc['so2'] - acc['so'] ** 2 / acc['n']
    return 1.0 - acc['sr'] / ss_tot if ss_tot > 0 else float('nan')


def new_acc():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-tiles', type=int, default=60,
                    help='tiles por fecha (muestra)')
    args = ap.parse_args()

    dem = rasterio.open(DEM_PATH)
    frac = {}
    for fn in sorted(os.listdir(SNOW_DIR)):
        m = re.match(r'SD_(\d{8})_1m\.tif$', fn)
        if m:
            with rasterio.open(os.path.join(SNOW_DIR, fn)) as sd:
                w = dem.window(*sd.bounds)
            frac[m.group(1)] = (w.row_off - math.floor(w.row_off),
                                w.col_off - math.floor(w.col_off))

    df = pd.read_csv(CSV_MAIN)
    df['date'] = df['tile_id'].str[:8]

    print(f"{'fecha':<10}{'desplaz. fila':>14}{'desplaz. col':>13}"
          f"{'R2 fraccion':>13}{'R2 1 pixel':>12}")
    tot_f, tot_1 = new_acc(), new_acc()
    filas = []
    for d, g in df.groupby('date'):
        if d not in frac:
            continue
        fr, fc = frac[d]
        a_f, a_1 = new_acc(), new_acc()
        for t in g['tile_id'].head(args.max_tiles):
            path = os.path.join(MASKS_DIR, t)
            if not os.path.exists(path):
                continue
            m = np.load(path).astype(np.float64)
            valid = np.isfinite(m) & (m > -100)
            m = np.where(valid, m, np.nan)

            # desplazamiento real de la fecha
            s, o = shift_bilinear(m, fr, fc)
            ok = np.isfinite(s) & np.isfinite(o) & (o > SNOW_THRESHOLD)
            if ok.sum() > 100:
                acc_r2(a_f, o[ok], s[ok]); acc_r2(tot_f, o[ok], s[ok])

            # peor caso: un pixel entero en diagonal
            s1, o1 = m[:-1, :-1], m[1:, 1:]
            ok1 = np.isfinite(s1) & np.isfinite(o1) & (o1 > SNOW_THRESHOLD)
            if ok1.sum() > 100:
                acc_r2(a_1, o1[ok1], s1[ok1]); acc_r2(tot_1, o1[ok1], s1[ok1])

        if a_f['n']:
            r_f, r_1 = r2_of(a_f), r2_of(a_1)
            filas.append(r_f)
            print(f'{d:<10}{fr:>14.2f}{fc:>13.2f}{r_f:>13.4f}{r_1:>12.4f}')

    print()
    print(f'TOTAL     R2 con el desfase real: {r2_of(tot_f):.4f}   '
          f'peor caso (1 pixel): {r2_of(tot_1):.4f}')
    print(f'          fecha mas afectada: R2 {min(filas):.4f}')
    print()
    print('Lectura: es el R2 maximo alcanzable por un modelo perfecto frente al')
    print('objetivo tal como esta. Compararlo con los R2 de los modelos (0.1-0.5):')
    print('si queda por encima de 0.97-0.98, el desfase es despreciable.')


if __name__ == '__main__':
    main()
