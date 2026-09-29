r"""
Diagnostico del redondeo de la georreferencia
=============================================

tile_georef.py coloca cada tile en la rejilla del DEM con un error menor de
un pixel (residuo mediano 0.13 m frente a 20-40 m sin alinear). Los origenes
de los vuelos son fraccionarios, y el residuo depende de como resolvio el
generador esa fraccion al leer el DEM.

Se prueban cuatro hipotesis y se mide, para una muestra de tiles, la
diferencia entre el canal de elevacion del tile y el DEM:

    redondeo      origen redondeado al entero mas cercano
    truncado      origen truncado hacia abajo
    techo         origen redondeado hacia arriba
    bilineal      DEM interpolado en la posicion fraccionaria exacta

La que de residuo ~0 es la que uso el generador.

Por que importa mas alla de la precision
----------------------------------------
El generador proyecta la NIEVE sobre la rejilla del recorte, que es
fraccionaria (window_transform de una ventana no entera). Si el DEM se
leyo tambien en esa rejilla fraccionaria (bilineal), entradas y objetivo
estan en la misma rejilla y no hay desfase interno. Si el DEM se leyo
redondeando a entero, la topografia y la nieve de cada tile estarian
desplazadas entre si hasta medio pixel.

Uso
---
    python scripts/check_georef_rounding.py
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import rasterio

TILE = 256
SNOW_DIR = os.path.join('Articulo 1', 'Data', 'izas', 'LiDAR', 'SnowDepth')
DEM_PATH = os.path.join('Articulo 1', 'Data', 'izas', 'LiDAR', 'Topografia',
                        'DEMbigIzas_1m.tif')
CSV_MAIN = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
IMAGES_DIR = 'dataset_v4_ms_sx200/images'


def bilinear_patch(dem, r, c):
    """Parche TILE x TILE del DEM en la posicion fraccionaria (r, c)."""
    r0, c0 = int(math.floor(r)), int(math.floor(c))
    fr, fc = r - r0, c - c0
    a = dem[r0:r0 + TILE + 1, c0:c0 + TILE + 1]
    if a.shape != (TILE + 1, TILE + 1):
        return None
    return ((1 - fr) * (1 - fc) * a[:-1, :-1] + (1 - fr) * fc * a[:-1, 1:]
            + fr * (1 - fc) * a[1:, :-1] + fr * fc * a[1:, 1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='tiles de la muestra')
    args = ap.parse_args()

    dem_ds = rasterio.open(DEM_PATH)
    dem = dem_ds.read(1).astype(np.float64)
    origins = {}
    for fn in sorted(os.listdir(SNOW_DIR)):
        m = re.match(r'SD_(\d{8})_1m\.tif$', fn)
        if m:
            with rasterio.open(os.path.join(SNOW_DIR, fn)) as sd:
                w = dem_ds.window(*sd.bounds)
            origins[m.group(1)] = (w.row_off, w.col_off)

    df = pd.read_csv(CSV_MAIN)
    df['date'] = df['tile_id'].str[:8]
    df = df[df['date'].isin(origins)].sample(min(args.n, len(df)), random_state=1)

    metodos = {
        'redondeo': lambda x: int(round(x)),
        'truncado': lambda x: int(math.floor(x)),
        'techo':    lambda x: int(math.ceil(x)),
    }
    res = {k: [] for k in list(metodos) + ['bilineal']}
    por_fecha = {}

    for t, d in zip(df['tile_id'], df['date']):
        path = os.path.join(IMAGES_DIR, t)
        if not os.path.exists(path):
            continue
        m = re.match(r'\d{8}_lidar_tile_(\d+)_(\d+)\.npy', t)
        ty, tx = int(m.group(1)), int(m.group(2))
        tile = np.load(path)[0].astype(np.float64)
        ok = np.isfinite(tile) & (tile > -100)
        if ok.sum() < 100:
            continue
        ro, co = origins[d]
        fila = {}
        for k, f in metodos.items():
            ref = dem[f(ro) + ty:f(ro) + ty + TILE, f(co) + tx:f(co) + tx + TILE]
            if ref.shape == tile.shape:
                v = float(np.median(np.abs(tile[ok] - ref[ok])))
                res[k].append(v)
                fila[k] = v
        ref = bilinear_patch(dem, ro + ty, co + tx)
        if ref is not None:
            v = float(np.median(np.abs(tile[ok] - ref[ok])))
            res['bilineal'].append(v)
            fila['bilineal'] = v
        por_fecha.setdefault(d, []).append(fila)

    print(f"Diferencia de elevacion tile vs DEM ({len(res['redondeo'])} tiles):")
    print(f"  {'metodo':<12}{'mediana':>10}{'p95':>10}{'maxima':>10}")
    for k, v in res.items():
        v = np.array(v)
        if len(v):
            print(f'  {k:<12}{np.median(v):>10.4f}{np.percentile(v, 95):>10.4f}{v.max():>10.4f}')
    mejor = min(res, key=lambda k: np.median(res[k]) if res[k] else 1e9)
    print(f'\n  Mejor ajuste: {mejor}')

    print('\nMejor metodo por fecha (con la parte fraccionaria del origen):')
    for d in sorted(por_fecha):
        filas = por_fecha[d]
        med = {k: np.median([f[k] for f in filas if k in f]) for k in res if any(k in f for f in filas)}
        k = min(med, key=med.get)
        ro, co = origins[d]
        print(f'  {d}  fraccion fila {ro % 1:.2f} col {co % 1:.2f}  ->  {k:<9} ({med[k]:.4f} m)')

    print()
    if mejor == 'bilineal':
        print('Interpretacion: el DEM se leyo en la rejilla FRACCIONARIA del vuelo,')
        print('la misma sobre la que se proyecto la nieve. Entradas y objetivo estan')
        print('alineados entre si: no hay desfase interno.')
    else:
        print('Interpretacion: el DEM se leyo redondeando la posicion a entero, pero la')
        print('nieve se proyecto sobre la rejilla fraccionaria. Puede haber un desfase')
        print('interno de hasta medio pixel entre topografia y nieve en cada tile.')


if __name__ == '__main__':
    main()
