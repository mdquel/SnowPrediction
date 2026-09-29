r"""
Georreferenciacion de los tiles y comprobacion de fuga del split espacial
=========================================================================

Problema detectado
------------------
El generador del dataset (data/generate_dataset_v4_ms_sx200.py, linea 305)
recorta cada fecha con los limites de SU raster de nieve:

    win = src_dem.window(*src_target.bounds)

y numera los tiles desde la esquina de ese recorte. Como cada vuelo cubre
un area algo distinta, cada fecha tiene su propio origen: "tile_0_512"
no es un sitio fijo del terreno, sino "512 m a la derecha del borde del
vuelo de ese dia". Se comprobo empiricamente: el canal de elevacion de dos
tiles con el mismo nombre difiere 20-40 m (mediana) entre fechas.

Consecuencias posibles
----------------------
* Cualquier analisis que empareje tiles entre fechas por su nombre esta
  mal alineado (p. ej. la climatologia del experimento B).
* El split espacial (data/make_spatial_split.py) asigna bandas de columnas
  por NOMBRE de tile, asumiendo que la misma (row, col) es el mismo sitio
  en todas las fechas. Si los orígenes difieren lo bastante, terreno de
  test podria aparecer en entrenamiento en otras fechas.

La red NO se ve afectada: cada tile se predice con sus propias entradas.

Que hace este script
--------------------
1. Para cada raster SD_<fecha>_1m.tif, reproduce exactamente el calculo del
   generador (dem.window(*sd.bounds)) y obtiene el origen del recorte en la
   rejilla fija del DEM de 1 m.
2. Situa cada tile en coordenadas comunes:
       fila_real    = row_off(fecha) + ty
       columna_real = col_off(fecha) + tx
3. VALIDA la correspondencia: el canal 0 de cada tile (elevacion) debe
   coincidir con el DEM leido directamente en esas coordenadas. Si la
   diferencia es ~0, la georreferencia es exacta.
4. Comprueba la fuga del split espacial: que fraccion del terreno cubierto
   por tiles de TEST (cualquier fecha) esta cubierto tambien por tiles de
   TRAIN o VAL (cualquier fecha).
5. Guarda data/tile_georef.csv: tile_id -> fila y columna reales, para
   reutilizarlo (p. ej. en la climatologia alineada).

Uso
---
    python scripts/tile_georef.py
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
CSV_SPATIAL = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200_spatial.csv'
IMAGES_DIR = 'dataset_v4_ms_sx200/images'
OUT_CSV = 'data/tile_georef.csv'


def parse_tile(tile_id):
    m = re.match(r'(\d{8})_lidar_tile_(\d+)_(\d+)\.npy', str(tile_id))
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    ap = argparse.ArgumentParser(description='Georreferencia de tiles y fuga del split espacial')
    ap.add_argument('--snow-dir', default=SNOW_DIR)
    ap.add_argument('--dem', default=DEM_PATH)
    ap.add_argument('--images', default=IMAGES_DIR)
    ap.add_argument('--n-validate', type=int, default=400,
                    help='tiles a validar contra el DEM (muestra aleatoria)')
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Origen de cada fecha en la rejilla del DEM (igual que el generador)
    # ------------------------------------------------------------------
    dem = rasterio.open(args.dem)
    print(f'DEM: {dem.width} x {dem.height} px, resolucion {dem.res}')
    origins = {}
    for fn in sorted(os.listdir(args.snow_dir)):       # NO recursivo, como el generador
        m = re.match(r'SD_(\d{8})_1m\.tif$', fn)
        if not m:
            continue
        with rasterio.open(os.path.join(args.snow_dir, fn)) as sd:
            win = dem.window(*sd.bounds)
        origins[m.group(1)] = (win.row_off, win.col_off, win.height, win.width)

    print(f'\n{len(origins)} rasters de nieve. Origen de cada vuelo en la rejilla del DEM:')
    r0 = min(v[0] for v in origins.values())
    c0 = min(v[1] for v in origins.values())
    print(f"  {'fecha':<10}{'fila':>10}{'columna':>10}{'alto':>8}{'ancho':>8}")
    no_entero = []
    for d, (ro, co, h, w) in sorted(origins.items()):
        print(f'  {d:<10}{ro - r0:>10.1f}{co - c0:>10.1f}{h:>8.0f}{w:>8.0f}')
        if abs(ro - round(ro)) > 0.01 or abs(co - round(co)) > 0.01:
            no_entero.append(d)
    rows = [v[0] for v in origins.values()]
    cols = [v[1] for v in origins.values()]
    print(f'\n  rango de desplazamiento: {max(rows)-min(rows):.0f} m en N-S, '
          f'{max(cols)-min(cols):.0f} m en E-O')
    grupos = len({(round(v[0]), round(v[1])) for v in origins.values()})
    print(f'  origenes distintos: {grupos} para {len(origins)} fechas')
    if no_entero:
        print(f'  origen fraccionario en {len(no_entero)} de {len(origins)} fechas '
              f'(esperado: se trunca, ver check_georef_rounding.py)')

    # ------------------------------------------------------------------
    # 2. Coordenadas reales de cada tile
    # ------------------------------------------------------------------
    sp = pd.read_csv(CSV_SPATIAL)
    parsed = sp['tile_id'].map(parse_tile)
    sp['date'] = [p[0] for p in parsed]
    sp['ty'] = [p[1] for p in parsed]
    sp['tx'] = [p[2] for p in parsed]
    faltan = sorted(set(sp['date']) - set(origins))
    if faltan:
        print(f'\n  [aviso] fechas del dataset sin raster: {faltan}')
    sp = sp[sp['date'].isin(origins)].copy()
    # El generador leyo el DEM TRUNCANDO el origen fraccionario de la ventana
    # (comprobado con check_georef_rounding.py: residuo 0 en 25 de 26 fechas).
    sp['row_real'] = [math.floor(origins[d][0]) + ty for d, ty in zip(sp['date'], sp['ty'])]
    sp['col_real'] = [math.floor(origins[d][1]) + tx for d, tx in zip(sp['date'], sp['tx'])]

    # ------------------------------------------------------------------
    # 3. Validacion contra el DEM
    # ------------------------------------------------------------------
    print('\nValidando la georreferencia contra el DEM...')
    dem_arr = dem.read(1).astype(np.float64)
    nod = dem.nodata
    muestra = sp.sample(min(args.n_validate, len(sp)), random_state=0)
    difs, n_ok, malas = [], 0, {}
    for _, r in muestra.iterrows():
        path = os.path.join(args.images, r['tile_id'])
        if not os.path.exists(path):
            continue
        tile_dem = np.load(path)[0].astype(np.float64)
        ref = dem_arr[r['row_real']:r['row_real'] + TILE, r['col_real']:r['col_real'] + TILE]
        if ref.shape != tile_dem.shape:
            continue
        ok = np.isfinite(tile_dem) & np.isfinite(ref) & (tile_dem > -100) & (ref > -100)
        if nod is not None:
            ok &= ref != nod
        if ok.sum() < 100:
            continue
        v = float(np.median(np.abs(tile_dem[ok] - ref[ok])))
        difs.append(v)
        if v > 0.01:
            malas.setdefault(r['date'], []).append(v)
        n_ok += 1
    difs = np.array(difs)
    print(f'  tiles validados: {n_ok}')
    print(f'  diferencia mediana de elevacion tile vs DEM: '
          f'mediana {np.median(difs):.4f} m, maxima {difs.max():.4f} m')
    if malas:
        for d, v in sorted(malas.items()):
            print(f'  fecha con residuo no nulo: {d}  ({len(v)} tiles, mediana {np.median(v):.3f} m)')
    validada = np.median(difs) < 0.01 and np.percentile(difs, 95) < 0.1
    print('  VALIDACION: ' + ('OK, la georreferencia es exacta.' if validada else
          'NO SUPERADA: la correspondencia no es un simple desplazamiento; '
          'no usar estas coordenadas sin revisar.'))

    # ------------------------------------------------------------------
    # 4. Fuga del split espacial
    # ------------------------------------------------------------------
    print('\nFuga del split espacial (terreno de TEST cubierto por otras bandas):')
    H = sp['row_real'].max() + TILE
    W = sp['col_real'].max() + TILE
    cov = {}
    for s in ('train', 'val', 'test'):
        m = np.zeros((H, W), dtype=bool)
        for rr, cc in sp.loc[sp['exp_spatial_split'] == s, ['row_real', 'col_real']].itertuples(index=False):
            m[rr:rr + TILE, cc:cc + TILE] = True
        cov[s] = m
    n_test = cov['test'].sum()
    for otra in ('train', 'val'):
        inter = (cov['test'] & cov[otra]).sum()
        print(f'  test compartido con {otra:<5}: {inter:>9d} px de {n_test} '
              f'({100 * inter / max(n_test, 1):.1f}% del terreno de test)')

    # fraccion por tile de test: cuanto de cada tile de test se vio en train
    fr = []
    for rr, cc in sp.loc[sp['exp_spatial_split'] == 'test', ['row_real', 'col_real']].itertuples(index=False):
        fr.append(cov['train'][rr:rr + TILE, cc:cc + TILE].mean())
    fr = np.array(fr)
    print(f'  tiles de test con algun pixel visto en train: {(fr > 0).sum()} de {len(fr)}'
          f'  (fraccion media vista: {100 * fr.mean():.1f}%)')

    # ------------------------------------------------------------------
    # 5. Guardar
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    sp[['tile_id', 'date', 'ty', 'tx', 'row_real', 'col_real', 'exp_spatial_split']].to_csv(
        args.out, index=False)
    print(f'\nGuardado: {args.out}')


if __name__ == '__main__':
    main()
