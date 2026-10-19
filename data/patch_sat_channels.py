r"""
Canales satelitales v2: canal 5 y persistencia recodificados (sin GPU)
=======================================================================

Escribe, para cada tesela del dataset v4_ms_sx200, un .npy aparte con 8
canales nuevos. NO toca las teselas originales: los 22 canales actuales
siguen siendo los mismos ficheros, y data/dataset.py anade estos 8 detras
cuando la config trae data.extra_channels_dir. Asi la unica diferencia entre
el modelo actual y los brazos A y B son los canales del satelite.

Problemas que corrige (analysis/sce_dia_del_vuelo.json,
analysis/persistencia_dias_despejados.json):
  * El canal 5 es la imagen SCE del MISMO dia del vuelo; solo 10 de 27 fechas
    la tienen y en el resto el canal vale 0 (60,6 % de las teselas). Ademas se
    remuestrea con bilineal sobre los codigos 0/10/11, que son categoricos.
  * La persistencia cuenta la nube (0) igual que el sin nieve (10): mide en
    buena parte nubosidad.

Canales nuevos (indices 22-29 una vez anadidos a los 22 originales):
  22  sat_nieve         imagen mas reciente de [vuelo - 2 d, vuelo]: (SCE == 11)
  23  sat_observado     misma imagen: (SCE == 10 u 11)
  24  pers_nieve_15d    N_nieve / N_despejado en [vuelo - 15 d, vuelo); 0 si N_despejado = 0
  25  pers_nieve_30d
  26  pers_nieve_60d
  27  frac_despejado_15d  N_despejado / N_imagenes; 0 si no hay imagenes
  28  frac_despejado_30d
  29  frac_despejado_60d

Las mascaras binarias (o sus sumas) se calculan a 10 m y se llevan a la
rejilla de 1 m de cada tesela con bilineal, como hace el generador con la
persistencia; en los bordes el valor es una fraccion con sentido. Fuera de la
huella del SCE quedan a 0 (sat_observado = 0: sin informacion).

Ventanas como el generador: la persistencia usa [vuelo - W, vuelo), sin el
dia del vuelo; el canal del dia usa [vuelo - 2, vuelo] y nunca informacion
posterior al vuelo.

Georreferencia: data/tile_georef.csv (fila_real, columna_real de cada
tesela en la rejilla del DEM de 1 m, validada contra la elevacion por
scripts/tile_georef.py). En algunas fechas esa posicion esta a 1 pixel de la
rejilla que uso el generador (el origen del vuelo es fraccionario y se
redondeo hacia el otro lado): por eso, para cada fecha, se prueban los 9
desplazamientos enteros de -1 a +1 en fila y columna sobre unas teselas, con
la persistencia y el canal 5 actuales como huella, y se aplica el que los
reproduce. El desplazamiento queda anotado por fecha en el resumen.

CONTROL (obligatorio, para el todo el dataset): con la misma via de calculo se
reconstruyen el canal 5 y la persistencia ACTUALES y se comparan con las
teselas existentes. Si la diferencia maxima supera la tolerancia, la rejilla
o la seleccion de imagenes no son las del generador y el script se detiene
sin escribir el resumen como valido.

Salida:
  <dataset>/sat_v2/<tile_id>       float32 (8, 256, 256)
  analysis/sat_v2_canales.json     imagen elegida y desfase por fecha, n de
                                   imagenes por ventana, cobertura y control

Uso (desde la raiz del repo)
----------------------------
    python data/patch_sat_channels.py               # todo el dataset
    python data/patch_sat_channels.py --limit 20    # prueba rapida
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

TILE = 256
WINDOWS = (15, 30, 60)          # PERSIST_WINDOWS del generador
LAG_MAX = 2                     # dias antes del vuelo para el canal del dia
DATE_FIXES_LIDAR = {'20210608': '20210607'}
TOL_CONTROL = 1e-4              # diferencia maxima admitida en el control
SHIFT_SAMPLE = 3                # teselas por fecha para elegir el desplazamiento
CHANNELS = ['sat_nieve', 'sat_observado',
            'pers_nieve_15d', 'pers_nieve_30d', 'pers_nieve_60d',
            'frac_despejado_15d', 'frac_despejado_30d', 'frac_despejado_60d']


def r6(x):
    return None if x is None else round(float(x), 6)


def read_codes(path):
    """Codigos 0/10/11 enteros (los float32 traen ruido de ~1e-17)."""
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        return np.rint(np.where(np.isfinite(a), a, 0.0)).astype(np.int16), s.transform, s.crs


def read_raw(path):
    """Valores tal cual los leia el generador para el canal 5."""
    with rasterio.open(path) as s:
        return s.read(1).astype(np.float32)


def to_tile(src, src_tr, src_crs, dst_tr, dst_crs):
    out = np.zeros((TILE, TILE), np.float32)
    reproject(source=src.astype(np.float32), destination=out,
              src_transform=src_tr, src_crs=src_crs,
              dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=Resampling.bilinear)
    return out


def main():
    ap = argparse.ArgumentParser(description='Canales satelitales v2')
    ap.add_argument('--dataset', default='dataset_v4_ms_sx200')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--sce-dir', default='Articulo 1/Data/izas/LiDAR/images_SCE')
    ap.add_argument('--dem', default='Articulo 1/Data/izas/LiDAR/Topografia/DEMbigIzas_1m.tif')
    ap.add_argument('--out-subdir', default='sat_v2')
    ap.add_argument('--summary', default='analysis/sat_v2_canales.json')
    ap.add_argument('--limit', type=int, default=0, help='solo las N primeras teselas (prueba)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    img_dir = os.path.join(args.dataset, 'images')
    out_dir = os.path.join(args.dataset, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(args.dem) as s:
        dem_tr, dem_crs = s.transform, s.crs
    with open(args.georef, encoding='utf-8') as f:
        georef = list(csv.DictReader(f))
    if args.limit:
        georef = georef[:args.limit]
    by_date = defaultdict(list)
    for r in georef:
        by_date[r['date']].append(r)

    idx = {}
    for p in sorted(glob.glob(os.path.join(args.sce_dir, 'izas_*.tif'))):
        m = re.fullmatch(r'izas_(\d{8})\.tif', os.path.basename(p))
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = p

    summary = {'descripcion': 'Canales satelitales v2 (canal 5 y persistencia recodificados). '
                              'Indices 22-29 detras de los 22 originales.',
               'canales': {22 + i: c for i, c in enumerate(CHANNELS)},
               'parametros': {'ventanas_dias': list(WINDOWS), 'desfase_max_dias': LAG_MAX,
                              'remuestreo': 'bilineal de mascaras binarias o sus sumas, 10 m -> 1 m'},
               'fuentes': {'sce': args.sce_dir, 'georef': args.georef, 'dem': args.dem},
               'por_fecha': {}}
    ctrl_max = {'canal5': 0.0, 'persistencia': 0.0}
    n_written = n_skipped = 0

    for date in sorted(by_date):
        t = datetime.strptime(DATE_FIXES_LIDAR.get(date, date), '%Y%m%d')

        # --- canal del dia: imagen mas reciente de [t - LAG_MAX, t]
        cand = sorted([dt for dt in idx if 0 <= (t - dt).days <= LAG_MAX], reverse=True)
        day = cand[0] if cand else None
        if day is not None:
            c_day, tr_sce, crs_sce = read_codes(idx[day])
            day_snow = (c_day == 11).astype(np.float32)
            day_obs = ((c_day == 10) | (c_day == 11)).astype(np.float32)

        # --- persistencia: sumas a 10 m por ventana [t - W, t)
        win_imgs = {w: [dt for dt in idx if t - timedelta(days=w) <= dt < t] for w in WINDOWS}
        codes = {dt: read_codes(idx[dt]) for dt in set().union(*win_imgs.values())}
        sums = {}
        for w in WINDOWS:
            if win_imgs[w]:
                st = np.stack([codes[dt][0] for dt in win_imgs[w]])
                sums[w] = ((st == 11).sum(0).astype(np.float32),
                           ((st == 10) | (st == 11)).sum(0).astype(np.float32),
                           len(win_imgs[w]))
            else:
                sums[w] = None
        if day is None and codes:
            _, tr_sce, crs_sce = next(iter(codes.values()))
        if day is None and not codes:
            raise SystemExit(f'STOP: {date} sin ninguna imagen SCE (ni del dia ni de las ventanas)')

        # --- control: canal 5 y persistencia ACTUALES reconstruidos. El
        # generador promedia la reproyeccion de cada imagen; como el bilineal es
        # lineal, eso es la reproyeccion de la suma entre N_imagenes, que es la
        # misma suma que usan los canales nuevos (diferencia ~1e-6 por float32).
        same_day = idx.get(t)
        raw5 = read_raw(same_day) if same_day else None

        # --- alineacion: tile_georef.csv puede estar a 1 pixel de la rejilla
        # que uso el generador (origenes fraccionarios redondeados hacia el otro
        # lado; detectado en 20240405, 20240508 y 20250327). Se elige el
        # desplazamiento entero que reproduce la persistencia y el canal 5
        # actuales en una muestra de teselas, y se aplica a toda la fecha.
        def tile_transform(r, dr=0, dc=0):
            return dem_tr * Affine.translation(int(r['col_real']) + dc, int(r['row_real']) + dr)

        def fingerprint_diff(r, dr, dc):
            tr = tile_transform(r, dr, dc)
            old = np.load(os.path.join(img_dir, r['tile_id']), mmap_mode='r')
            diffs = []
            if raw5 is not None:
                diffs.append(np.abs(to_tile(raw5, tr_sce, crs_sce, tr, dem_crs) - old[5]).max())
            for wi, w in enumerate(WINDOWS):
                if sums[w] is not None:
                    p = to_tile(sums[w][0], tr_sce, crs_sce, tr, dem_crs) / sums[w][2]
                    diffs.append(np.abs(p - old[14 + wi]).max())
            return float(max(diffs)) if diffs else 0.0

        sample = by_date[date][:SHIFT_SAMPLE]
        shifts = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        scores = {s: max(fingerprint_diff(r, *s) for r in sample) for s in shifts}
        shift = (0, 0) if scores[(0, 0)] <= TOL_CONTROL else min(scores, key=scores.get)
        informative = raw5 is not None or any(sums[w] is not None for w in WINDOWS)

        d5 = dp = 0.0
        for r in by_date[date]:
            tid = r['tile_id']
            tile_tr = tile_transform(r, *shift)

            new = np.zeros((len(CHANNELS), TILE, TILE), np.float32)
            old_pers = []
            if day is not None:
                new[0] = to_tile(day_snow, tr_sce, crs_sce, tile_tr, dem_crs)
                new[1] = to_tile(day_obs, tr_sce, crs_sce, tile_tr, dem_crs)
            for wi, w in enumerate(WINDOWS):
                if sums[w] is None:
                    old_pers.append(np.zeros((TILE, TILE), np.float32))
                    continue
                n_snow, n_clear, n_img = sums[w]
                s1 = to_tile(n_snow, tr_sce, crs_sce, tile_tr, dem_crs)
                c1 = to_tile(n_clear, tr_sce, crs_sce, tile_tr, dem_crs)
                ok = c1 > 1e-6
                new[2 + wi][ok] = s1[ok] / c1[ok]
                new[5 + wi] = c1 / n_img
                old_pers.append(s1 / n_img)             # persistencia ACTUAL

            # control contra la tesela existente (solo se leen los canales 5 y 14-16)
            old = np.load(os.path.join(img_dir, tid), mmap_mode='r')
            ref5 = (to_tile(raw5, tr_sce, crs_sce, tile_tr, dem_crs) if raw5 is not None
                    else np.zeros((TILE, TILE), np.float32))
            d5 = max(d5, float(np.abs(ref5 - old[5]).max()))
            for wi in range(len(WINDOWS)):
                dp = max(dp, float(np.abs(old_pers[wi] - old[14 + wi]).max()))
            del old

            out_path = os.path.join(out_dir, tid)
            if os.path.exists(out_path) and not args.overwrite:
                n_skipped += 1
                continue
            np.save(out_path, np.clip(new, 0.0, 1.0))
            n_written += 1

        ctrl_max['canal5'] = max(ctrl_max['canal5'], d5)
        ctrl_max['persistencia'] = max(ctrl_max['persistencia'], dp)
        summary['por_fecha'][date] = {
            'n_teselas': len(by_date[date]),
            'imagen_del_dia': day.strftime('%Y%m%d') if day else None,
            'desfase_dias': (t - day).days if day else None,
            'imagen_mismo_dia_en_canal5_actual': bool(same_day),
            'n_imagenes_ventana': {f'{w}d': len(win_imgs[w]) for w in WINDOWS},
            'desplazamiento_georef_fila_col': list(shift),
            'alineacion_verificable': informative,
            'alineacion_dif_sin_desplazar': r6(scores[(0, 0)]),
            'control_max_dif_canal5': r6(d5),
            'control_max_dif_persistencia': r6(dp)}
        print(f"{date}: desplazamiento {shift}, dia {summary['por_fecha'][date]['imagen_del_dia']} "
              f"(desfase {summary['por_fecha'][date]['desfase_dias']}), ventanas "
              f"{[len(win_imgs[w]) for w in WINDOWS]}, {len(by_date[date])} teselas, "
              f"control canal5 {d5:.2e} persistencia {dp:.2e}  ({(time.time() - t0) / 60:.1f} min)")

    ok = ctrl_max['canal5'] <= TOL_CONTROL and ctrl_max['persistencia'] <= TOL_CONTROL
    summary['control'] = {'tolerancia': TOL_CONTROL,
                          'max_dif_canal5_actual': r6(ctrl_max['canal5']),
                          'max_dif_persistencia_actual': r6(ctrl_max['persistencia']),
                          'superado': ok,
                          'lectura': 'reconstruir el canal 5 y la persistencia actuales con la misma '
                                     'via de calculo valida la rejilla y la seleccion de imagenes'}
    summary['teselas'] = {'escritas': n_written, 'ya_existian': n_skipped}
    moved = {d: v['desplazamiento_georef_fila_col'] for d, v in summary['por_fecha'].items()
             if v['desplazamiento_georef_fila_col'] != [0, 0]}
    summary['control']['fechas_con_desplazamiento_georef'] = moved
    if moved:
        print(f'Fechas en que tile_georef.csv esta a 1 pixel de la rejilla del generador: {moved}')
    os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
    with open(args.summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nControl: canal 5 {ctrl_max['canal5']:.2e}, persistencia {ctrl_max['persistencia']:.2e} "
          f"(tolerancia {TOL_CONTROL}) -> {'SUPERADO' if ok else 'NO SUPERADO'}")
    print(f'Teselas escritas {n_written}, ya existian {n_skipped}. Resumen: {args.summary} '
          f'({(time.time() - t0) / 60:.1f} min)')
    if not ok:
        raise SystemExit('STOP: el control no se supera; no uses sat_v2 hasta revisar la rejilla.')


if __name__ == '__main__':
    main()
