r"""
Cobertura del satelite en el dia del vuelo (sin GPU, solo lee los SCE)
=======================================================================

El canal 5 del dataset (data/generate_dataset_v4_ms_sx200.py, bloque B) es la
imagen SCE del MISMO dia del vuelo; si no existe, el canal queda a 0 en toda
la tesela. Este script mide cuanto se recupera si se toma la imagen mas
cercana dentro de una tolerancia de k dias, para elegir k ANTES de
recodificar el canal.

Para cada fecha de vuelo y cada k en {0, 1, 2, 3}, en dos modos:
    pasado      solo imagenes de [vuelo - k, vuelo]        (sin informacion posterior)
    simetrico   imagenes de [vuelo - k, vuelo + k]; a igual distancia, la anterior

y dos maneras de elegir:
    imagen      la imagen mas cercana entera: % del dominio despejado en ella
    pixel       por pixel, la observacion despejada (10 u 11) mas cercana
                dentro de la ventana: % del dominio con alguna

Dominio: pixeles despejados al menos una vez en las imagenes leidas (fuera de
la cuenca el rectangulo vale 0 siempre). Codigos: 11 nieve, 10 sin nieve,
0 sin observacion; los float32 traen ruido de ~1e-17 y se redondean.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/check_sce_flight_day.py
"""

import argparse
import csv
import glob
import json
import os
import re
from datetime import datetime, timedelta

import numpy as np
import rasterio

KS = (0, 1, 2, 3)
DATE_FIXES_LIDAR = {'20210608': '20210607'}     # como el generador


def r4(x):
    return round(float(x), 4)


def read_codes(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
    return np.rint(np.where(np.isfinite(a), a, 0.0)).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description='Cobertura del satelite en el dia del vuelo')
    ap.add_argument('--sce-dir', default='Articulo 1/Data/izas/LiDAR/images_SCE')
    ap.add_argument('--csv', default='dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv')
    ap.add_argument('--out', default='analysis/sce_dia_del_vuelo.json')
    args = ap.parse_args()

    with open(args.csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    dates = sorted({r['date'] for r in rows})
    n_tiles = {d: sum(r['date'] == d for r in rows) for d in dates}

    idx = {}
    for p in sorted(glob.glob(os.path.join(args.sce_dir, 'izas_*.tif'))):
        m = re.fullmatch(r'izas_(\d{8})\.tif', os.path.basename(p))
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = p

    need = {}
    for d in dates:
        t = datetime.strptime(DATE_FIXES_LIDAR.get(d, d), '%Y%m%d')
        for dt in idx:
            if abs((dt - t).days) <= max(KS):
                need[dt] = None
    for dt in need:
        need[dt] = read_codes(idx[dt])
    clear = {dt: (c == 10) | (c == 11) for dt, c in need.items()}
    domain = np.zeros(next(iter(need.values())).shape, bool)
    for c in clear.values():
        domain |= c
    nd = int(domain.sum())

    out = {'descripcion': 'Cobertura del satelite en el dia del vuelo con tolerancia de k dias. '
                          'Porcentajes del dominio (pixeles de 10 m de la cuenca).',
           'fuente': {'sce': args.sce_dir, 'fechas': args.csv,
                      'canal_actual': 'data/generate_dataset_v4_ms_sx200.py, bloque B'},
           'dominio_pixeles_10m': nd, 'por_fecha': {}, 'resumen': {}}

    print(f'{len(dates)} fechas; dominio {nd} pixeles de 10 m\n')
    print(f"{'fecha':<10}{'tes.':>6}  " + '  '.join(f'k={k} pas/sim (img|pix)' for k in KS))
    for d in dates:
        t = datetime.strptime(DATE_FIXES_LIDAR.get(d, d), '%Y%m%d')
        rec = {'n_teselas': n_tiles[d]}
        cells = []
        for k in KS:
            for modo in ('pasado', 'simetrico'):
                lo, hi = -k, (0 if modo == 'pasado' else k)
                cand = sorted([dt for dt in need if lo <= (dt - t).days <= hi],
                              key=lambda dt: (abs((dt - t).days), (dt - t).days > 0))
                if cand:
                    best = cand[0]
                    img_pct = 100 * clear[best][domain].mean()
                    any_clear = np.zeros(domain.shape, bool)
                    for dt in cand:
                        any_clear |= clear[dt]
                    pix_pct = 100 * any_clear[domain].mean()
                    rec[f'k{k}_{modo}'] = {'imagen': best.strftime('%Y%m%d'),
                                           'desfase_dias': (best - t).days,
                                           'pct_despejado_imagen': r4(img_pct),
                                           'pct_despejado_pixel': r4(pix_pct),
                                           'n_imagenes_ventana': len(cand)}
                else:
                    img_pct = pix_pct = 0.0
                    rec[f'k{k}_{modo}'] = {'imagen': None, 'desfase_dias': None,
                                           'pct_despejado_imagen': 0.0,
                                           'pct_despejado_pixel': 0.0,
                                           'n_imagenes_ventana': 0}
                cells.append((img_pct, pix_pct))
        out['por_fecha'][d] = rec
        s = '  '.join(f'{cells[2*i][0]:3.0f}/{cells[2*i+1][0]:3.0f} ({cells[2*i][1]:3.0f}|{cells[2*i+1][1]:3.0f})'
                      for i in range(len(KS)))
        print(f'{d:<10}{n_tiles[d]:>6}  {s}')

    # resumen ponderado por teselas: % de teselas con dato y % medio despejado
    tot = sum(n_tiles.values())
    print(f"\nResumen (ponderado por teselas, {tot} teselas):")
    for k in KS:
        for modo in ('pasado', 'simetrico'):
            key = f'k{k}_{modo}'
            con_img = sum(n_tiles[d] for d in dates if out['por_fecha'][d][key]['imagen'])
            img = sum(n_tiles[d] * out['por_fecha'][d][key]['pct_despejado_imagen'] for d in dates) / tot
            pix = sum(n_tiles[d] * out['por_fecha'][d][key]['pct_despejado_pixel'] for d in dates) / tot
            fechas = sum(bool(out['por_fecha'][d][key]['imagen']) for d in dates)
            out['resumen'][key] = {'fechas_con_imagen': fechas,
                                   'pct_teselas_con_imagen': r4(100 * con_img / tot),
                                   'pct_despejado_medio_imagen': r4(img),
                                   'pct_despejado_medio_pixel': r4(pix)}
            print(f'  k={k} {modo:<10} fechas con imagen {fechas:2d}/27  teselas con imagen '
                  f'{100 * con_img / tot:5.1f} %  despejado medio: imagen {img:5.1f} %, pixel {pix:5.1f} %')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
