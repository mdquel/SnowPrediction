r"""
Dias despejados en las ventanas de persistencia (sin GPU, solo lee los SCE)
============================================================================

Pregunta: si la persistencia se recodifica como fraccion de nieve SOLO en los
dias en que el satelite vio el suelo,

    persistencia_nueva = N_nieve / (N_nieve + N_sin_nieve)

hay que saber cuantas veces la ventana no tiene ningun dia despejado, porque
ahi la fraccion no esta definida.

La persistencia actual (data/generate_dataset_v4_ms_sx200.py, l. 218-256) es
la media de (SCE == 11) sobre todas las imagenes de la ventana [vuelo - W,
vuelo), con W = 15, 30, 60 dias: la nube (0) y el sin nieve (10) cuentan
igual. Aqui se reproduce esa misma seleccion de imagenes y, por pixel y a la
resolucion nativa del producto (10 m):

    N_img     imagenes en la ventana
    N_desp    imagenes en que el pixel esta despejado (codigo 10 u 11)
    N_nieve   imagenes en que el pixel tiene nieve (codigo 11)

Dominio: pixeles despejados al menos una vez en todo el archivo leido. El
resto del rectangulo (fuera de la cuenca) vale 0 en todas las imagenes y no
se cuenta.

Por fecha de vuelo y ventana se guarda: N_img; fraccion del dominio con
N_desp = 0 (persistencia nueva indefinida); percentiles de N_desp; media de
la persistencia actual y de la nueva (donde esta definida) para medir el
sesgo de la nube.

Codigos: 11 nieve, 10 sin nieve, 0 sin observacion (nube o fuera de cuenca).
Desde 2021-02-17 los ficheros son float32 con ruido numerico de ~1e-17 en
lugar de 0: se clasifican redondeando al entero mas cercano.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/check_persistence_clear_days.py
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

WINDOWS = (15, 30, 60)        # como PERSIST_WINDOWS del generador


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def flight_dates(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        return sorted({row['date'] for row in csv.DictReader(f)})


def sce_index(sce_dir):
    """Mismo indice que build_sce_index(): solo izas_YYYYMMDD.tif."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(sce_dir, 'izas_*.tif'))):
        m = re.fullmatch(r'izas_(\d{8})\.tif', os.path.basename(p))
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = p
    return idx


def read_codes(path):
    """Codigos enteros 0/10/11 (los float32 traen ruido de ~1e-17)."""
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    return np.rint(a).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description='Dias despejados en las ventanas de persistencia')
    ap.add_argument('--sce-dir', default='Articulo 1/Data/izas/LiDAR/images_SCE')
    ap.add_argument('--csv', default='dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv')
    ap.add_argument('--out', default='analysis/persistencia_dias_despejados.json')
    args = ap.parse_args()

    dates = [datetime.strptime(d, '%Y%m%d') for d in flight_dates(args.csv)]
    idx = sce_index(args.sce_dir)
    need = sorted({dt for d in dates for dt in idx
                   if d - timedelta(days=max(WINDOWS)) <= dt < d})
    codes = {dt: read_codes(idx[dt]) for dt in need}
    print(f'{len(dates)} fechas de vuelo; {len(need)} imagenes SCE en sus ventanas')

    otros = sorted({int(v) for c in codes.values() for v in np.unique(c)} - {0, 10, 11})
    if otros:
        raise SystemExit(f'STOP: codigos no esperados {otros}')
    domain = np.zeros(next(iter(codes.values())).shape, bool)
    for c in codes.values():
        domain |= (c == 10) | (c == 11)
    nd = int(domain.sum())
    print(f'dominio: {nd} pixeles de 10 m despejados alguna vez ({100 * nd / domain.size:.1f} % del rectangulo)')

    out = {'descripcion': 'Dias despejados en las ventanas de persistencia por fecha de vuelo. '
                          'Persistencia actual = media de (SCE==11) sobre las imagenes de la '
                          'ventana; nueva = N_nieve / N_despejado. 6 decimales.',
           'fuente': {'sce': args.sce_dir, 'fechas': args.csv,
                      'calculo_actual': 'data/generate_dataset_v4_ms_sx200.py, compute_persistence()'},
           'dominio_pixeles_10m': nd,
           'ventanas_dias': list(WINDOWS),
           'por_fecha': {}}

    print(f"\n{'fecha':<10}{'W':>4}{'N_img':>7}{'sin despejado':>15}{'N_desp p10/p50/p90':>21}"
          f"{'pers. actual':>14}{'pers. nueva':>13}")
    for d in dates:
        key = d.strftime('%Y%m%d')
        out['por_fecha'][key] = {}
        for w in WINDOWS:
            sel = [codes[dt] for dt in need if d - timedelta(days=w) <= dt < d]
            n_img = len(sel)
            if n_img == 0:
                rec = {'N_imagenes': 0, 'frac_dominio_sin_despejado': 1.0,
                       'N_despejado_p10_p50_p90': [0, 0, 0],
                       'persistencia_actual_media': 0.0, 'persistencia_nueva_media': None,
                       'nota': 'ninguna imagen en la ventana: el generador pone la persistencia a 0'}
            else:
                st = np.stack(sel)
                n_desp = ((st == 10) | (st == 11)).sum(0)[domain]
                n_nieve = (st == 11).sum(0)[domain]
                actual = n_nieve / n_img
                ok = n_desp > 0
                nueva = n_nieve[ok] / n_desp[ok]
                rec = {'N_imagenes': n_img,
                       'frac_dominio_sin_despejado': r6(1 - ok.mean()),
                       'N_despejado_p10_p50_p90': [int(np.percentile(n_desp, q)) for q in (10, 50, 90)],
                       'frac_imagenes_despejadas_media': r6((n_desp / n_img).mean()),
                       'persistencia_actual_media': r6(actual.mean()),
                       'persistencia_nueva_media': r6(nueva.mean()) if ok.any() else None,
                       'persistencia_actual_media_donde_nueva_definida': r6(actual[ok].mean()) if ok.any() else None}
            out['por_fecha'][key][f'{w}d'] = rec
            pn = rec['persistencia_nueva_media']
            print(f"{key:<10}{w:>4}{n_img:>7}{100 * rec['frac_dominio_sin_despejado']:>14.1f}%"
                  f"{'/'.join(map(str, rec['N_despejado_p10_p50_p90'])):>21}"
                  f"{rec['persistencia_actual_media']:>14.3f}{'   -' if pn is None else f'{pn:>13.3f}'}")

    # resumen: cuantas combinaciones fecha-ventana tienen pixeles indefinidos
    res = {}
    for w in WINDOWS:
        f = [out['por_fecha'][k][f'{w}d']['frac_dominio_sin_despejado'] for k in out['por_fecha']]
        res[f'{w}d'] = {'fechas_con_algun_pixel_indefinido': int(sum(x > 0 for x in f)),
                        'fechas_con_mas_del_10pct_indefinido': int(sum(x > 0.10 for x in f)),
                        'fechas_totalmente_indefinidas': int(sum(x >= 1.0 for x in f)),
                        'fechas_sin_imagenes': [k for k in out['por_fecha']
                                                if out['por_fecha'][k][f'{w}d']['N_imagenes'] == 0]}
    out['resumen'] = res
    print('\nResumen por ventana:')
    for w, r in res.items():
        print(f"  {w}: {r['fechas_con_algun_pixel_indefinido']} fechas con algun pixel indefinido, "
              f"{r['fechas_con_mas_del_10pct_indefinido']} con mas del 10 %, "
              f"{r['fechas_totalmente_indefinidas']} totalmente indefinidas; sin imagenes: {r['fechas_sin_imagenes']}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
