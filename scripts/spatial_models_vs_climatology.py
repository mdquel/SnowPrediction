r"""
Experimento C ampliado - Red frente a RF en terreno nunca sobrevolado
=====================================================================

Pregunta (revision de Hernan)
-----------------------------
En terreno nuevo la climatologia no existe, asi que la competencia real de
la red son otros modelos que tambien parten solo de la topografia. ¿Supera
la red al Random Forest, contra el mismo techo y en las mismas condiciones?
Si el RF llega a lo mismo, el papel no es especifico de la red y se
recomienda el metodo mas simple.

Condiciones identicas para los dos modelos
------------------------------------------
* Banda de test del split espacial, sin los tiles con terreno visto en
  entrenamiento (fuga del split, 28 de 730).
* Techo: climatologia del mismo sitio con los OTROS anyos.
* Nivel perfecto: cada fecha se desplaza a su media real (oraculo), para
  medir solo la calidad del patron.
* Misma mascara de pixeles para red, RF y techo en cada tile.
* Metricas: R2 (acumulado) y SPAEF canonico (utils.metrics.compute_spaef,
  media por tile). Fracciones del techo con un decimal.

Criterio (fijado antes de ejecutar)
-----------------------------------
    la red supera al RF en R2 Y en SPAEF, y los rangos de las tres
    semillas NO se solapan en ninguna de las dos
        -> el papel en terreno nuevo es especifico de la red
    en otro caso
        -> equivalentes (o gana el RF): se recomienda el mas simple

Uso
---
    python scripts/spatial_models_vs_climatology.py ^
        --net <npz s7> <npz s42> <npz s123> ^
        --rf  <npz s7> <npz s42> <npz s123>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from utils.metrics import compute_spaef   # noqa: E402

TILE = 256
SNOW_THRESHOLD = 0.01
MIN_PX = 10
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
GEOREF_CSV = 'data/tile_georef.csv'


def new_acc():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0}


def add(acc, o, p):
    acc['n'] += o.size
    acc['so'] += o.sum()
    acc['so2'] += (o ** 2).sum()
    acc['sr'] += ((o - p) ** 2).sum()


def r2(acc):
    ss = acc['so2'] - acc['so'] ** 2 / acc['n']
    return 1.0 - acc['sr'] / ss if ss > 0 else float('nan')


def build_year_climatology(geo, masks_dir, H, W):
    years = sorted(geo['year'].unique())
    ysum = {y: np.zeros((H, W), dtype=np.float32) for y in years}
    ycnt = {y: np.zeros((H, W), dtype=np.int16) for y in years}
    for d, g in geo.groupby('date'):
        mos = np.full((H, W), np.nan, dtype=np.float32)
        for t, r, c in g[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
            path = os.path.join(masks_dir, t)
            if not os.path.exists(path):
                continue
            m = np.load(path).astype(np.float32)
            ok = np.isfinite(m) & (m > -100)
            mos[r:r + TILE, c:c + TILE][ok] = m[ok]
        v = np.isfinite(mos)
        y = int(d[:4])
        ysum[y][v] += mos[v]
        ycnt[y][v] += 1
    tsum = sum(ysum.values())
    tcnt = sum(c.astype(np.int32) for c in ycnt.values())
    return ysum, ycnt, tsum, tcnt


def evaluate(npz_path, pos, train_cov, clim_patch):
    """Devuelve metricas del modelo y del techo sobre la misma mascara."""
    z = np.load(npz_path)
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None

    items = []
    for i, t in enumerate(ids):
        t = str(t)
        if t not in pos:
            continue
        r, c = pos[t]
        if train_cov[r:r + TILE, c:c + TILE].any():
            continue                                   # fuga del split
        cp = clim_patch(int(t[:4]), r, c)
        tg, pr = tgts[i], preds[i]
        m = (tg > SNOW_THRESHOLD) & np.isfinite(tg) & np.isfinite(pr) & np.isfinite(cp)
        if vals is not None:
            m &= vals[i] > 0.5
        if m.sum() >= MIN_PX:
            items.append((i, t, t[:8], cp, m))

    # medias por fecha
    acc_d = {}
    for i, t, d, cp, m in items:
        s = acc_d.setdefault(d, [0.0, 0.0, 0.0, 0])
        s[0] += float(tgts[i][m].sum()); s[1] += float(preds[i][m].sum())
        s[2] += float(cp[m].sum()); s[3] += int(m.sum())
    real = {d: v[0] / v[3] for d, v in acc_d.items()}
    med = {'mod': {d: v[1] / v[3] for d, v in acc_d.items()},
           'clim': {d: v[2] / v[3] for d, v in acc_d.items()}}

    accs = {k: new_acc() for k in ('mod_raw', 'mod_orac', 'clim_raw', 'clim_orac')}
    sp = {k: [] for k in ('mod_raw', 'mod_orac', 'clim_orac')}
    for i, t, d, cp, m in items:
        o = tgts[i][m].astype(np.float64)
        pm = preds[i][m].astype(np.float64)
        pc = cp[m].astype(np.float64)
        pm_o = pm + (real[d] - med['mod'][d])
        pc_o = pc + (real[d] - med['clim'][d])
        add(accs['mod_raw'], o, pm); add(accs['mod_orac'], o, pm_o)
        add(accs['clim_raw'], o, pc); add(accs['clim_orac'], o, pc_o)
        for k, x in (('mod_raw', pm), ('mod_orac', pm_o), ('clim_orac', pc_o)):
            v = compute_spaef(o, x)
            if not np.isnan(v):
                sp[k].append(v)
    return {'n_tiles': len(items),
            'r2_raw': r2(accs['mod_raw']), 'r2_orac': r2(accs['mod_orac']),
            'spaef_raw': float(np.mean(sp['mod_raw'])),
            'spaef_orac': float(np.mean(sp['mod_orac'])),
            'techo_r2': r2(accs['clim_orac']),
            'techo_spaef': float(np.mean(sp['clim_orac']))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', nargs='+', required=True, help='npz de la red (una por semilla)')
    ap.add_argument('--rf', nargs='+', required=True, help='npz del RF (una por semilla)')
    ap.add_argument('--georef', default=GEOREF_CSV)
    ap.add_argument('--masks', default=MASKS_DIR)
    ap.add_argument('--out', default='analysis/eC_net_vs_rf.json')
    ap.add_argument('--exclude-dates', nargs='*', default=[],
                    help='fechas YYYYMMDD a quitar de la evaluacion y de la climatologia '
                         '(p. ej. un mapa duplicado). Los modelos no se reentrenan.')
    args = ap.parse_args()

    geo = pd.read_csv(args.georef, dtype={'date': str})
    geo['year'] = geo['date'].str[:4].astype(int)
    # El terreno visto en entrenamiento se calcula con TODOS los tiles con los
    # que se entrenaron los modelos, aunque luego se excluya alguna fecha.
    geo_train = geo.copy()
    if args.exclude_dates:
        n0 = len(geo)
        geo = geo[~geo['date'].isin(args.exclude_dates)].copy()
        print(f'Fechas excluidas: {args.exclude_dates} ({n0 - len(geo)} tiles fuera '
              f'de la evaluacion y de la climatologia)')
    H = int(geo_train['row_real'].max()) + TILE
    W = int(geo_train['col_real'].max()) + TILE
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}

    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in geo_train.loc[geo_train['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True

    print('Reconstruyendo el mapa de cada fecha en coordenadas reales...', flush=True)
    ysum, ycnt, tsum, tcnt = build_year_climatology(geo, args.masks, H, W)

    def clim_patch(year, r, c):
        s = tsum[r:r + TILE, c:c + TILE] - ysum[year][r:r + TILE, c:c + TILE]
        n = tcnt[r:r + TILE, c:c + TILE] - ycnt[year][r:r + TILE, c:c + TILE]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    res = {'red': [], 'rf': []}
    for label, paths in (('red', args.net), ('rf', args.rf)):
        for p in paths:
            print(f'  evaluando {label}: {os.path.basename(p)}', flush=True)
            e = evaluate(p, pos, train_cov, clim_patch)
            e['archivo'] = os.path.basename(p)
            e['frac_r2'] = e['r2_orac'] / e['techo_r2']
            e['frac_spaef'] = e['spaef_orac'] / e['techo_spaef']
            res[label].append(e)

    print()
    print(f"  {'modelo':<34}{'tiles':>6}{'R2':>8}{'frac R2':>9}{'SPAEF':>8}{'frac SPAEF':>12}")
    for label in ('red', 'rf'):
        for e in res[label]:
            print(f"  {label + ' ' + e['archivo'][:28]:<34}{e['n_tiles']:>6}{e['r2_orac']:>8.3f}"
                  f"{100 * e['frac_r2']:>8.1f}%{e['spaef_orac']:>8.3f}{100 * e['frac_spaef']:>11.1f}%")
    t0 = res['red'][0]
    print(f"  {'techo (climatologia)':<34}{'':>6}{t0['techo_r2']:>8.3f}{'':>9}{t0['techo_spaef']:>8.3f}")
    print('  (nivel perfecto: solo calidad del patron)')

    print()
    veredicto = True
    for met in ('frac_r2', 'frac_spaef'):
        rv = [e[met] for e in res['red']]
        fv = [e[met] for e in res['rf']]
        gana = np.mean(rv) > np.mean(fv)
        separados = min(rv) > max(fv)
        nombre = 'R2   ' if met == 'frac_r2' else 'SPAEF'
        print(f'  {nombre}: red {100 * np.mean(rv):.1f}% [{100 * min(rv):.1f}-{100 * max(rv):.1f}]'
              f'   RF {100 * np.mean(fv):.1f}% [{100 * min(fv):.1f}-{100 * max(fv):.1f}]'
              f"   -> {'red mejor, rangos separados' if separados else ('red mejor, rangos solapados' if gana else 'RF igual o mejor')}")
        veredicto &= separados

    print()
    if veredicto:
        print('  VEREDICTO: la red supera al RF en R2 y SPAEF con rangos separados.')
        print('  El papel en terreno nuevo es especifico de la red.')
    else:
        print('  VEREDICTO: no se cumple el criterio. Red y RF no se distinguen (o gana')
        print('  el RF): se recomienda el metodo mas simple.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
