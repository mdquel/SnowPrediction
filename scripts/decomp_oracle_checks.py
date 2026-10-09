r"""
Descomposicion T1a / T1b / T2 / T3 y oraculos por fecha (sin GPU)
=================================================================

Para un conjunto de predicciones guardadas (una por semilla):
    - Descomposicion del MSE (Gupta et al. 2009) por tile, ponderada por
      pixeles, con T1 separado en T1a (nivel de la fecha) y T1b (sesgo
      entre tiles dentro de la fecha). Misma definicion que
      scripts/e2_scale_checks.py.
    - R2 acumulado sin corregir, con oraculo aditivo y con oraculo
      multiplicativo (una cifra por fecha, con los pixeles con nieve de la
      fecha).
    - Comprobacion de la identidad MSE_aditivo = MSE - T1a.

Por que la identidad es exacta: con d = o - s y delta_d = media de d en la
fecha, sum (d - delta_d)^2 = sum d^2 - N_d * delta_d^2 sobre los pixeles de
esa fecha. Sumando fechas y dividiendo por N: MSE_aditivo = MSE - T1a. Se
cumple siempre que el oraculo se calcule sobre los MISMOS pixeles que la
descomposicion y que la prediccion corregida no se recorte a 0 (aqui no se
recorta, igual que en B, C y A).

Opcion --exclude-leak: quita los tiles de test con terreno visto en train
(split espacial, hallazgo 8), como en el experimento C.

Mismos filtros que analysis/e1_error_decomposition.py (funciones
importadas): pixeles con nieve, tiles con >= 10 pixeles y varianza no nula.

No modifica nada; escribe solo el JSON de --out.

Uso
---
    python scripts/decomp_oracle_checks.py --npz <s7> <s42> <s123> \
        --label espacial --exclude-leak --out analysis/decomp_espacial.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'analysis'))
from e1_error_decomposition import (tile_mask, decompose_tile,          # noqa: E402
                                    date_from_tile_id, MIN_VALID_PIXELS)

TILE = 256


def leak_ids(georef):
    g = pd.read_csv(georef)
    H, W = int(g['row_real'].max()) + TILE, int(g['col_real'].max()) + TILE
    cov = np.zeros((H, W), bool)
    for r, c in g.loc[g['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        cov[r:r + TILE, c:c + TILE] = True
    test = g[g['exp_spatial_split'] == 'test']
    return {t for t, r, c in test[['tile_id', 'row_real', 'col_real']].itertuples(index=False)
            if cov[r:r + TILE, c:c + TILE].any()}


def analyse(path, skip=frozenset()):
    z = np.load(path)
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    rows, obs, sim, dates = [], [], [], []
    for i in range(len(ids)):
        if str(ids[i]) in skip:
            continue
        m = tile_mask(tgts[i], None, 'snow')
        if m.sum() < MIN_VALID_PIXELS:
            continue
        o, s = tgts[i][m].astype(np.float64), preds[i][m].astype(np.float64)
        dec = decompose_tile(o, s)
        if dec is None:
            continue
        d = date_from_tile_id(ids[i])
        rows.append((d, dec['n'], dec['mu_s'] - dec['mu_o'], dec['t1'], dec['t2'], dec['t3'], dec['mse']))
        obs.append(o); sim.append(s); dates.append(d)

    n = np.array([r[1] for r in rows], float); w = n / n.sum()
    b = np.array([r[2] for r in rows]); dl = np.array([r[0] for r in rows])
    T1, T2, T3, MSE = (float(np.sum(w * np.array([r[k] for r in rows]))) for k in (3, 4, 5, 6))
    b_d = {d: float(np.sum(n[dl == d] * b[dl == d]) / n[dl == d].sum()) for d in np.unique(dl)}
    bd = np.array([b_d[d] for d in dl])
    T1a, T1b = float(np.sum(w * bd ** 2)), float(np.sum(w * (b - bd) ** 2))

    O, S = np.concatenate(obs), np.concatenate(sim)
    mo = {d: np.mean(np.concatenate([o for o, x in zip(obs, dates) if x == d])) for d in b_d}
    ms = {d: np.mean(np.concatenate([s for s, x in zip(sim, dates) if x == d])) for d in b_d}
    S_add = np.concatenate([s + (mo[d] - ms[d]) for s, d in zip(sim, dates)])
    S_mul = np.concatenate([s * (mo[d] / ms[d]) for s, d in zip(sim, dates)])
    sst = np.sum((O - O.mean()) ** 2)
    r2 = lambda X: float(1 - np.sum((O - X) ** 2) / sst)
    mse_raw = float(np.mean((O - S) ** 2))
    mse_add = float(np.mean((O - S_add) ** 2))
    return {'n_tiles': len(rows), 'n_fechas': len(b_d),
            'MSE': MSE, 'T1a': T1a, 'T1b': T1b, 'T2': T2, 'T3': T3,
            'T1a_pct': 100 * T1a / MSE, 'T1b_pct': 100 * T1b / MSE,
            'T2_pct': 100 * T2 / MSE, 'T3_pct': 100 * T3 / MSE,
            'escala_T1_T2_pct': 100 * (T1a + T1b + T2) / MSE,
            'escala_T1a_T2_pct': 100 * (T1a + T2) / MSE,
            'R2_raw': r2(S), 'R2_oraculo_add': r2(S_add), 'R2_oraculo_mult': r2(S_mul),
            'identidad': {'MSE_pixeles': mse_raw, 'MSE_aditivo': mse_add,
                          'MSE_menos_T1a': mse_raw - T1a,
                          'diferencia_abs': abs(mse_add - (mse_raw - T1a)),
                          'diferencia_rel': abs(mse_add - (mse_raw - T1a)) / mse_add}}


def summ(v):
    v = np.asarray(v, float)
    return {'media': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--exclude-leak', action='store_true')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    skip = frozenset(leak_ids(args.georef)) if args.exclude_leak else frozenset()
    per = {}
    for p in args.npz:
        name = os.path.basename(p).replace('_predictions.npz', '')
        print(f'  {name}', flush=True)
        per[name] = analyse(p, skip)
    keys = ('T1a_pct', 'T1b_pct', 'T2_pct', 'T3_pct', 'escala_T1_T2_pct', 'escala_T1a_T2_pct',
            'R2_raw', 'R2_oraculo_add', 'R2_oraculo_mult')
    res = {'label': args.label, 'excluye_fuga': args.exclude_leak, 'n_excluidos_fuga': len(skip),
           'resumen': {k: summ([r[k] for r in per.values()]) for k in keys},
           'max_diferencia_rel_identidad': max(r['identidad']['diferencia_rel'] for r in per.values()),
           'por_semilla': per}
    print(f"\n{args.label}: {len(per)} ejecucion(es), tiles {[r['n_tiles'] for r in per.values()]}")
    for k in keys:
        s = res['resumen'][k]
        print(f"  {k:<20}{s['media']:>9.3f} [{s['min']:.3f}-{s['max']:.3f}]")
    print(f"  identidad MSE_aditivo = MSE - T1a: diferencia relativa maxima "
          f"{res['max_diferencia_rel_identidad']:.2e}")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'Guardado en: {args.out}')


if __name__ == '__main__':
    main()
