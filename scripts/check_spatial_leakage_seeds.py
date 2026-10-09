r"""
Hallazgo 8 con las tres semillas: efecto de la fuga del split espacial
======================================================================

Repite la medida de scripts/check_spatial_leakage_effect.py (que solo se
hizo con la semilla 7 e imprimia por pantalla) para varias semillas y guarda
el resultado en JSON. La logica de medida es la MISMA: se importa de ese
script (new_acc, add) en lugar de copiarla.

Grupos (por tile de test del split espacial):
    todos         los 730 tiles de test
    limpios       tiles sin ningun pixel de terreno visto en train
    contaminados  tiles con terreno visto en train en otras fechas

Dominio: pixeles con nieve (HS > 0.01 m), validos y finitos; tiles con al
menos 10 pixeles. R2 acumulado sobre todos los pixeles del grupo.

No modifica nada. Escribe solo el JSON de --out.

Uso
---
    python scripts/check_spatial_leakage_seeds.py --npz <s7> <s42> <s123> \
        --out analysis/hallazgo8_fuga_espacial.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_spatial_leakage_effect import (new_acc, add, TILE,          # noqa: E402
                                          SNOW_THRESHOLD, MIN_PX)


def metrics(acc):
    if acc['n'] == 0:
        return None
    ss_tot = acc['so2'] - acc['so'] ** 2 / acc['n']
    return {'R2': round(float(1.0 - acc['sr'] / ss_tot), 4),
            'MAE': round(float(acc['sa'] / acc['n']), 4),
            'corr_media_tile': round(float(np.mean(acc['corrs'])), 4) if acc['corrs'] else None,
            'n_pixeles': int(acc['n'])}


def measure(npz_path, pos, train_cov):
    z = np.load(npz_path)
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None
    accs = {'todos': new_acc(), 'limpios': new_acc(), 'contaminados': new_acc()}
    n_t = {k: 0 for k in accs}
    fracs = []
    for i, t in enumerate(ids):
        t = str(t)
        if t not in pos.index:
            continue
        r, c = pos.loc[t]
        frac = float(train_cov[r:r + TILE, c:c + TILE].mean())
        fracs.append(frac)
        m = tgts[i] > SNOW_THRESHOLD
        if vals is not None:
            m &= vals[i] > 0.5
        m &= np.isfinite(preds[i]) & np.isfinite(tgts[i])
        if m.sum() < MIN_PX:
            continue
        o = tgts[i][m].astype(np.float64)
        p = preds[i][m].astype(np.float64)
        grupo = 'contaminados' if frac > 0 else 'limpios'
        for k in ('todos', grupo):
            add(accs[k], o, p)
            n_t[k] += 1
    fr = np.array(fracs)
    out = {k: {**(metrics(accs[k]) or {}), 'n_tiles': n_t[k]} for k in accs}
    out['_tiles_con_terreno_visto'] = int((fr > 0).sum())
    out['_fraccion_media_vista_pct'] = round(100 * float(fr.mean()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True, help='un _predictions.npz por semilla')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--out', default='analysis/hallazgo8_fuga_espacial.json')
    args = ap.parse_args()

    g = pd.read_csv(args.georef)
    H = int(g['row_real'].max()) + TILE
    W = int(g['col_real'].max()) + TILE
    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in g.loc[g['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True
    pos = g.set_index('tile_id')[['row_real', 'col_real']]

    res = {'descripcion': 'Hallazgo 8: R2 del ResUNet++ espacial (lambda=0) en test con y sin '
                          'los tiles cuyo terreno se ve en train en otras fechas. Solo pixeles '
                          'con nieve (HS > 0.01 m). Logica de scripts/check_spatial_leakage_effect.py.',
           'por_semilla': {}}
    print(f"  {'semilla':<44}{'todos':>8}{'limpios':>9}{'contam.':>9}  tiles (t/l/c)")
    for p in args.npz:
        name = os.path.basename(p).replace('_predictions.npz', '')
        r = measure(p, pos, train_cov)
        res['por_semilla'][name] = r
        print(f"  {name:<44}{r['todos']['R2']:>8.4f}{r['limpios']['R2']:>9.4f}"
              f"{r['contaminados']['R2']:>9.4f}  {r['todos']['n_tiles']}/"
              f"{r['limpios']['n_tiles']}/{r['contaminados']['n_tiles']}")

    for k in ('todos', 'limpios', 'contaminados'):
        v = [r[k]['R2'] for r in res['por_semilla'].values()]
        res[f'{k}_R2'] = {'media': round(float(np.mean(v)), 4), 'min': min(v), 'max': max(v)}
    print('\n  media [min-max]:')
    for k in ('todos', 'limpios', 'contaminados'):
        s = res[f'{k}_R2']
        print(f"    {k:<13} {s['media']:.4f} [{s['min']:.4f}-{s['max']:.4f}]")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
