r"""
Efecto de la fuga del split espacial sobre las metricas
=======================================================

tile_georef.py mostro que el split espacial (bandas de columnas asignadas
por NOMBRE de tile) no es estanco: como cada vuelo tiene su propio origen,
28 de 730 tiles de test cubren terreno que en otras fechas es de
entrenamiento, un 2.6% de los pixeles de test en promedio.

Este script mide si eso infla el resultado. Compara las metricas del
modelo espacial sobre:
    todos         los 730 tiles de test
    limpios       los tiles sin ningun pixel de terreno visto en train
    contaminados  los tiles con terreno visto en train

Si "limpios" apenas difiere de "todos", la fuga no altera el resultado.
Si los contaminados salen claramente mejores que los limpios, es senal de
que la red ha memorizado esas zonas.

Uso
---
    python scripts/check_spatial_leakage_effect.py --npz <ruta al _predictions.npz>
"""

import argparse

import numpy as np
import pandas as pd

TILE = 256
SNOW_THRESHOLD = 0.01
MIN_PX = 10


def new_acc():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0, 'sa': 0.0, 'corrs': []}


def add(acc, o, p):
    acc['n'] += o.size
    acc['so'] += o.sum()
    acc['so2'] += (o ** 2).sum()
    acc['sr'] += ((o - p) ** 2).sum()
    acc['sa'] += np.abs(o - p).sum()
    if o.std() > 1e-9 and p.std() > 1e-9:
        acc['corrs'].append(float(np.corrcoef(o, p)[0, 1]))


def report(name, acc, n_tiles):
    if acc['n'] == 0:
        print(f'  {name:<14}{"(vacio)":>10}')
        return
    ss_tot = acc['so2'] - acc['so'] ** 2 / acc['n']
    r2 = 1.0 - acc['sr'] / ss_tot
    mae = acc['sa'] / acc['n']
    corr = np.mean(acc['corrs']) if acc['corrs'] else float('nan')
    print(f'  {name:<14}{n_tiles:>7}{r2:>9.4f}{mae:>9.4f}{corr:>9.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--georef', default='data/tile_georef.csv')
    args = ap.parse_args()

    g = pd.read_csv(args.georef)
    H = g['row_real'].max() + TILE
    W = g['col_real'].max() + TILE
    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in g.loc[g['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True

    pos = g.set_index('tile_id')[['row_real', 'col_real']]

    z = np.load(args.npz)
    ids = z['tile_ids']
    preds = np.asarray(z['preds'])
    tgts = np.asarray(z['targets'])
    vals = np.asarray(z['valids']) if 'valids' in z.files else None

    accs = {'todos': new_acc(), 'limpios': new_acc(), 'contaminados': new_acc()}
    n_t = {k: 0 for k in accs}
    fr_list, sin_georef = [], 0
    for i, t in enumerate(ids):
        t = str(t)
        if t not in pos.index:
            sin_georef += 1
            continue
        r, c = pos.loc[t]
        frac = float(train_cov[r:r + TILE, c:c + TILE].mean())
        fr_list.append(frac)

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

    fr = np.array(fr_list)
    print(f'Tiles en el .npz: {len(ids)}  |  con georreferencia: {len(fr)}  |  sin ella: {sin_georef}')
    print(f'Tiles con terreno visto en train: {(fr > 0).sum()}  '
          f'(fraccion media vista: {100 * fr.mean():.1f}%)')
    print()
    print(f"  {'grupo':<14}{'tiles':>7}{'R2':>9}{'MAE':>9}{'corr':>9}")
    for k in accs:
        report(k, accs[k], n_t[k])
    print()
    print('Lectura: si "limpios" apenas difiere de "todos", la fuga no altera el')
    print('resultado espacial. Si "contaminados" sale claramente mejor que')
    print('"limpios", la red podria haber memorizado esas zonas.')


if __name__ == '__main__':
    main()
