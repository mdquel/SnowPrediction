r"""
Random Forest del split temporal entrenado con TODOS los pixeles validos
========================================================================

Misma comprobacion de justicia que scripts/rf_spatial_allpix.py, ahora en el
split temporal (test 2025). El RF temporal actual (results/rf_v6_s*) se
entreno solo con pixeles de nieve (HS > 0.01 m); la red, con todos los
validos.

Que se reutiliza (importado de scripts/rf_spatial_allpix.py, sin copiar)
-----------------------------------------------------------------------
    normalize, BEST_PARAMS, MAX_PIXELS     (de baselines/compute_spaef_rf_v6.py)
    load_pixels_all   2 M pixeles entre TODOS los validos, tope por tile
                      MAX_PIXELS // n_tiles + 1, recorte final, RandomState(seed)
    predict_test      R2 acumulado sobre pixeles con nieve y SPAEF medio por
                      tile (tiles con >= 10 px de nieve): la misma evaluacion
                      que los metrics.json de la red y del RF actual

Conjunto de entrenamiento (--train-set)
---------------------------------------
    train     2021-2023, igual que la red (la red usa 2024 solo para elegir
              checkpoint). Es lo que se pidio.
    trainval  2021-2024, como el RF actual ("metodologia paper"). Sirve para
              aislar el efecto del cambio de pixeles: con 'train' cambian dos
              cosas a la vez respecto al RF actual (pixeles y anyos).

Ojo: el RF actual muestreaba 2 M pixeles GLOBALES de train y anyadia TODOS
los de validacion sin submuestrear (baselines/eval_rf_v6_seeds.py:113-124).
Aqui, en 'trainval', train y validacion forman un unico conjunto con un
unico presupuesto de 2 M y el mismo tope por tile que en el espacial.

No modifica nada existente. Escribe en results/rf_temporal_allpix_<set>/ y
en analysis/rf_temporal_allpix_<set>.json.

Uso
---
    python scripts/rf_temporal_allpix.py                     (train, 3 semillas)
    python scripts/rf_temporal_allpix.py --train-set trainval
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_spatial_allpix import (load_splits, load_pixels_all, predict_test,   # noqa: E402
                               BEST_PARAMS, MAX_PIXELS)
from sklearn.ensemble import RandomForestRegressor                         # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
CSV_TEMPORAL = _REPO / 'dataset_v4_ms_sx200' / 'dataset_v4_ms_sx200.csv'

# Referencias para la comparacion (leidas de sus metrics.json en el informe)
RF_OLD = 'results/rf_v6_s{s}/rf_v6_s{s}_metrics.json'
NET_FULL = 'results/norm_v3/lambda/resunetpp_v3_sp00_s{s}/resunetpp_v3_sp00_s{s}_metrics.json'


def _metrics(path):
    d = json.load(open(_REPO / path, encoding='utf-8'))
    d = d.get('test_metrics', d)
    return float(d['R2']), float(d['SPAEF'])


def _summary(vals):
    v = np.array(vals)
    return {'media': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4)}


def main():
    ap = argparse.ArgumentParser(description='RF v6 temporal con todos los pixeles validos')
    ap.add_argument('--seeds', nargs='+', type=int, default=[7, 42, 123])
    ap.add_argument('--train-set', choices=('train', 'trainval'), default='train')
    args = ap.parse_args()

    train_df, val_df, test_df = load_splits(str(CSV_TEMPORAL), source='lidar',
                                            split_type='temporal')
    fit_df = (train_df if args.train_set == 'train'
              else pd.concat([train_df, val_df], ignore_index=True))
    tag = f'rf_temporal_allpix_{args.train_set}'
    print(f'Split temporal: train {len(train_df)} | val {len(val_df)} | test {len(test_df)} tiles')
    print(f"Entrenamiento con: {args.train_set} ({len(fit_df)} tiles, anyos "
          f"{sorted(fit_df['year'].unique().tolist())})")
    print(f'Hiperparametros (RF v6, sin reoptimizar): {BEST_PARAMS}')
    print(f'Pixeles: {MAX_PIXELS:,} entre TODOS los validos '
          f'(tope por tile {MAX_PIXELS // len(fit_df) + 1})')

    rows = {}
    for seed in args.seeds:
        name = f'{tag}_s{seed}'
        out_dir = _REPO / 'results' / tag / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

        X, y, n_bare, n_tot = load_pixels_all(fit_df, seed)
        print(f'  {n_tot:,} pixeles, de ellos {n_bare:,} con HS <= 0.01 m '
              f'({100 * n_bare / n_tot:.1f} %)')

        t0 = time.time()
        rf = RandomForestRegressor(**BEST_PARAMS, n_jobs=-1, random_state=seed)
        rf.fit(X, y)
        del X, y
        print(f'  Entrenamiento: {(time.time() - t0) / 60:.1f} min')

        ids, preds, tgts, vals, r2, spaef, n_sp = predict_test(rf, test_df)
        print(f'  Test 2025 (solo nieve): R2 {r2:.4f}   SPAEF {spaef:.4f} ({n_sp} tiles)')
        np.savez_compressed(out_dir / f'{name}_predictions.npz',
                            tile_ids=ids, preds=preds, targets=tgts, valids=vals)
        m = {'experiment': name, 'split': 'temporal', 'train_set': args.train_set,
             'train_pixels': 'todos los validos', 'params': BEST_PARAMS,
             'max_pixels': MAX_PIXELS, 'seed': seed,
             'train_frac_bare': round(n_bare / n_tot, 4),
             'R2': round(r2, 4), 'SPAEF': round(spaef, 4), 'SPAEF_n_tiles': n_sp}
        with open(out_dir / f'{name}_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2)
        rows[str(seed)] = m
        del rf

    # --- comparacion ---
    seeds = [int(s) for s in rows]
    new = ([rows[str(s)]['R2'] for s in seeds], [rows[str(s)]['SPAEF'] for s in seeds])
    old = list(zip(*[_metrics(RF_OLD.format(s=s)) for s in seeds]))
    net = list(zip(*[_metrics(NET_FULL.format(s=s)) for s in seeds]))
    res = {'descripcion': 'RF temporal entrenado con todos los pixeles validos frente al RF '
                          'actual (solo nieve, train+val) y a la red temporal completa. '
                          'Test 2025, evaluacion solo sobre nieve.',
           'train_set': args.train_set,
           'rf_nuevo': {'R2': _summary(new[0]), 'SPAEF': _summary(new[1]), 'por_semilla': rows},
           'rf_actual_solo_nieve': {'R2': _summary(old[0]), 'SPAEF': _summary(old[1]),
                                    'ruta': RF_OLD},
           'red_temporal_completa': {'R2': _summary(net[0]), 'SPAEF': _summary(net[1]),
                                     'ruta': NET_FULL}}

    print(f"\nTest 2025, media [min-max] de las semillas {seeds}")
    for k, lab in (('rf_nuevo', f'RF todos los validos ({args.train_set})'),
                   ('rf_actual_solo_nieve', 'RF actual (solo nieve, train+val)'),
                   ('red_temporal_completa', 'Red temporal completa')):
        r, s = res[k]['R2'], res[k]['SPAEF']
        print(f"  {lab:<38} R2 {r['media']:.3f} [{r['min']:.3f}-{r['max']:.3f}]   "
              f"SPAEF {s['media']:.3f} [{s['min']:.3f}-{s['max']:.3f}]")

    out = _REPO / 'analysis' / f'{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {out}')


if __name__ == '__main__':
    main()
