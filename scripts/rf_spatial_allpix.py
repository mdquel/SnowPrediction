r"""
Random Forest espacial entrenado con TODOS los pixeles validos
==============================================================

Comprobacion de justicia del experimento C (pregunta del 8-oct-2026)
--------------------------------------------------------------------
El RF de C (scripts/rf_spatial.py) se entrena solo con pixeles con nieve
(HS > 0.01 m, baselines/compute_spaef_rf_v6.py:91), mientras que la red se
entrena con todos los pixeles validos, incluido el suelo desnudo. La
evaluacion de C es la misma para los dos (solo pixeles con nieve), pero el
entrenamiento no. Este script quita esa asimetria: el RF ve tambien los
pixeles validos con HS <= 0.01 m.

Todo lo demas es IDENTICO a scripts/rf_spatial.py:
    - se importan normalize, BEST_PARAMS, MAX_PIXELS, IMGS_DIR y MASKS_DIR
      de baselines/compute_spaef_rf_v6.py (mismo preprocesado e
      hiperparametros, sin reoptimizar)
    - solo la banda de train del split espacial
    - 2 millones de pixeles, mismo tope por tile (MAX_PIXELS // n_tiles + 1),
      mismo recorte final aleatorio y mismo generador (RandomState(seed))
    - semillas 7, 42 y 123

Unico cambio: que pixeles entran al muestreo de entrenamiento.
    rf_spatial.py       : validos con HS > 0.01 m
    este script         : todos los validos (finitos y > -100), con HS
                          negativo recortado a 0 como en la red

La prediccion se hace sobre todos los pixeles validos. La evaluacion se
hace despues con scripts/spatial_models_vs_climatology.py, igual que en C,
que aplica su propia mascara (solo nieve), de modo que la comparacion con
las cifras actuales es directa.

No modifica nada existente. Escribe solo en results/rf_spatial_allpix/.

Uso
---
    python scripts/rf_spatial_allpix.py
    python scripts/rf_spatial_allpix.py --seeds 7
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import types                                                           # noqa: E402

import pandas as pd                                                    # noqa: E402


def load_splits(csv_path, source='lidar', split_type='spatial'):
    """Copia literal de data.dataset.load_splits (sin la impresion).

    data/dataset.py importa torch, que no esta en todos los equipos y no hace
    falta para el RF. Se replica aqui la funcion (filtra por fuente y reparte
    por la columna exp_<split>_split) y se registra como modulo data.dataset
    para que compute_spaef_rf_v6 se pueda importar sin torch y sin cambiarlo.
    """
    df = pd.read_csv(csv_path)
    if source != 'all':
        df = df[df['source'] == source].reset_index(drop=True)
    col = {'temporal': 'exp_temporal_split', 'spatial': 'exp_spatial_split'}[split_type]
    return tuple(df[df[col] == s].reset_index(drop=True) for s in ('train', 'val', 'test'))


_stub = types.ModuleType('data.dataset')
_stub.load_splits = load_splits
sys.modules.setdefault('data', types.ModuleType('data'))
sys.modules['data.dataset'] = _stub

from utils.metrics import compute_spaef                                # noqa: E402
from baselines.compute_spaef_rf_v6 import (normalize, BEST_PARAMS,     # noqa: E402
                                           MAX_PIXELS, IMGS_DIR, MASKS_DIR)

CSV_SPATIAL = _REPO / 'dataset_v4_ms_sx200' / 'dataset_v4_ms_sx200_spatial.csv'
OUT_ROOT = _REPO / 'results' / 'rf_spatial_allpix'
SNOW_THRESHOLD = 0.01


def load_pixels_all(df, seed, max_pixels=MAX_PIXELS):
    """Como compute_spaef_rf_v6.load_pixels, pero con TODOS los validos.

    Mismo generador, mismo tope por tile y mismo recorte final; solo cambia
    la mascara de pixeles elegibles.
    """
    rng = np.random.RandomState(seed)
    budget_per_tile = (max_pixels // len(df) + 1) if max_pixels else None

    X_list, y_list = [], []
    n_bare = n_tot = 0
    for row in df.itertuples():
        img_path = IMGS_DIR / row.tile_id
        mask_path = MASKS_DIR / row.tile_id
        if not img_path.exists() or not mask_path.exists():
            continue
        img = np.load(img_path).astype(np.float32)
        raw = np.load(mask_path).astype(np.float32)
        valid = np.isfinite(raw) & (raw > -100)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        X_tile = normalize(img)[valid.flatten()]
        y_tile = np.maximum(raw[valid], 0.0)
        if budget_per_tile and n_valid > budget_per_tile:
            idx = rng.choice(n_valid, budget_per_tile, replace=False)
            X_tile = X_tile[idx]
            y_tile = y_tile[idx]
        X_list.append(X_tile)
        y_list.append(y_tile)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    if max_pixels and X.shape[0] > max_pixels:
        idx = rng.choice(X.shape[0], max_pixels, replace=False)
        X, y = X[idx], y[idx]
    n_tot = y.size
    n_bare = int((y <= SNOW_THRESHOLD).sum())
    return X, y, n_bare, n_tot


def predict_test(rf, test_df):
    """Predice todos los pixeles validos; metricas rapidas solo sobre nieve."""
    ids, preds, tgts, vals, spaefs = [], [], [], [], []
    obs_all, sim_all = [], []
    for row in test_df.itertuples():
        img_path = IMGS_DIR / row.tile_id
        mask_path = MASKS_DIR / row.tile_id
        if not img_path.exists() or not mask_path.exists():
            continue
        img = np.load(img_path).astype(np.float32)
        raw = np.load(mask_path).astype(np.float32)
        valid = np.isfinite(raw) & (raw > -100)
        tgt = np.where(valid, np.maximum(raw, 0.0), 0.0).astype(np.float32)

        pred = np.full(tgt.shape, np.nan, dtype=np.float32)
        if valid.sum() > 0:
            X = normalize(img)[valid.flatten()]
            pred[valid] = np.maximum(rf.predict(X), 0).astype(np.float32)
            snow = valid & (tgt > SNOW_THRESHOLD)
            if snow.sum() >= 10:
                v = compute_spaef(tgt[snow], pred[snow])
                if not np.isnan(v):
                    spaefs.append(v)
                obs_all.append(tgt[snow].astype(np.float64))
                sim_all.append(pred[snow].astype(np.float64))

        ids.append(row.tile_id)
        preds.append(pred)
        tgts.append(tgt)
        vals.append(valid.astype(np.float32))

    o = np.concatenate(obs_all)
    s = np.concatenate(sim_all)
    r2 = 1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)
    return (np.array(ids), np.stack(preds), np.stack(tgts), np.stack(vals),
            float(r2), float(np.mean(spaefs)), len(spaefs))


def main():
    ap = argparse.ArgumentParser(description='RF v6 espacial con todos los pixeles validos')
    ap.add_argument('--seeds', nargs='+', type=int, default=[7, 42, 123])
    args = ap.parse_args()

    train_df, val_df, test_df = load_splits(str(CSV_SPATIAL), source='lidar',
                                            split_type='spatial')
    print(f'Split espacial: train {len(train_df)} | val {len(val_df)} (NO se usa) '
          f'| test {len(test_df)} tiles')
    print(f'Hiperparametros (RF v6, sin reoptimizar): {BEST_PARAMS}')
    print(f'Pixeles de entrenamiento: {MAX_PIXELS:,} entre TODOS los validos '
          f'(tope por tile {MAX_PIXELS // len(train_df) + 1})')

    for seed in args.seeds:
        name = f'rf_spatial_allpix_s{seed}'
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

        print('  Cargando pixeles de entrenamiento (banda de train, todos los validos)...',
              flush=True)
        X, y, n_bare, n_tot = load_pixels_all(train_df, seed)
        print(f'  {n_tot:,} pixeles, de ellos {n_bare:,} con HS <= 0.01 m '
              f'({100 * n_bare / n_tot:.1f} %)')

        print('  Entrenando...', flush=True)
        t0 = time.time()
        rf = RandomForestRegressor(**BEST_PARAMS, n_jobs=-1, random_state=seed)
        rf.fit(X, y)
        del X, y
        print(f'  Entrenamiento: {(time.time() - t0) / 60:.1f} min')

        print('  Prediciendo test...', flush=True)
        t0 = time.time()
        ids, preds, tgts, vals, r2, spaef, n_sp = predict_test(rf, test_df)
        print(f'  Prediccion: {(time.time() - t0) / 60:.1f} min')
        print(f'  Control rapido (730 tiles, solo nieve): R2 {r2:.4f}   '
              f'SPAEF {spaef:.4f} ({n_sp} tiles)')

        np.savez_compressed(out_dir / f'{name}_predictions.npz',
                            tile_ids=ids, preds=preds, targets=tgts, valids=vals)
        with open(out_dir / f'{name}_metrics.json', 'w', encoding='utf-8') as f:
            json.dump({'experiment': name, 'split': 'spatial', 'train_only': True,
                       'train_pixels': 'todos los validos', 'params': BEST_PARAMS,
                       'max_pixels': MAX_PIXELS, 'seed': seed,
                       'train_frac_bare': round(n_bare / n_tot, 4),
                       'R2_snow_730': round(r2, 4), 'SPAEF_snow_730': round(spaef, 4),
                       'SPAEF_n_tiles': n_sp}, f, indent=2)
        print(f'  Guardado en {out_dir}')
        del rf

    print('\nListo. Siguiente paso: scripts/spatial_models_vs_climatology.py con '
          '--rf results/rf_spatial_allpix/*/..._predictions.npz '
          '--out analysis/eC_net_vs_rf_allpix.json')


if __name__ == '__main__':
    main()
