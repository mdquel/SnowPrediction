r"""
Random Forest en el split espacial (competidor topografico de la red)
=====================================================================

Por que
-------
El experimento C mostro que, en terreno nunca sobrevolado, la red recupera
el 74.6-79.5% del techo climatologico. Pero en ese escenario la competencia
de la red no es la climatologia (que ahi no existe), sino otros modelos que
tambien parten solo de la topografia. Si el RF llega a lo mismo, ese papel
no seria especifico de la red y habria que recomendar el metodo mas simple.

Protocolo: identico al RF v6 del split temporal
-----------------------------------------------
Se IMPORTAN las funciones de baselines/compute_spaef_rf_v6.py (normalize,
load_pixels) y sus constantes (BEST_PARAMS, MAX_PIXELS), en lugar de
copiarlas, para garantizar que el preprocesado y los hiperparametros son
exactamente los mismos:
    - dataset_v4_ms_sx200, los mismos 22 canales que la red
    - hiperparametros de la optimizacion Optuna v6, sin reoptimizar
    - 2 millones de pixeles de nieve, submuestreados por tile
    - semillas 7, 42 y 123

Unica diferencia, deliberada: el RF temporal se entrena con train + val.
En el split espacial eso le daria ventaja injusta: la banda de validacion
toca a la de test y, con el desalineamiento de los vuelos, comparte el
60.5% del terreno de test en otras fechas. La red NO entrena con
validacion (solo elige checkpoint). Aqui el RF se entrena SOLO con la
banda de entrenamiento, igual que la red.

La normalizacion del RF v6 usa las constantes antiguas, pero un RF es
invariante a reescalados monotonos de las entradas (corta por umbrales),
asi que no afecta.

Salida
------
results/rf_spatial/rf_spatial_s<seed>/rf_spatial_s<seed>_predictions.npz
con el mismo formato que las predicciones de la red (tile_ids, preds,
targets, valids). Solo se predicen los pixeles con nieve y dato valido,
que son los unicos que se evaluan; el resto queda como NaN.

Uso
---
    python scripts/rf_spatial.py
    python scripts/rf_spatial.py --seeds 7
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.dataset import load_splits                                   # noqa: E402
from utils.metrics import compute_spaef                                # noqa: E402
from baselines.compute_spaef_rf_v6 import (normalize, load_pixels,     # noqa: E402
                                           BEST_PARAMS, MAX_PIXELS,
                                           IMGS_DIR, MASKS_DIR)

CSV_SPATIAL = _REPO / 'dataset_v4_ms_sx200' / 'dataset_v4_ms_sx200_spatial.csv'
OUT_ROOT = _REPO / 'results' / 'rf_spatial'
SNOW_THRESHOLD = 0.01


def predict_test(rf, test_df):
    """Predice los tiles de test en el formato de la red."""
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
        tgt = np.where(valid, raw, 0.0).astype(np.float32)
        snow = valid & (tgt > SNOW_THRESHOLD)

        pred = np.full(tgt.shape, np.nan, dtype=np.float32)
        if snow.sum() > 0:
            X = normalize(img)[snow.flatten()]
            pred[snow] = np.maximum(rf.predict(X), 0).astype(np.float32)
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
    ap = argparse.ArgumentParser(description='RF v6 en el split espacial')
    ap.add_argument('--seeds', nargs='+', type=int, default=[7, 42, 123])
    args = ap.parse_args()

    train_df, val_df, test_df = load_splits(str(CSV_SPATIAL), source='lidar',
                                            split_type='spatial')
    print(f'Split espacial: train {len(train_df)} | val {len(val_df)} (NO se usa) '
          f'| test {len(test_df)} tiles')
    print(f'Hiperparametros (RF v6, sin reoptimizar): {BEST_PARAMS}')

    for seed in args.seeds:
        name = f'rf_spatial_s{seed}'
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")

        print('  Cargando pixeles de entrenamiento (solo banda de train)...', flush=True)
        X, y = load_pixels(train_df, seed, max_pixels=MAX_PIXELS)
        print(f'  {X.shape[0]:,} pixeles')

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
        print(f'  R2 (todos los tiles de test): {r2:.4f}   SPAEF: {spaef:.4f} ({n_sp} tiles)')

        np.savez_compressed(out_dir / f'{name}_predictions.npz',
                            tile_ids=ids, preds=preds, targets=tgts, valids=vals)
        with open(out_dir / f'{name}_metrics.json', 'w', encoding='utf-8') as f:
            json.dump({'experiment': name, 'split': 'spatial', 'train_only': True,
                       'params': BEST_PARAMS, 'max_pixels': MAX_PIXELS, 'seed': seed,
                       'R2': round(r2, 4), 'SPAEF': round(spaef, 4),
                       'SPAEF_n_tiles': n_sp}, f, indent=2)
        print(f'  Guardado en {out_dir}')
        del rf

    print('\nListo.')


if __name__ == '__main__':
    main()
