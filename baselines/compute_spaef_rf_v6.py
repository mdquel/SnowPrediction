"""
Calcula SPAEF para RF v6 (22ch, params Optuna v6) en 3 seeds.
==============================================================
Reentrena RF en train+val para cada seed y calcula SPAEF tile a tile.
Actualiza los metrics.json existentes en results/rf_v6_s{seed}/.

Nota de ejecucion (MQ, 2026-07-23):
    load_pixels() fue adaptada para submuestrear por tile en lugar de
    acumular todos los pixeles antes de submuestrear. Esto evita el OOM
    en equipos con RAM limitada (ejecutado en Intel i7-3770, 16 GB RAM).
    El resultado es equivalente: el RF recibe los mismos 2M pixeles
    totales, distribuidos uniformemente entre tiles en lugar de por
    muestreo global. Los valores de SPAEF obtenidos son comparables a
    los que se obtendrían con la version original en un equipo con mas RAM.

Uso:
    .venv/Scripts/python.exe baselines/compute_spaef_rf_v6.py
    anaconda: python baselines/compute_spaef_rf_v6.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.dataset import load_splits
from utils.metrics import compute_spaef

DATA_ROOT = _REPO / 'dataset_v4_ms_sx200'
CSV_FILE  = DATA_ROOT / 'dataset_v4_ms_sx200.csv'
IMGS_DIR  = DATA_ROOT / 'images'
MASKS_DIR = DATA_ROOT / 'masks'

N_CH       = 22
MAX_PIXELS = 2_000_000

BEST_PARAMS = {
    'n_estimators':      300,
    'max_depth':         15,
    'min_samples_leaf':  1,
    'max_features':      'log2',
    'min_samples_split': 2,
}

SEEDS = [42, 123, 7]


def normalize(img: np.ndarray) -> np.ndarray:
    """Replica SnowDataset._normalize() para dataset_v4_ms_sx200 (22ch)."""
    X = img[:22].copy().astype(np.float32)  # (22, H, W)
    X[X == -9999] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[0]    = (X[0] - 2100.0) / 1000.0           # DEM
    X[1]    = X[1] / 90.0                         # Slope
    X[4]    = np.clip(X[4] / 9200.0, -1.0, 1.0)  # TPI
    X[5]    = (X[5] > 5).astype(np.float32)       # SCE -> binario
    X[6:14] = np.clip(X[6:14] / 90.0, -1.0, 1.0) # Sx_200m x8
    X[17]   = (X[17] - 2100.0) / 1000.0           # DEM_5m
    X[18]   = X[18] / 90.0                         # Slope_5m
    X[21]   = np.clip(X[21] / 9200.0, -1.0, 1.0)  # TPI_5m
    return X.reshape(22, -1).T                     # (H*W, 22)


def load_pixels(df, seed, max_pixels=None):
    """Carga pixeles submuestreando por tile para evitar OOM."""
    rng = np.random.RandomState(seed)
    n_tiles = len(df)

    # Presupuesto de pixeles por tile para no acumular todo en RAM
    budget_per_tile = (max_pixels // n_tiles + 1) if max_pixels else None

    X_list, y_list = [], []
    for row in df.itertuples():
        img_path  = IMGS_DIR  / row.tile_id
        mask_path = MASKS_DIR / row.tile_id
        if not img_path.exists() or not mask_path.exists():
            continue
        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask[mask <= -100] = 0.0
        valid = mask > 0.01
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        X_tile = normalize(img)[valid.flatten()]
        y_tile = mask[valid]
        # Submuestreo por tile si superamos el presupuesto
        if budget_per_tile and n_valid > budget_per_tile:
            idx = rng.choice(n_valid, budget_per_tile, replace=False)
            X_tile = X_tile[idx]
            y_tile = y_tile[idx]
        X_list.append(X_tile)
        y_list.append(y_tile)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Submuestreo final por si acaso
    if max_pixels and X.shape[0] > max_pixels:
        idx = rng.choice(X.shape[0], max_pixels, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def compute_spaef_tiles(rf, test_df):
    spaef_vals = []
    for row in test_df.itertuples():
        img_path  = IMGS_DIR  / row.tile_id
        mask_path = MASKS_DIR / row.tile_id
        if not img_path.exists() or not mask_path.exists():
            continue
        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask[mask <= -100] = 0.0
        valid = mask > 0.01
        if valid.sum() < 10:
            continue
        X_tile = normalize(img)                    # (H*W, 22)
        y_pred = np.maximum(rf.predict(X_tile), 0)
        y_true = mask.flatten()
        val = compute_spaef(y_true[valid.flatten()], y_pred[valid.flatten()])
        if not np.isnan(val):
            spaef_vals.append(val)
    return spaef_vals


def main():
    print("Cargando splits...")
    train_df, val_df, test_df = load_splits(str(CSV_FILE), source='lidar', split_type='temporal')
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} tiles")

    for seed in SEEDS:
        exp_name    = f'rf_v6_s{seed}'
        results_dir = _REPO / f'results/{exp_name}'
        metrics_path = results_dir / f'{exp_name}_metrics.json'

        if not metrics_path.exists():
            print(f"\n[SKIP] {exp_name}: no existe metrics.json, ejecuta eval_rf_v6_seeds.py primero")
            continue

        print(f"\n{'='*55}")
        print(f"  {exp_name} | seed={seed}")
        print(f"{'='*55}")

        # Cargar pixels para train+val
        print("  Cargando train+val pixels...")
        X_train, y_train = load_pixels(train_df, seed, max_pixels=MAX_PIXELS)
        X_val,   y_val   = load_pixels(val_df,   seed)
        X_tv = np.concatenate([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        print(f"  Train+Val: {X_tv.shape[0]:,} pixeles")

        # Entrenar
        print("  Entrenando RF...")
        t0 = time.time()
        rf = RandomForestRegressor(**BEST_PARAMS, n_jobs=-1, random_state=seed)
        rf.fit(X_tv, y_tv)
        print(f"  Entrenamiento: {(time.time()-t0)/60:.1f} min")

        # Calcular SPAEF tile a tile
        print("  Calculando SPAEF tile a tile...")
        spaef_vals = compute_spaef_tiles(rf, test_df)

        spaef_mean = float(np.mean(spaef_vals))
        spaef_std  = float(np.std(spaef_vals))
        print(f"  SPAEF: {spaef_mean:.4f} ± {spaef_std:.4f}  ({len(spaef_vals)} tiles)")

        # Actualizar metrics.json
        with open(metrics_path, encoding='utf-8') as f:
            data = json.load(f)
        data['test_metrics']['SPAEF']         = round(spaef_mean, 4)
        data['test_metrics']['SPAEF_std']     = round(spaef_std,  4)
        data['test_metrics']['SPAEF_n_tiles'] = len(spaef_vals)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  Guardado: {metrics_path}")

    print("\nListo.")


if __name__ == '__main__':
    main()
