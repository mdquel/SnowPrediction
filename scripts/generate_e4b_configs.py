r"""
E4b - Generador de folds y configuraciones.
===========================================

Que es E4b
----------
La repeticion corregida de E4, no un experimento nuevo. Responde a la
misma pregunta —si una senal de escala externa mejora la extrapolacion a
anyos anomalos— pero arregla tres problemas que E4 tenia:

  1. SCALE_NORM derivaba sus limites del rango observado en todas las
     fechas, incluido el anyo de test de cada fold. Misma clase de fuga
     que se corrigio en su dia con NORM_V3_SPATIAL. Ya sustituido en
     dataset.py por limites fisicos que no dependen de los datos, lo que
     cambia los valores normalizados y obliga a reentrenar.

  2. Una sola semilla. Con la varianza observada entre semillas (0.03 a
     0.11 en R2), la mejora media de E4 (+0.048) no es distinguible de
     cero con n=1. Se repite con 7, 42 y 123.

  3. Faltaba un brazo. Entre el baseline y el dual con senales cambiaban
     DOS cosas a la vez, la arquitectura y las senales, de modo que la
     ganancia no quedaba atribuida. Es el mismo defecto que invalidaba el
     experimento meteorologico del manuscrito. Se anade el brazo
     intermedio: dual SIN senales, sobre el mismo subconjunto.

Los tres brazos
---------------
    1. baseline    resunetpp             sin senales
    2. dual        resunetpp_dual        sin senales   <- brazo nuevo
    3. dualscale   resunetpp_dual_scale  con senales

Comparando 1 con 2 se aisla el efecto de la arquitectura; comparando 2
con 3, el de las senales.

Variante sin submuestreo (solo fold 2025)
------------------------------------------
El submuestreo a un tope comun existe para comparar FOLDS entre si: en
este dataset los anyos mas anomalos son los que menos tiles aportan, de
modo que excluirlos deja mas datos de entrenamiento y anomalia y tamano
quedan confundidos.

Pero para comparar dos ARQUITECTURAS dentro de un mismo fold ese recorte
no hace falta: ambas entrenan con los mismos datos, sean los que sean.
Por eso se generan ademas configs del fold 2025 con todos los tiles
disponibles (3866 en lugar de 2284). Eso responde a la objecion de que el
baseline estaba deprimido por falta de datos, sin necesidad de imputar
senales en las fechas que faltan, que seria fabricar datos justo en la
variable que se esta probando.

Prioridad de ejecucion
----------------------
Se generan los cuatro folds, pero la primera tanda es solo el fold 2025:
es el mas anomalo, el anyo de test del manuscrito y el unico con
encadenamiento temporal hacia adelante puro. Con sus tres brazos y tres
semillas (15 entrenamientos) se responde a las tres objeciones en el fold
que sostiene el hallazgo, y luego se valora si ampliar.

Uso
---
    python scripts/generate_e4b_configs.py --dry-run
    python scripts/generate_e4b_configs.py
"""

import argparse
import os

import numpy as np
import pandas as pd

ALL_YEARS = [2021, 2022, 2023, 2024, 2025]
VAL_YEAR = 2024
TEST_YEARS = [y for y in ALL_YEARS if y != VAL_YEAR]
SEEDS = [7, 42, 123]
FULL_FOLD = 2025          # fold que ademas se genera sin submuestrear

CSV_IN = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
SIGNALS_CSV = 'data/scale_signals.csv'
CSV_OUT_DIR = 'dataset_v4_ms_sx200/loyo_e4b'
CFG_OUT_DIR = 'configs/norm_v3'

SIGNAL_COLS = ['snow_pct', 'pdd_15d', 'pdd_30d', 'ppAcc_mm']

# Los tres brazos. 'scale' indica si recibe las senales externas.
ARMS = [
    dict(key='base', arch='resunetpp', scale=False, loss='spatial_mse',
         loss_params='  lambda_pearson: 0.0\n',
         label='baseline (ResUNet++ estandar)'),
    dict(key='dual', arch='resunetpp_dual', scale=False, loss='dual_head',
         loss_params='  lambda_mu: 1.0\n  lambda_sigma: 1.0\n',
         label='doble cabeza SIN senales (brazo de atribucion)'),
    dict(key='dualscale', arch='resunetpp_dual_scale', scale=True,
         loss='dual_head',
         loss_params='  lambda_mu: 1.0\n  lambda_sigma: 1.0\n',
         label='doble cabeza CON senales de escala externas'),
]

CONFIG_TEMPLATE = """experiment:
  name: {name}
# E4b - {label}
# test = {year}, val = {val_year}, seed = {seed}.
# Solo fechas con las cuatro senales de escala ({n_dates} de 27).
# Train: {train_desc}
# Hiperparametros NO re-optimizados (control experimental).
data:
  root: dataset_v4_ms_sx200
  csv_file: loyo_e4b/{csv_name}
  images_dir: images
  masks_dir: masks
  source: lidar
  split_type: temporal
  use_sce: false
  augmentation: false
  norm_version: v3
{scale_block}model:
  architecture: {arch}
  in_channels: 22
  out_channels: 1
  features:
  - 64
  - 128
  - 256
  - 512
  dropout_p: 0.077
  num_groups: 8
training:
  seed: {seed}
  batch_size: 8
  learning_rate: 0.0001287
  epochs: 50
  loss: {loss}
{loss_params}  masked_loss: true
  weight_decay: 0.0001239
  optimizer: adamw
  grad_clip: 1.0
  early_stopping: false
  num_workers: 0
  device: auto
evaluation:
  save_predictions: true
output:
  models_dir: results/norm_v3/{outdir}/weights
  results_dir: results/norm_v3/{outdir}/{name}
  model_name: {name}
"""


def build_fold(df, test_year, train_cap, seed=0):
    """Reasigna exp_temporal_split y submuestrea el train si procede.

    Los tiles sobrantes se marcan como 'unused'; load_splits() solo lee
    'train', 'val' y 'test', asi que los ignora.

    Con train_cap=None no se submuestrea: se usan todos los tiles
    disponibles.
    """
    fold = df.copy()
    fold['exp_temporal_split'] = np.where(
        fold['year'] == test_year, 'test',
        np.where(fold['year'] == VAL_YEAR, 'val', 'train'))

    if train_cap is None:
        return fold

    idx = fold.index[fold['exp_temporal_split'] == 'train']
    if len(idx) > train_cap:
        sub = (fold.loc[idx]
               .groupby('year', group_keys=False)[fold.columns.tolist()]
               .apply(lambda g: g.sample(
                   max(1, int(round(train_cap * len(g) / len(idx)))),
                   random_state=seed)))
        drop = [i for i in idx if i not in set(sub.index)]
        fold.loc[drop, 'exp_temporal_split'] = 'unused'
    return fold


def write_configs(year, csv_name, n_dates, train_desc, suffix, dry_run):
    """Escribe las configs de los tres brazos x tres semillas."""
    written = []
    for arm in ARMS:
        for seed in SEEDS:
            name = f"e4b_{arm['key']}{suffix}_loyo{year}_s{seed}"
            outdir = f"loyo_e4b_{arm['key']}"
            scale_block = ''
            if arm['scale']:
                scale_block = ("  scale_signals_csv: data/scale_signals.csv\n"
                               f"  scale_signal_cols: {SIGNAL_COLS}\n")
            if not dry_run:
                path = os.path.join(CFG_OUT_DIR, f'{name}.yaml')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(CONFIG_TEMPLATE.format(
                        name=name, label=arm['label'], year=year,
                        val_year=VAL_YEAR, seed=seed, n_dates=n_dates,
                        train_desc=train_desc, csv_name=csv_name,
                        scale_block=scale_block, arch=arm['arch'],
                        loss=arm['loss'], loss_params=arm['loss_params'],
                        outdir=outdir))
            written.append(name)
    return written


def main():
    ap = argparse.ArgumentParser(description='Genera folds y configs de E4b')
    ap.add_argument('--dry-run', action='store_true',
                    help='muestra el reparto sin escribir ficheros')
    ap.add_argument('--signals', default=SIGNALS_CSV)
    args = ap.parse_args()

    for p in (CSV_IN, args.signals):
        if not os.path.exists(p):
            raise SystemExit(f'No se encuentra {p}. Ejecuta desde la raiz del repo.')

    df = pd.read_csv(CSV_IN)
    df = df[df['source'] == 'lidar'].reset_index(drop=True)
    n_total = len(df)

    # --- restringir a fechas con las cuatro senales ---
    sig = pd.read_csv(args.signals)
    adm = sig[sig['admisible'] == True].copy()
    keys = set(pd.to_datetime(adm['fecha']).dt.strftime('%Y%m%d'))
    df = df[df['date'].astype(str).isin(keys)].reset_index(drop=True)

    excl = sorted(set(sig['fecha']) - set(adm['fecha']))
    print(f'Fechas con las cuatro senales: {len(adm)} de {len(sig)}')
    print(f'Excluidas: {", ".join(excl)}')
    print(f'Tiles: {len(df)} de {n_total} ({100*len(df)/n_total:.1f}%)\n')

    sizes = [int(((df['year'] != y) & (df['year'] != VAL_YEAR)).sum())
             for y in TEST_YEARS]
    train_cap = min(sizes)
    n_val = int((df['year'] == VAL_YEAR).sum())
    print(f'Train disponible por fold: {dict(zip(TEST_YEARS, sizes))}')
    print(f'Tope de submuestreo      : {train_cap} tiles')
    print(f'Validacion ({VAL_YEAR})        : {n_val} tiles\n')

    if not args.dry_run:
        os.makedirs(CSV_OUT_DIR, exist_ok=True)
        os.makedirs(CFG_OUT_DIR, exist_ok=True)

    total = 0

    # ---------- bloque principal: 4 folds submuestreados ----------
    print('BLOQUE PRINCIPAL (submuestreado a %d tiles)' % train_cap)
    print(f"{'fold':<10}{'train':>7}{'val':>6}{'test':>7}{'unused':>8}   configs")
    print('-' * 62)
    for year in TEST_YEARS:
        fold = build_fold(df, year, train_cap)
        n = fold['exp_temporal_split'].value_counts()
        csv_name = f'dataset_e4b_loyo{year}.csv'
        if not args.dry_run:
            fold.to_csv(os.path.join(CSV_OUT_DIR, csv_name), index=False)
        w = write_configs(year, csv_name, len(adm),
                          f'submuestreado a {train_cap} tiles',
                          '', args.dry_run)
        total += len(w)
        print(f'e4b_loyo{year:<2}{n.get("train",0):>7}{n.get("val",0):>6}'
              f'{n.get("test",0):>7}{n.get("unused",0):>8}   {len(w)}')

    # ---------- variante sin submuestrear, solo fold 2025 ----------
    print()
    print('VARIANTE SIN SUBMUESTREAR (solo fold %d)' % FULL_FOLD)
    fold = build_fold(df, FULL_FOLD, None)
    n = fold['exp_temporal_split'].value_counts()
    csv_name = f'dataset_e4b_full_loyo{FULL_FOLD}.csv'
    if not args.dry_run:
        fold.to_csv(os.path.join(CSV_OUT_DIR, csv_name), index=False)
    w = write_configs(FULL_FOLD, csv_name, len(adm),
                      f'sin submuestrear ({n.get("train",0)} tiles)',
                      '_full', args.dry_run)
    total += len(w)
    print(f"{'fold':<10}{'train':>7}{'val':>6}{'test':>7}{'unused':>8}   configs")
    print('-' * 62)
    print(f'e4b_full{FULL_FOLD:<2}{n.get("train",0):>7}{n.get("val",0):>6}'
          f'{n.get("test",0):>7}{n.get("unused",0):>8}   {len(w)}')

    print()
    print(f'Total de configuraciones: {total}')
    print(f'  brazos: {len(ARMS)}   semillas: {SEEDS}')

    if args.dry_run:
        print('\n[dry-run] No se ha escrito ningun fichero.')
        return

    print()
    print(f'CSVs   : {CSV_OUT_DIR}/')
    print(f'Configs: {CFG_OUT_DIR}/e4b_*.yaml')
    print()
    print('PRIMERA TANDA (fold 2025, 15 entrenamientos, ~30 h):')
    print()
    print('  for s in 7 42 123; do')
    for arm in ARMS:
        print(f"    python main.py --config {CFG_OUT_DIR}/"
              f"e4b_{arm['key']}_loyo{FULL_FOLD}_s$s.yaml --mode both || break")
    for arm in (ARMS[0], ARMS[2]):
        print(f"    python main.py --config {CFG_OUT_DIR}/"
              f"e4b_{arm['key']}_full_loyo{FULL_FOLD}_s$s.yaml --mode both || break")
    print('  done')
    print()
    print('Los otros tres folds quedan generados; se lanzan si tras ver')
    print('los resultados de 2025 se decide ampliar.')


if __name__ == '__main__':
    main()
