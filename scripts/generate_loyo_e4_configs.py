r"""
E4 - Generador de folds y configuraciones.
==========================================

Que prepara
-----------
El experimento E4: doble cabeza con senales de ESCALA EXTERNAS.

E2 establecio que el error de extrapolacion interanual es de magnitud y
no de patron. E3 intento resolverlo separando patron y escala dentro de
la arquitectura, y fallo: la cabeza de escala tenia que estimar cuanta
nieve hay este anyo mirando la topografia, que es identica todos los
anyos. Se le pidio adivinar algo que no estaba en sus entradas.

E4 le da la senal que si la contiene. El analisis de senales midio que
la cobertura de nieve satelital correlaciona +0.678 con la magnitud real
descontada la estacionalidad, y que los grados-dia positivos alcanzan
-0.690 en anomalia intra-anual. Son complementarias: la primera
distingue anyos nevados de secos, los segundos siguen la fusion dentro
de una temporada.

Por que un generador nuevo y no el de E2
-----------------------------------------
E4 solo puede usar las fechas con las cuatro senales disponibles: 23 de
27. Eso cambia el reparto de datos, asi que los folds hay que
regenerarlos. Se deja intacto scripts/generate_loyo_configs.py para que
E2 siga siendo reproducible.

Que produce
-----------
Cuatro folds (test = 2021, 2022, 2023, 2025; validacion fija en 2024) y
OCHO configuraciones: cuatro de baseline (ResUNet++ estandar) y cuatro
de doble cabeza, sobre exactamente las mismas particiones.

Por que se reentrena tambien el baseline
-----------------------------------------
Los resultados de E2 se obtuvieron con las 27 fechas y no son
comparables con estos. Si se comparase el modelo nuevo contra aquellos
numeros, una diferencia podria deberse a las senales de escala o a que
el conjunto de datos ha cambiado, sin poder distinguirlo.

Es el mismo fallo que tenia el experimento meteorologico del manuscrito
original, donde los modelos de 26 canales usaban una arquitectura mas
pequena que los de 22: cambiaban dos cosas a la vez y la diferencia no
era atribuible.

Submuestreo del train
---------------------
Se iguala a TRAIN_CAP tiles en los cuatro folds. En este dataset los
anyos mas anomalos son tambien los mas pequenos, de modo que excluirlos
deja MAS datos de entrenamiento; sin igualar, anomalia y tamano de train
quedan confundidos.

Limitacion declarada
--------------------
El conjunto de validacion (2024) baja de 431 a 284 tiles, porque dos de
sus siete fechas quedan excluidas por huecos en la serie meteorologica.
Se acepta: 284 tiles bastan para seleccionar checkpoint, y cambiar el
anyo de validacion costaria uno de los cuatro folds.

Uso
---
    python scripts/generate_loyo_e4_configs.py --dry-run
    python scripts/generate_loyo_e4_configs.py
"""

import argparse
import os

import numpy as np
import pandas as pd

ALL_YEARS = [2021, 2022, 2023, 2024, 2025]
VAL_YEAR = 2024
TEST_YEARS = [y for y in ALL_YEARS if y != VAL_YEAR]

CSV_IN = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
SIGNALS_CSV = 'data/scale_signals.csv'
CSV_OUT_DIR = 'dataset_v4_ms_sx200/loyo_e4'
CFG_OUT_DIR = 'configs/norm_v3'

# Rangos observados de las senales admisibles, usados para normalizar.
# Se fijan aqui y no se recalculan por fold: si cada fold normalizase con
# sus propias constantes, el modelo veria escalas distintas en cada uno y
# los resultados no serian comparables entre folds.
SIGNAL_COLS = ['snow_pct', 'pdd_15d', 'pdd_30d', 'ppAcc_mm']

CONFIG_TEMPLATE = """experiment:
  name: {name}
# E4 - {arch_label}, test = {year}, val = {val_year}.
# Solo fechas con las cuatro senales de escala disponibles ({n_dates} de 27).
# Train submuestreado a {train_cap} tiles en los cuatro folds.
# Hiperparametros NO re-optimizados (control experimental).
data:
  root: dataset_v4_ms_sx200
  csv_file: loyo_e4/dataset_e4_loyo{year}.csv
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
  seed: 7
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
    """Reasigna exp_temporal_split y submuestrea el train.

    Los tiles sobrantes se marcan como 'unused'; load_splits() solo lee
    'train', 'val' y 'test', asi que los ignora.
    """
    fold = df.copy()
    fold['exp_temporal_split'] = np.where(
        fold['year'] == test_year, 'test',
        np.where(fold['year'] == VAL_YEAR, 'val', 'train'))

    idx = fold.index[fold['exp_temporal_split'] == 'train']
    if train_cap is not None and len(idx) > train_cap:
        sub = (fold.loc[idx]
               .groupby('year', group_keys=False)[fold.columns.tolist()]
               .apply(lambda g: g.sample(
                   max(1, int(round(train_cap * len(g) / len(idx)))),
                   random_state=seed)))
        drop = [i for i in idx if i not in set(sub.index)]
        fold.loc[drop, 'exp_temporal_split'] = 'unused'
    return fold


def main():
    ap = argparse.ArgumentParser(description='Genera folds y configs de E4')
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
    print(f'Tiles: {len(df)} de {n_total} '
          f'({100*len(df)/n_total:.1f}% del dataset)\n')

    # --- tope de submuestreo: minimo train entre folds ---
    sizes = [int(((df['year'] != y) & (df['year'] != VAL_YEAR)).sum())
             for y in TEST_YEARS]
    train_cap = min(sizes)
    n_val = int((df['year'] == VAL_YEAR).sum())
    print(f'Train disponible por fold: {sizes}')
    print(f'Tope de submuestreo      : {train_cap} tiles')
    print(f'Validacion ({VAL_YEAR})        : {n_val} tiles\n')

    if not args.dry_run:
        os.makedirs(CSV_OUT_DIR, exist_ok=True)
        os.makedirs(CFG_OUT_DIR, exist_ok=True)

    print(f"{'fold':<10}{'train':>7}{'val':>6}{'test':>7}{'unused':>8}   anyos de train")
    print('-' * 66)
    for year in TEST_YEARS:
        fold = build_fold(df, year, train_cap)
        n = fold['exp_temporal_split'].value_counts()
        tr_years = sorted(int(y) for y in
                          fold.loc[fold['exp_temporal_split'] == 'train', 'year'].unique())
        print(f'e4_loyo{year:<3}{n.get("train",0):>7}{n.get("val",0):>6}'
              f'{n.get("test",0):>7}{n.get("unused",0):>8}   {tr_years}')

        if args.dry_run:
            continue

        fold.to_csv(os.path.join(CSV_OUT_DIR, f'dataset_e4_loyo{year}.csv'), index=False)

        # --- bloque de senales de escala, solo para el modelo dual ---
        scale_block = (
            "  scale_signals_csv: data/scale_signals.csv\n"
            f"  scale_signal_cols: {SIGNAL_COLS}\n"
        )

        variants = [
            dict(arch='resunetpp', arch_label='baseline (ResUNet++ estandar)',
                 name=f'resunetpp_e4_loyo{year}', outdir='loyo_e4_base',
                 loss='spatial_mse', loss_params='  lambda_pearson: 0.0\n',
                 scale_block=''),
            dict(arch='resunetpp_dual_scale',
                 arch_label='doble cabeza con senales de escala externas',
                 name=f'resunetpp_dualscale_e4_loyo{year}', outdir='loyo_e4_dual',
                 loss='dual_head',
                 loss_params='  lambda_mu: 1.0\n  lambda_sigma: 1.0\n',
                 scale_block=scale_block),
        ]
        for v in variants:
            path = os.path.join(CFG_OUT_DIR, f"{v['name']}.yaml")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(CONFIG_TEMPLATE.format(
                    year=year, val_year=VAL_YEAR, n_dates=len(adm),
                    train_cap=train_cap, **v))

    print()
    if args.dry_run:
        print('[dry-run] No se ha escrito ningun fichero.')
        return

    print(f'CSVs   : {CSV_OUT_DIR}/dataset_e4_loyo<YYYY>.csv')
    print(f'Configs: {CFG_OUT_DIR}/resunetpp_e4_loyo<YYYY>.yaml         (baseline)')
    print(f'         {CFG_OUT_DIR}/resunetpp_dualscale_e4_loyo<YYYY>.yaml (dual)')
    print()
    print('Para lanzar los ocho entrenamientos:')
    print('  for y in 2025 2021 2022 2023; do')
    print('    python main.py --config configs/norm_v3/resunetpp_e4_loyo$y.yaml --mode both || break')
    print('    python main.py --config configs/norm_v3/resunetpp_dualscale_e4_loyo$y.yaml --mode both || break')
    print('  done')
    print()
    print('NOTA: la arquitectura resunetpp_dual_scale y la lectura de')
    print('scale_signals_csv en el dataset todavia no estan implementadas.')
    print('Las configs quedan listas para cuando lo esten.')


if __name__ == '__main__':
    main()