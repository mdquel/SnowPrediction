r"""
E2 - Generador de folds Leave-One-Year-Out (LOYO).

Objetivo cientifico
-------------------
Contrastar si el PATRON de acumulacion de nieve es estable entre anyos.

E1 mostro que la correlacion media por tile es menor al extrapolar en el
tiempo (0.564) que en el espacio (0.639). Si el patron fuese una propiedad
fija del terreno, ambas deberian ser similares. E2 pone a prueba esa
observacion con cinco puntos en lugar de uno.

Diseno
------
Cinco entrenamientos identicos salvo por que anyo se excluye:

    fold        test    val     train
    loyo2021    2021    2024    2022, 2023, 2025
    loyo2022    2022    2024    2021, 2023, 2025
    loyo2023    2023    2024    2021, 2022, 2025
    loyo2025    2025    2024    2021, 2022, 2023

El anyo de VALIDACION es FIJO (2024) en los cuatro folds. Esto es
deliberado: si la validacion cambiase de fold a fold, el tamano del train
variaria por dos motivos a la vez (que anyo se excluye y que anyo valida),
y no seria posible atribuir la degradacion del rendimiento a la anomalia
del anyo de test. Con validacion fija, la unica variable que cambia entre
folds es el anyo excluido.

Se elige 2024 por ser el anyo mas pequeno (431 tiles). El coste es que
2024 nunca actua como test, de modo que hay cuatro folds en lugar de cinco.
Las anomalias de HS cubiertas siguen siendo amplias (de -37% a +48%).

Submuestreo del train (por defecto activo)
------------------------------------------
En este dataset los anyos mas anomalos son tambien los mas pequenos, de modo
que excluirlos deja MAS tiles para entrenar. Sin corregirlo, |anomalia| y
tamano de train correlacionan a r = +0.90 entre folds: cualquier degradacion
del rendimiento seria atribuible a ambas causas por igual.

Para evitarlo, el train se submuestrea al minimo comun entre folds,
estratificando por anyo. Los tiles descartados se marcan como 'unused' y
load_splits() los ignora. Desactivable con --no-balance-train, no recomendado.

Que produce
-----------
Por cada fold:
  * dataset_v4_ms_sx200/loyo/dataset_loyo<YYYY>.csv
      copia del CSV original con la columna exp_temporal_split reasignada
  * configs/norm_v3/resunetpp_v3_loyo<YYYY>.yaml
      config identica a la del split temporal lambda=0, apuntando a ese CSV

Ademas imprime una tabla con el tamano de cada split y la anomalia de HS
del anyo de test, que es la variable explicativa del analisis posterior.

Limitaciones conocidas (declararlas en el paper)
------------------------------------------------
* Los anyos tienen tamanos muy distintos (2022: 1802 tiles; 2025: 220).
  Con validacion fija el train varia solo por el anyo excluido, pero sigue
  sin ser identico entre folds, y el numero de tiles de TEST varia mucho
  (220-1802). Con cuatro folds no es posible separar por completo el efecto
  de la anomalia del efecto del tamano de muestra. La correlacion observada
  entre HS medio y numero de tiles es debil (Spearman -0.20), lo que mitiga
  pero no elimina la confusion.
* Para que la metrica de respuesta sea comparable entre folds, en el
  analisis conviene submuestrear el test a un numero fijo de tiles
  (p.ej. 220, el minimo). Eso se hace despues, sobre las predicciones
  guardadas, sin coste computacional adicional.
* Los hiperparametros provienen de una optimizacion sobre el split
  temporal original y no se re-optimizan por fold, para que la unica
  variable que cambie sea el anyo excluido.

Uso
---
    python scripts/generate_loyo_configs.py
    python scripts/generate_loyo_configs.py --dry-run   # solo la tabla
"""

import argparse
import os

import numpy as np
import pandas as pd

ALL_YEARS = [2021, 2022, 2023, 2024, 2025]

# Anyo de validacion FIJO en todos los folds. Se elige 2024 por ser el mas
# pequeno (431 tiles), de modo que el train varie lo menos posible entre
# folds. Coste: 2024 no puede ser anyo de test -> cuatro folds, no cinco.
VAL_YEAR = 2024
TEST_YEARS = [y for y in ALL_YEARS if y != VAL_YEAR]

CSV_IN = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
CSV_OUT_DIR = 'dataset_v4_ms_sx200/loyo'
CFG_OUT_DIR = 'configs/norm_v3'

SNOW_THRESHOLD = 0.01     # m, mismo criterio que el pipeline
HS_SAMPLE_TILES = 150     # tiles por anyo para estimar HS medio

CONFIG_TEMPLATE = """experiment:
  name: resunetpp_v3_loyo{year}
# E2 - Leave-One-Year-Out: test = {year}, val = {val_year}, train = {train_years}.
# Identico a resunetpp_v3_sp00_s7.yaml salvo el CSV de splits.
# Hiperparametros NO re-optimizados por fold (control experimental).
data:
  root: dataset_v4_ms_sx200
  csv_file: loyo/dataset_loyo{year}.csv
  images_dir: images
  masks_dir: masks
  source: lidar
  split_type: temporal
  use_sce: false
  augmentation: false
  norm_version: v3
model:
  architecture: resunetpp
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
  loss: spatial_mse
  lambda_pearson: 0.0
  masked_loss: true
  weight_decay: 0.0001239
  optimizer: adamw
  grad_clip: 1.0
  early_stopping: false
  num_workers: 0
  device: auto
evaluation:
  save_predictions: true
output:
  models_dir: results/norm_v3/loyo/weights
  results_dir: results/norm_v3/loyo/resunetpp_v3_loyo{year}
  model_name: resunetpp_v3_loyo{year}
"""


def mean_hs_by_year(df, masks_dir, n_sample=HS_SAMPLE_TILES, seed=0):
    """HS media por anyo, estimada sobre una muestra de tiles."""
    out = {}
    for year, g in df.groupby('year'):
        vals = []
        sample = g['tile_id'].sample(min(n_sample, len(g)), random_state=seed)
        for tile in sample:
            path = os.path.join(masks_dir, tile)
            if not os.path.exists(path):
                continue
            m = np.load(path).ravel()
            m = m[np.isfinite(m) & (m > SNOW_THRESHOLD)]
            if m.size:
                vals.append(float(m.mean()))
        out[int(year)] = float(np.mean(vals)) if vals else float('nan')
    return out


def build_fold(df, test_year, train_cap=None, seed=0):
    """Devuelve una copia del df con exp_temporal_split reasignado.

    Si train_cap se especifica, el conjunto de entrenamiento se submuestrea
    a ese numero de tiles, estratificando por anyo para conservar la
    proporcion relativa de cada anyo de train. Los tiles sobrantes se
    marcan como 'unused' y load_splits() los ignora (solo lee 'train',
    'val' y 'test').

    El submuestreo existe para que el tamano del train sea identico en
    todos los folds. Sin el, en este dataset los anyos mas anomalos son
    tambien los mas pequenos, de modo que excluirlos deja MAS datos de
    entrenamiento: anomalia y tamano de train quedan confundidos y la
    degradacion del rendimiento no puede atribuirse a una sola causa.
    """
    val_year = VAL_YEAR
    fold = df.copy()
    fold['exp_temporal_split'] = np.where(
        fold['year'] == test_year, 'test',
        np.where(fold['year'] == val_year, 'val', 'train'))

    if train_cap is not None:
        train_idx = fold.index[fold['exp_temporal_split'] == 'train']
        if len(train_idx) > train_cap:
            sub = (fold.loc[train_idx]
                   .groupby('year', group_keys=False)[fold.columns.tolist()]
                   .apply(lambda g: g.sample(
                       max(1, int(round(train_cap * len(g) / len(train_idx)))),
                       random_state=seed)))
            keep = set(sub.index)
            drop = [i for i in train_idx if i not in keep]
            fold.loc[drop, 'exp_temporal_split'] = 'unused'
    return fold, val_year


def main():
    ap = argparse.ArgumentParser(description='Genera folds LOYO (E2)')
    ap.add_argument('--dry-run', action='store_true',
                    help='muestra la tabla sin escribir ficheros')
    ap.add_argument('--no-balance-train', dest='balance_train',
                    action='store_false',
                    help='NO igualar el tamano del train entre folds '
                         '(no recomendado: confunde anomalia y tamano)')
    ap.set_defaults(balance_train=True)
    args = ap.parse_args()

    if not os.path.exists(CSV_IN):
        raise SystemExit(f'No se encuentra {CSV_IN}. Ejecuta desde la raiz del repo.')

    df = pd.read_csv(CSV_IN)
    df = df[df['source'] == 'lidar'].reset_index(drop=True)

    print('Calculando HS media por anyo (muestra de tiles)...')
    hs = mean_hs_by_year(df, MASKS_DIR)

    # Cap del train = minimo entre folds, para igualar tamanos.
    if args.balance_train:
        sizes = []
        for y in TEST_YEARS:
            sizes.append(int(((df['year'] != y) & (df['year'] != VAL_YEAR)).sum()))
        train_cap = min(sizes)
        print(f'Submuestreo de train activado: {train_cap} tiles en todos los folds')
    else:
        train_cap = None
        print('Submuestreo de train DESACTIVADO (--no-balance-train)')

    if not args.dry_run:
        os.makedirs(CSV_OUT_DIR, exist_ok=True)
        os.makedirs(CFG_OUT_DIR, exist_ok=True)

    print()
    print(f"{'fold':<10} {'val':>5} {'train':>6} {'val_n':>6} {'test_n':>7} "
          f"{'HS_test':>8} {'HS_train':>9} {'anomalia':>9}")
    print('-' * 72)

    rows = []
    for year in TEST_YEARS:
        fold, val_year = build_fold(df, year, train_cap=train_cap)
        train_years = [y for y in ALL_YEARS if y not in (year, val_year)]

        n_train = int((fold['exp_temporal_split'] == 'train').sum())
        n_val = int((fold['exp_temporal_split'] == 'val').sum())
        n_test = int((fold['exp_temporal_split'] == 'test').sum())

        # anomalia del anyo de test respecto a la media de los anyos de train
        hs_train = float(np.mean([hs[y] for y in train_years]))
        anom = (hs[year] - hs_train) / hs_train

        print(f'loyo{year:<6} {val_year:>5} {n_train:>6} {n_val:>6} {n_test:>7} '
              f'{hs[year]:>8.3f} {hs_train:>9.3f} {anom:>+8.1%}')

        rows.append({
            'fold': f'loyo{year}', 'test_year': year, 'val_year': val_year,
            'train_years': train_years, 'n_train': n_train, 'n_val': n_val,
            'n_test': n_test, 'hs_test': hs[year], 'hs_train': hs_train,
            'anomaly': anom, 'abs_anomaly': abs(anom),
        })

        if not args.dry_run:
            csv_path = os.path.join(CSV_OUT_DIR, f'dataset_loyo{year}.csv')
            fold.to_csv(csv_path, index=False)

            cfg_path = os.path.join(CFG_OUT_DIR, f'resunetpp_v3_loyo{year}.yaml')
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write(CONFIG_TEMPLATE.format(
                    year=year, val_year=val_year,
                    train_years=', '.join(str(y) for y in train_years)))

    # aviso sobre la confusion tamano/anomalia
    abs_anom = [r['abs_anomaly'] for r in rows]
    n_train = [r['n_train'] for r in rows]
    if len(set(n_train)) > 1:
        rho = float(np.corrcoef(abs_anom, n_train)[0, 1])
        print()
        print(f'Correlacion |anomalia| vs n_train : r = {rho:+.3f}')
        if abs(rho) > 0.7:
            print('  AVISO: anomalia y tamano de train estan fuertemente asociados.')
            print('  No sera posible atribuir la degradacion a una sola causa.')
        else:
            print('  Asociacion debil: la anomalia y el tamano de train son')
            print('  razonablemente independientes entre folds.')

    if args.dry_run:
        print('\n[dry-run] No se ha escrito ningun fichero.')
        return

    print()
    print(f'CSVs escritos en  : {CSV_OUT_DIR}/')
    print(f'Configs escritas  : {CFG_OUT_DIR}/resunetpp_v3_loyo<YYYY>.yaml')
    print()
    print('Para lanzar los cinco folds:')
    for year in TEST_YEARS:
        print(f'  python main.py --config {CFG_OUT_DIR}/resunetpp_v3_loyo{year}.yaml --mode both')


if __name__ == '__main__':
    main()