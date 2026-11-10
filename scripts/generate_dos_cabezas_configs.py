r"""
Configs de la red de dos cabezas (protocolo 2026, apartado 8.13)
================================================================

Misma entrada que la red con climatologia (configs/clim_v1): los 26 canales
del brazo B mas la climatologia (30) y su mascara (31), 28 en total. Mismos
hiperparametros (Optuna Trial 17), 50 epocas, perdida enmascarada, lambda 0 y
las semillas 7, 42 y 123. Lo que cambia:

    * modelo: resunetpp_dos_cabezas (models/resunetpp_dos_cabezas.py), con w
      en [0.5, 1.5] y delta en [-0.5, +0.5] m, un valor por tesela;
    * climatologia: clim_v2 (data/patch_clim_v2.py). En entrenamiento, cada
      temporada recibe la de las otras dos; en validacion y test es identica
      a clim_v1;
    * sin apagado aleatorio de la climatologia (no hay clim_channels ni
      clim_dropout), y por tanto sin evaluacion "_sinclim";
    * niveles por fecha: analysis/dos_cabezas/niveles_s<semilla>.csv
      (scripts/dos_cabezas_niveles.py). Nivel observado en entrenamiento y
      validacion; nivel de la red B de la misma semilla en el test (mejor
      epoca con nivel_m, ultima con nivel_last_m);
    * perdida solo sobre los pixeles con climatologia (canal 31 > 0), en
      entrenamiento y en validacion. El test se evalua sobre todos los
      pixeles validos, como el resto de modelos.

clim_index es la posicion de la climatologia DENTRO de los 28 canales de
entrada (26); loss_requires_channel es el indice de su mascara en la pila
completa (31).

Uso (desde la raiz del repo)
----------------------------
    python scripts/generate_dos_cabezas_configs.py
"""

import os

import yaml

SEEDS = (7, 42, 123)
B = [c for c in range(22) if c not in (5, 14, 15, 16)] + [22, 23] + list(range(24, 30))
CLIM, CLIM_DISP = 30, 31
OUT_DIR = 'configs/dos_cabezas'
RES_DIR = 'results/dos_cabezas'


def config(seed):
    name = f'resunetpp_dc1_s{seed}'
    idx = B + [CLIM, CLIM_DISP]
    return {
        'experiment': {'name': name},
        'data': {'root': 'dataset_v4_ms_sx200',
                 'csv_file': 'dataset_v4_ms_sx200.csv',
                 'images_dir': 'images', 'masks_dir': 'masks',
                 'source': 'lidar', 'split_type': 'temporal',
                 'use_sce': False, 'augmentation': False, 'norm_version': 'v3',
                 'extra_channels_dir': ['sat_v2', 'clim_v2'],
                 'channel_indices': idx,
                 'date_levels_csv': f'analysis/dos_cabezas/niveles_s{seed}.csv',
                 'date_level_cols': ['nivel_m', 'clim_media_m'],
                 'date_level_cols_last': ['nivel_last_m', 'clim_media_m'],
                 'loss_requires_channel': CLIM_DISP},
        'model': {'architecture': 'resunetpp_dos_cabezas', 'in_channels': len(idx),
                  'out_channels': 1, 'features': [64, 128, 256, 512],
                  'dropout_p': 0.077, 'num_groups': 8,
                  'clim_index': idx.index(CLIM),
                  'w_range': [0.5, 1.5], 'delta_max': 0.5},
        'training': {'seed': seed, 'batch_size': 8, 'learning_rate': 1.287e-04,
                     'epochs': 50, 'loss': 'spatial_mse', 'lambda_pearson': 0.0,
                     'masked_loss': True, 'weight_decay': 1.239e-04,
                     'optimizer': 'adamw', 'grad_clip': 1.0, 'early_stopping': False,
                     'num_workers': 0, 'device': 'auto'},
        'evaluation': {'save_predictions': True},
        'output': {'models_dir': f'{RES_DIR}/weights',
                   'results_dir': f'{RES_DIR}/{name}',
                   'model_name': name},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SEEDS:
        cfg = config(s)
        path = os.path.join(OUT_DIR, f"{cfg['experiment']['name']}.yaml")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('# Red de dos cabezas, protocolo 2026 (8.13) (scripts/generate_dos_cabezas_configs.py)\n')
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"{path}: {cfg['model']['in_channels']} canales, climatologia en la posicion "
              f"{cfg['model']['clim_index']} de la entrada")


if __name__ == '__main__':
    main()
