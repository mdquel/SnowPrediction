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
    * el arranque (la combinacion) se guarda como epoca 0 y solo lo
      sustituye una epoca que mejore la perdida de validacion
      (training.save_initial, 8.15);
    * perdida solo sobre los pixeles con climatologia (canal 31 > 0), en
      entrenamiento y en validacion. El test se evalua sobre todos los
      pixeles validos, como el resto de modelos.

clim_index es la posicion de la climatologia DENTRO de los 28 canales de
entrada (26); loss_requires_channel es el indice de su mascara en la pila
completa (31).

Variante dc2 (apartado 8.16)
----------------------------
Igual que dc1 salvo el modelo: el codificador ve solo los 26 canales de B,
arranca con los pesos de la mejor epoca de la red B de la misma semilla y
queda congelado; las cabezas no reciben L(d) ni Cm(d). Solo se entrenan las
cabezas. Resultados en results/dos_cabezas_v2.

Uso (desde la raiz del repo)
----------------------------
    python scripts/generate_dos_cabezas_configs.py               # dc1
    python scripts/generate_dos_cabezas_configs.py --variante dc2
"""

import argparse
import os

import yaml

SEEDS = (7, 42, 123)
B = [c for c in range(22) if c not in (5, 14, 15, 16)] + [22, 23] + list(range(24, 30))
CLIM, CLIM_DISP = 30, 31
OUT_DIR = 'configs/dos_cabezas'
RES_DIR = {'dc1': 'results/dos_cabezas', 'dc2': 'results/dos_cabezas_v2'}
B_WEIGHTS = 'results/regen2024/weights/resunetpp_sat2_B_s{seed}.pth'


def config(seed, variante='dc1'):
    name = f'resunetpp_{variante}_s{seed}'
    idx = B + [CLIM, CLIM_DISP]
    res_dir = RES_DIR[variante]
    cfg = {
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
                     'num_workers': 0, 'device': 'auto', 'save_initial': True},
        'evaluation': {'save_predictions': True},
        'output': {'models_dir': f'{res_dir}/weights',
                   'results_dir': f'{res_dir}/{name}',
                   'model_name': name},
    }
    if variante == 'dc2':
        cfg['model'].update({'encoder_in_channels': len(B),
                             'encoder_init': B_WEIGHTS.format(seed=seed),
                             'freeze_encoder': True,
                             'head_signals': False})
    return cfg


def main():
    ap = argparse.ArgumentParser(description='Configs de la red de dos cabezas')
    ap.add_argument('--variante', choices=sorted(RES_DIR), default='dc1')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SEEDS:
        cfg = config(s, a.variante)
        path = os.path.join(OUT_DIR, f"{cfg['experiment']['name']}.yaml")
        with open(path, 'w', encoding='utf-8') as f:
            apartado = '8.16' if a.variante == 'dc2' else '8.13'
            f.write(f'# Red de dos cabezas {a.variante}, protocolo 2026 ({apartado}) '
                    '(scripts/generate_dos_cabezas_configs.py)\n')
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"{path}: {cfg['model']['in_channels']} canales, climatologia en la posicion "
              f"{cfg['model']['clim_index']} de la entrada")


if __name__ == '__main__':
    main()
