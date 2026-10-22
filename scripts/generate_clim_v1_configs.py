r"""
Configs del experimento de climatologia v1 (paso 2)
===================================================

Brazo Bclim = brazo B del paso 1 (configs/sat_v2/resunetpp_sat2_B_s*.yaml)
mas los 2 canales de climatologia (data/patch_clim_channels.py), con TODO lo
demas identico: split temporal, norm v3, perdida enmascarada, lambda 0,
50 epocas, Optuna Trial 17 y las semillas 7, 42 y 123.

Pila completa de canales que lee data/dataset.py:
    0-21   los 22 originales
    22-29  sat_v2 (canal del dia, persistencia y fraccion despejada)
    30     clim       climatologia causal (m)
    31     clim_disp  1 donde hay climatologia

Brazo B: 0-21 sin 5 ni 14-16, mas 22-29 (26 canales). Bclim: B + 30, 31
(28 canales). En el entrenamiento, con probabilidad 0,5 por tesela, los
canales 30 y 31 se ponen a 0 (fijado antes de entrenar). La evaluacion se
hace dos veces: con climatologia y con ella apagada (sufijo _sinclim).

La comparacion con B se hace contra results/sat_v2 (misma maquina y entorno).

Uso (desde la raiz del repo)
----------------------------
    python scripts/generate_clim_v1_configs.py
"""

import os

import yaml

SEEDS = (7, 42, 123)
B = [c for c in range(22) if c not in (5, 14, 15, 16)] + [22, 23] + list(range(24, 30))
CLIM = [30, 31]
OUT_DIR = 'configs/clim_v1'


def config(seed):
    name = f'resunetpp_clim1_Bclim_s{seed}'
    idx = B + CLIM
    return {
        'experiment': {'name': name},
        'data': {'root': 'dataset_v4_ms_sx200',
                 'csv_file': 'dataset_v4_ms_sx200.csv',
                 'images_dir': 'images', 'masks_dir': 'masks',
                 'source': 'lidar', 'split_type': 'temporal',
                 'use_sce': False, 'augmentation': False, 'norm_version': 'v3',
                 'extra_channels_dir': ['sat_v2', 'clim_v1'],
                 'channel_indices': idx,
                 'clim_channels': CLIM,
                 'clim_dropout': 0.5},
        'model': {'architecture': 'resunetpp', 'in_channels': len(idx),
                  'out_channels': 1, 'features': [64, 128, 256, 512],
                  'dropout_p': 0.077, 'num_groups': 8},
        'training': {'seed': seed, 'batch_size': 8, 'learning_rate': 1.287e-04,
                     'epochs': 50, 'loss': 'spatial_mse', 'lambda_pearson': 0.0,
                     'masked_loss': True, 'weight_decay': 1.239e-04,
                     'optimizer': 'adamw', 'grad_clip': 1.0, 'early_stopping': False,
                     'num_workers': 0, 'device': 'auto'},
        'evaluation': {'save_predictions': True},
        'output': {'models_dir': 'results/clim_v1/weights',
                   'results_dir': f'results/clim_v1/{name}',
                   'model_name': name},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SEEDS:
        cfg = config(s)
        path = os.path.join(OUT_DIR, f"{cfg['experiment']['name']}.yaml")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('# Experimento de climatologia v1, brazo Bclim (scripts/generate_clim_v1_configs.py)\n')
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"{path}: {cfg['model']['in_channels']} canales")


if __name__ == '__main__':
    main()
