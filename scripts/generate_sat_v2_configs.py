r"""
Configs del experimento de canales satelitales v2 (paso 1)
==========================================================

Tres brazos x tres semillas en el split temporal (train 2021-23, val 2024,
test 2025), con todo lo demas IGUAL que configs/norm_v3/resunetpp_v3_sp00_s*.yaml
(lambda 0, norm v3, perdida enmascarada, 50 epocas, Optuna Trial 17):

    base  los 22 canales actuales; se reentrena en el mismo entorno que A y B
          para que la comparacion no mezcle maquinas ni versiones (hallazgo 14:
          repetir el mismo entrenamiento ya mueve el R2 0,0125)
    A     canal 5 sustituido por sat_nieve y sat_observado (22, 23)
    B     como A y ademas persistencia 14-16 sustituida por
          pers_nieve_15/30/60 (24-26) y frac_despejado_15/30/60 (27-29)

B - A aisla el efecto de la persistencia; A - base el del canal del dia.

Todas guardan predicciones (evaluation.save_predictions) para que la
descomposicion del error y cualquier analisis posterior salgan del mismo
entrenamiento que las metricas.

Uso (desde la raiz del repo)
----------------------------
    python scripts/generate_sat_v2_configs.py
"""

import os

import yaml

SEEDS = (7, 42, 123)
ORIG = list(range(22))
DIA = [22, 23]
PERS_NUEVA = list(range(24, 30))
ARMS = {
    'base': {'indices': None, 'extra': False},
    'A': {'indices': [c for c in ORIG if c != 5] + DIA, 'extra': True},
    'B': {'indices': [c for c in ORIG if c not in (5, 14, 15, 16)] + DIA + PERS_NUEVA,
          'extra': True},
}
OUT_DIR = 'configs/sat_v2'


def config(arm, seed):
    spec = ARMS[arm]
    name = f'resunetpp_sat2_{arm}_s{seed}'
    data = {'root': 'dataset_v4_ms_sx200',
            'csv_file': 'dataset_v4_ms_sx200.csv',
            'images_dir': 'images', 'masks_dir': 'masks',
            'source': 'lidar', 'split_type': 'temporal',
            'use_sce': False, 'augmentation': False, 'norm_version': 'v3'}
    if spec['extra']:
        data['extra_channels_dir'] = 'sat_v2'
        data['channel_indices'] = spec['indices']
    return {
        'experiment': {'name': name},
        'data': data,
        'model': {'architecture': 'resunetpp',
                  'in_channels': len(spec['indices']) if spec['indices'] else 22,
                  'out_channels': 1, 'features': [64, 128, 256, 512],
                  'dropout_p': 0.077, 'num_groups': 8},
        'training': {'seed': seed, 'batch_size': 8, 'learning_rate': 1.287e-04,
                     'epochs': 50, 'loss': 'spatial_mse', 'lambda_pearson': 0.0,
                     'masked_loss': True, 'weight_decay': 1.239e-04,
                     'optimizer': 'adamw', 'grad_clip': 1.0, 'early_stopping': False,
                     'num_workers': 0, 'device': 'auto'},
        'evaluation': {'save_predictions': True},
        'output': {'models_dir': 'results/sat_v2/weights',
                   'results_dir': f'results/sat_v2/{name}',
                   'model_name': name},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for arm in ARMS:
        for s in SEEDS:
            cfg = config(arm, s)
            path = os.path.join(OUT_DIR, f"{cfg['experiment']['name']}.yaml")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f'# Experimento satelital v2, brazo {arm} (scripts/generate_sat_v2_configs.py)\n')
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"{path}: {cfg['model']['in_channels']} canales")


if __name__ == '__main__':
    main()
