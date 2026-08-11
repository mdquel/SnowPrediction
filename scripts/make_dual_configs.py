"""Genera las configs LOYO para la arquitectura de doble cabeza,
a partir de las configs LOYO existentes."""
import os

YEARS = [2021, 2022, 2023, 2025]
CFG_DIR = 'configs/norm_v3'

for y in YEARS:
    src = os.path.join(CFG_DIR, f'resunetpp_v3_loyo{y}.yaml')
    dst = os.path.join(CFG_DIR, f'resunetpp_dual_loyo{y}.yaml')

    with open(src, encoding='utf-8') as f:
        txt = f.read()

    txt = txt.replace(f'resunetpp_v3_loyo{y}', f'resunetpp_dual_loyo{y}')
    txt = txt.replace('architecture: resunetpp', 'architecture: resunetpp_dual')
    txt = txt.replace('loss: spatial_mse', 'loss: dual_head')
    txt = txt.replace('lambda_pearson: 0.0',
                      'lambda_mu: 1.0\n  lambda_sigma: 1.0')
    txt = txt.replace('results/norm_v3/loyo/', 'results/norm_v3/loyo_dual/')

    with open(dst, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'creada: {dst}')