r"""
Resumen de la ablacion v3 (sin entrenar nada)
=============================================

Compara el modelo completo con los cinco grupos de la ablacion de canales de
la suite v3 de Hernan, todos con el mismo pipeline: normalizacion v3, split
temporal (test 2025), perdida enmascarada, lambda 0, semillas 7, 42 y 123.

    completo   results/norm_v3/lambda/resunetpp_v3_sp00_s<seed>
    ablacion   results/norm_v3/ablation/resunetpp_v3_abl_<grupo>_s<seed>

Antes de comparar, verifica (sobre el YAML interpretado, no el texto) que las
configuraciones del completo y de cada grupo solo difieren en el nombre, las
rutas de salida y los canales, e informa de los canales que quita cada grupo.
Si difieren en algo mas, lo avisa.

Interes principal: el canal 5 (SCE). Al binarizarse con > 5, los codigos 10
(sin nieve) y 11 (nieve) pasan ambos a 1, asi que el canal codifica
observacion despejada, no nieve. Este resumen mide cuanto pesa: si los
rangos de sin_sce y del completo se solapan, quitar el canal no cambia el
resultado de forma distinguible.

Uso
---
    python scripts/ablation_v3_summary.py
Salida: analysis/ablation_v3_summary.json
"""

import json
import os

import numpy as np
import yaml

SEEDS = (7, 42, 123)
GROUPS = ('sin_pers', 'sin_sce', 'sin_sx', 'sin_topo1', 'sin_topo5')
FULL = 'results/norm_v3/lambda/resunetpp_v3_sp00_s{s}/resunetpp_v3_sp00_s{s}_metrics.json'
ABL = 'results/norm_v3/ablation/resunetpp_v3_abl_{g}_s{s}/resunetpp_v3_abl_{g}_s{s}_metrics.json'
CFG_FULL = 'configs/norm_v3/resunetpp_v3_sp00_s{s}.yaml'
CFG_ABL = 'configs/norm_v3/resunetpp_v3_abl_{g}_s{s}.yaml'
OUT = 'analysis/ablation_v3_summary.json'

# claves que se espera que difieran entre el completo y la ablacion
ESPERADAS = {('experiment', 'name'), ('output', 'models_dir'),
             ('output', 'results_dir'), ('output', 'model_name'),
             ('model', 'in_channels'), ('data', 'channel_indices')}


def metrics(path):
    d = json.load(open(path, encoding='utf-8'))
    d = d.get('test_metrics', d)
    return float(d['R2']), float(d['SPAEF'])


def _flat(d, pre=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, pre + (k,)))
        else:
            out[pre + (k,)] = v
    return out


def config_diffs(a, b):
    """Compara los YAML ya interpretados (no el texto): devuelve las claves
    que difieren y no son de las esperadas, y los canales quitados."""
    fa = _flat(yaml.safe_load(open(a, encoding='utf-8')))
    fb = _flat(yaml.safe_load(open(b, encoding='utf-8')))
    raras = [f"{'.'.join(k)}: {fa.get(k)!r} -> {fb.get(k)!r}"
             for k in sorted(set(fa) | set(fb))
             if k not in ESPERADAS and fa.get(k) != fb.get(k)]
    n_in = fa[('model', 'in_channels')]
    quitados = sorted(set(range(n_in)) - set(fb.get(('data', 'channel_indices'), range(n_in))))
    return raras, quitados


def resume(vals):
    v = np.array(vals)
    return {'media': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
            'por_semilla': dict(zip([str(s) for s in SEEDS], v.round(4).tolist()))}


def main():
    # 1. verificacion de configuraciones
    print('Verificando configuraciones (completo frente a cada grupo, semilla 7)...')
    avisos, canales = {}, {}
    for g in GROUPS:
        a, b = CFG_FULL.format(s=7), CFG_ABL.format(g=g, s=7)
        if not (os.path.exists(a) and os.path.exists(b)):
            avisos[g] = ['falta alguna config']
            continue
        raras, canales[g] = config_diffs(a, b)
        print(f'  {g:<10} quita canales {canales[g]}')
        if raras:
            avisos[g] = raras
    if avisos:
        print('  AVISO: diferencias no esperadas:')
        for g, r in avisos.items():
            print(f'    {g}:'); [print(f'      {x}') for x in r]
    else:
        print('  OK: solo difieren nombre, rutas y canales (comparando el YAML interpretado).')

    # 2. tabla
    full = [metrics(FULL.format(s=s)) for s in SEEDS]
    res = {'completo': {'ruta': FULL.format(s='<seed>'),
                        'R2': resume([x[0] for x in full]),
                        'SPAEF': resume([x[1] for x in full])}}
    for g in GROUPS:
        m = [metrics(ABL.format(g=g, s=s)) for s in SEEDS]
        res[g] = {'ruta': ABL.format(g=g, s='<seed>'), 'canales_quitados': canales.get(g),
                  'R2': resume([x[0] for x in m]),
                  'SPAEF': resume([x[1] for x in m])}

    print('\nTest 2025, media [min-max] de las semillas 7, 42, 123')
    print(f"  {'modelo':<11}{'R2':>24}{'dif':>9}{'SPAEF':>24}{'dif':>9}")
    for k, v in res.items():
        r, s_ = v['R2'], v['SPAEF']
        dr = r['media'] - res['completo']['R2']['media']
        ds = s_['media'] - res['completo']['SPAEF']['media']
        print(f"  {k:<11}{r['media']:>8.3f} [{r['min']:.3f}-{r['max']:.3f}]"
              f"{(dr if k != 'completo' else 0):>+9.3f}"
              f"{s_['media']:>8.3f} [{s_['min']:.3f}-{s_['max']:.3f}]"
              f"{(ds if k != 'completo' else 0):>+9.3f}")

    # 3. canal 5
    def solapan(a, b):
        return not (a['max'] < b['min'] or b['max'] < a['min'])
    sol = {met: solapan(res['sin_sce'][met], res['completo'][met]) for met in ('R2', 'SPAEF')}
    print('\nCanal 5 (sin_sce frente al completo):')
    for met in ('R2', 'SPAEF'):
        print(f"  {met:<6} rangos {'SE SOLAPAN' if sol[met] else 'NO se solapan'}")
    lectura = ('quitar el canal no cambia el resultado de forma distinguible'
               if all(sol.values()) else
               'quitar el canal cambia el resultado: hay que mirarlo con cuidado')
    print(f'  Lectura: {lectura}.')

    res['_verificacion_configs'] = avisos or 'OK'
    res['_canal5'] = {'rangos_solapan': sol, 'lectura': lectura}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {OUT}')


if __name__ == '__main__':
    main()
