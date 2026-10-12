r"""
Cifras canonicas para Resultados 5.1 (sin GPU, solo lee JSON)
=============================================================

Reune en un unico JSON las cifras que usa la seccion 5.1, cada una con el
fichero del que sale. No recalcula nada sobre predicciones ni mascaras: solo
lee JSON ya guardados y hace medias, minimos y maximos entre semillas.

    1. Barrido de lambda, pipeline v3 + perdida enmascarada (split temporal,
       test 2025, 3 semillas)
    2. Rendimiento base, red frente a RF justo, sin corregir el nivel
       a) split temporal (train 2021-23, test 2025: 220 teselas)
       b) split espacial (702 teselas sin fuga)
    3. Diagnostico LOYO-E2 por fold (3 semillas)
    4. Ablacion v3

No modifica nada. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/build_canonical_5_1.py
"""

import argparse
import json
import os

import numpy as np

SEEDS = ['7', '42', '123']
LAMBDAS = [('0.0', '00'), ('0.1', '01'), ('0.25', '025'), ('0.4', '04'),
           ('0.5', '05'), ('0.6', '06'), ('0.75', '075'), ('1.0', '10')]
FOLDS = ['2021', '2022', '2023', '2025']


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def mmm(vals, nd=4):
    v = np.asarray(vals, float)
    return {'media': round(float(v.mean()), nd), 'min': round(float(v.min()), nd),
            'max': round(float(v.max()), nd), 'por_semilla': [round(float(x), nd) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='analysis/cifras_canonicas_5_1.json')
    args = ap.parse_args()
    out = {'descripcion': 'Cifras canonicas de Resultados 5.1; cada bloque lleva su fuente. '
                          'Medias, minimos y maximos entre las semillas 7, 42 y 123 '
                          '(por_semilla en ese orden).'}

    # ------------------------------------------------------------ 1. lambda
    tpl = 'results/norm_v3/lambda/resunetpp_v3_sp{c}_s{s}/resunetpp_v3_sp{c}_s{s}_metrics.json'
    lam = {}
    for name, code in LAMBDAS:
        ms = [load(tpl.format(c=code, s=s)) for s in SEEDS]
        lam[name] = {'R2': mmm([m['R2'] for m in ms]), 'SPAEF': mmm([m['SPAEF'] for m in ms]),
                     'n_teselas_test': ms[0]['SPAEF_n_tiles']}
    best_r2 = max(lam, key=lambda k: lam[k]['R2']['media'])
    best_sp = max(lam, key=lambda k: lam[k]['SPAEF']['media'])
    out['1_barrido_lambda'] = {
        'fuente': tpl.replace('{c}', '<lambda>').replace('{s}', '<semilla>'),
        'resumen_ya_guardado': 'results/norm_v3/lambda/norm_v3_lambda_summary.json (media y desviacion)',
        'configuracion': 'ResUNet++, norm_version v3, masked_loss true, loss spatial_mse, '
                         'split temporal (train 2021-23, val 2024, test 2025), 50 epocas, '
                         'configs/norm_v3/resunetpp_v3_sp<lambda>_s<semilla>.yaml',
        'por_lambda': lam, 'mejor_lambda_R2': best_r2, 'mejor_lambda_SPAEF': best_sp,
        'aviso': 'Respecto al barrido antiguo cambian DOS cosas a la vez (normalizacion v3 y '
                 'perdida enmascarada; commits ded1a0d y f372687). No hay ejecuciones v3 sin '
                 'mascara: el barrido no aisla el efecto de la mascara.'}

    # ------------------------------------------------------- 2a. temporal
    rf_t = load('analysis/rf_temporal_allpix_train.json')
    net_t = [load(tpl.format(c='00', s=s)) for s in SEEDS]
    out['2a_base_temporal'] = {
        'definicion': 'split temporal completo: train 2021-2023 (todas las teselas), val 2024, '
                      'test 2025 (220 teselas, 5 fechas); R2 acumulado sobre pixeles con nieve, '
                      'SPAEF medio por tesela; sin corregir el nivel. NO es el fold LOYO 2025 '
                      '(ese submuestrea el train a 2659 teselas).',
        'red': {'R2': mmm([m['R2'] for m in net_t]), 'SPAEF': mmm([m['SPAEF'] for m in net_t]),
                'fuente': tpl.format(c='00', s='<semilla>')},
        'rf_justo': {'R2': mmm([rf_t['rf_nuevo']['por_semilla'][s]['R2'] for s in SEEDS]),
                     'SPAEF': mmm([rf_t['rf_nuevo']['por_semilla'][s]['SPAEF'] for s in SEEDS]),
                     'fuente': 'analysis/rf_temporal_allpix_train.json (rf_nuevo; todos los '
                               'pixeles validos, train 2021-2023)'},
        'rf_anterior_suplementario': {'R2': rf_t['rf_actual_solo_nieve']['R2'],
                                      'SPAEF': rf_t['rf_actual_solo_nieve']['SPAEF'],
                                      'nota': 'solo nieve y con 2024 en train'}}

    # -------------------------------------------------------- 2b. espacial
    c = load('analysis/eC_net_vs_rf_allpix.json')
    h8 = load('analysis/hallazgo8_fuga_espacial.json')
    de = load('analysis/decomp_espacial.json')
    out['2b_base_espacial'] = {
        'canonico_red_frente_a_rf': {
            'definicion': '702 teselas sin fuga; pixeles con nieve, dato valido y climatologia '
                          'disponible (mascara comun de C); sin corregir el nivel',
            'red': {'R2': mmm([r['r2_raw'] for r in c['red']]),
                    'SPAEF': mmm([r['spaef_raw'] for r in c['red']])},
            'rf_justo': {'R2': mmm([r['r2_raw'] for r in c['rf']]),
                         'SPAEF': mmm([r['spaef_raw'] for r in c['rf']])},
            'fuente': 'analysis/eC_net_vs_rf_allpix.json (scripts/spatial_models_vs_climatology.py)'},
        'otras_cifras_y_por_que_difieren': {
            'hallazgo8_limpios': {'R2': mmm([h8['por_semilla'][k]['limpios']['R2'] for k in h8['por_semilla']]),
                                  'mascara': 'todos los pixeles con nieve de las 702 teselas, '
                                             'sin exigir climatologia',
                                  'fuente': 'analysis/hallazgo8_fuga_espacial.json'},
            'decomp_espacial': {'R2': mmm([v['R2_raw'] for v in de['por_semilla'].values()]),
                                'mascara': 'la misma que hallazgo 8 (mismas cifras por semilla); '
                                           '0,471 frente a 0,472 es solo redondeo',
                                'fuente': 'analysis/decomp_espacial.json'},
            'hallazgo8_todos_730': {'R2': mmm([h8['por_semilla'][k]['todos']['R2'] for k in h8['por_semilla']]),
                                    'fuente': 'analysis/hallazgo8_fuga_espacial.json'}}}

    # ------------------------------------------------------------ 3. LOYO
    sc = load('analysis/e2_scale_checks.json')
    orc = load('analysis/e2_oraculos_loyo.json')
    dec = {'7': load('analysis/e2_decomposition.json'),
           '42': load('analysis/e2_decomposition_s42.json'),
           '123': load('analysis/e2_decomposition_s123.json')}

    def dec_row(s, fold):
        # etiquetas: 'loyo2023 (-38%)' en s7; '2023 s42' en s42 y s123
        return next(r for r in dec[s] if r['label'].replace('loyo', '').startswith(fold))

    loyo = {}
    for f in FOLDS:
        ps = sc['por_fold'][f]['por_semilla']
        o = orc['por_fold'][f]
        loyo[f] = {
            'n_teselas_test': ps['7']['n_tiles'],
            'R2_sin_corregir': mmm([ps[s]['R2_raw'] for s in SEEDS]),
            'R2_nivel_perfecto_aditivo': mmm([ps[s]['R2_oraculo_add'] for s in SEEDS]),
            'T1a_pct': mmm([ps[s]['T1a_pct'] for s in SEEDS], 2),
            'T1b_pct': mmm([ps[s]['T1b_pct'] for s in SEEDS], 2),
            'T2_pct': mmm([100 * ps[s]['T2'] / ps[s]['MSE'] for s in SEEDS], 2),
            'T3_pct': mmm([100 * ps[s]['T3'] / ps[s]['MSE'] for s in SEEDS], 2),
            'escala_T1_T2_pct': mmm([ps[s]['escala_pct_T1_T2'] for s in SEEDS], 2),
            'correlacion_patron_tesela': mmm([dec_row(s, f)['mean_rho'] for s in SEEDS]),
            'sesgo_medio_pixeles_m': mmm([ps[s]['sesgo_pixeles'] for s in SEEDS]),
            'sesgo_medio_teselas_m': mmm([ps[s]['sesgo_medio_tiles'] for s in SEEDS]),
            'anomalia_exacta_pct': sc['anomalia'][f]['anomalia_pct'],
            'anomalia_muestreo_pct': sc['anomalia'][f]['anomalia_muestreo_pct'],
            'control_T2_T3_e2_oraculos': {'T2_pct_media': o['T2_pct']['media'],
                                          'T3_pct_media': o['T3_pct']['media']}}
    out['3_loyo_e2'] = {
        'fuentes': {'R2, T1a, T1b, T2, T3, escala, sesgos, anomalia': 'analysis/e2_scale_checks.json',
                    'correlacion de patron': 'analysis/e2_decomposition.json (s7), '
                                             'e2_decomposition_s42.json, e2_decomposition_s123.json',
                    'control': 'analysis/e2_oraculos_loyo.json'},
        'definiciones': {'sesgo': 'prediccion - observacion (m); pixeles: ponderado por pixeles '
                                  '(sesgo de la fecha agregado); teselas: media sin ponderar de '
                                  'las teselas (el mean_bias de E1/E2)',
                         'anomalia_exacta': '(HS media del anyo de test - media de las HS anuales '
                                            'de train sin 2024) / esa media; todos los pixeles con nieve',
                         'escala': '(T1+T2)/MSE, criterio pre-registrado'},
        'por_fold': loyo,
        'media_4_folds': {'R2_sin_corregir': orc['media_4_folds']['R2_raw'],
                          'R2_nivel_perfecto_aditivo': orc['media_4_folds']['R2_oraculo_add']},
        'escala_mayor_50': sc['escala_mayor_50'],
        'signo_sesgo_opuesto_anomalia': sc['signo_opuesto_total']}

    # -------------------------------------------------------- 4. ablacion
    ab = load('analysis/ablation_v3_summary.json')
    out['4_ablacion_v3'] = {
        'fuente': 'analysis/ablation_v3_summary.json (scripts/ablation_v3_summary.py)',
        'alcance': 'split temporal, test 2025, lambda 0, norm v3, masked_loss, 3 semillas',
        'grupos': {g: {'canales_quitados': ab[g].get('canales_quitados', []),
                       'R2': {k: ab[g]['R2'][k] for k in ('media', 'min', 'max')},
                       'SPAEF': {k: ab[g]['SPAEF'][k] for k in ('media', 'min', 'max')}}
                   for g in ('completo', 'sin_pers', 'sin_sce', 'sin_sx', 'sin_topo1', 'sin_topo5')}}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'Guardado en: {args.out}')


if __name__ == '__main__':
    main()
