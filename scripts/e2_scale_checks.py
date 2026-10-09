r"""
Tres comprobaciones sobre las predicciones de E2/E2b (sin GPU)
==============================================================

Sobre los 4 folds LOYO x 3 semillas (7 en E2, 42 y 123 en E2b). No
modifica nada: escribe solo el JSON de --out.

1. Separacion del termino de sesgo T1
   T1 (por tile, ponderado por pixeles) = sum_t w_t * b_t^2, con
   b_t = mu_s,t - mu_o,t y w_t = n_t / N. Se separa EXACTAMENTE en
       T1a (fecha)        = sum_d W_d * b_d^2
       T1b (entre tiles)  = sum_t w_t * (b_t - b_d)^2
   donde b_d es el sesgo medio de la fecha con todos sus pixeles con nieve
   (media predicha - media observada de la fecha) y W_d = N_d / N.
   T1a + T1b = T1 (identidad de la varianza; el script lo comprueba).
   Se informa ademas de la fraccion de escala contando solo T1a + T2.

2. Oraculo aditivo frente al multiplicativo
   R2 acumulado (pixeles con nieve) tras corregir cada fecha:
       multiplicativo: pred * (mu_o,d / mu_s,d)   (el R2_recalibrated de E1/E2)
       aditivo:        pred + (mu_o,d - mu_s,d)   (como en B, C y A)

3. Anomalia exacta del anyo de test
   HS del anyo = media de TODOS los pixeles validos con HS > 0.01 m de
   todos los tiles del anyo. El test de cada fold son todos los tiles de su
   anyo, asi que se obtiene de los targets del .npz de la semilla 7.
   Anomalia = (HS_test - HS_train) / HS_train, con HS_train la media de las
   HS anuales de los anyos de entrenamiento (sin el de test ni 2024), igual
   que la estimacion por muestreo de scripts/generate_loyo_configs.py. Se
   da tambien la variante con todos los pixeles de train juntos.
   Se comprueba si el signo del sesgo es el opuesto al de la anomalia.

Mismos filtros que analysis/e1_error_decomposition.py, cuyas funciones se
importan (tile_mask, decompose_tile, date_from_tile_id): pixeles con nieve,
tiles con al menos 10 pixeles y varianza no nula.

Uso
---
    python scripts/e2_scale_checks.py --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas ^
        --out analysis/e2_scale_checks.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / 'analysis'))
from e1_error_decomposition import (tile_mask, decompose_tile,          # noqa: E402
                                    date_from_tile_id, MIN_VALID_PIXELS)

FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
VAL_YEAR = '2024'
SNOW = 0.01
ANOM_MUESTREO = {'2021': 16, '2022': -16, '2023': -38, '2025': 49}   # etiquetas de E2 (%)


def npz_path(fold, seed, seed7_dir, seeds_dir):
    if seed == '7':
        return os.path.join(seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                            f'resunetpp_v3_loyo{fold}_predictions.npz')
    return os.path.join(seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{seed}',
                        f'resunetpp_v3_loyo{fold}_s{seed}_predictions.npz')


def r2(o, s):
    return float(1.0 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2))


def analyse(path):
    z = np.load(path)
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None

    tiles = []                      # (fecha, n, b_t, t1, t2, t3, mse_t)
    obs, sim, dates = [], [], []
    for i in range(len(ids)):
        m = tile_mask(tgts[i], None, 'snow')
        if m.sum() < MIN_VALID_PIXELS:
            continue
        o, s = tgts[i][m], preds[i][m]
        dec = decompose_tile(o, s)
        if dec is None:
            continue
        d = date_from_tile_id(ids[i])
        tiles.append((d, dec['n'], dec['mu_s'] - dec['mu_o'], dec['t1'], dec['t2'],
                      dec['t3'], dec['mse']))
        obs.append(o.astype(np.float64)); sim.append(s.astype(np.float64)); dates.append(d)

    n = np.array([t[1] for t in tiles], float)
    w = n / n.sum()
    b = np.array([t[2] for t in tiles])
    T = {k: float(np.sum(w * np.array([t[j] for t in tiles])))
         for k, j in (('T1', 3), ('T2', 4), ('T3', 5), ('MSE', 6))}

    # sesgo de cada fecha con todos sus pixeles = media ponderada de b_t
    dlist = np.array([t[0] for t in tiles])
    b_d = {d: float(np.sum(n[dlist == d] * b[dlist == d]) / n[dlist == d].sum())
           for d in np.unique(dlist)}
    bd_t = np.array([b_d[d] for d in dlist])
    T1a = float(np.sum(w * bd_t ** 2))
    T1b = float(np.sum(w * (b - bd_t) ** 2))
    assert abs(T1a + T1b - T['T1']) <= 1e-6 * max(T['T1'], 1e-12), 'T1a+T1b != T1'

    # oraculos por fecha
    mo = {d: np.mean(np.concatenate([o for o, dd in zip(obs, dates) if dd == d])) for d in b_d}
    ms = {d: np.mean(np.concatenate([s for s, dd in zip(sim, dates) if dd == d])) for d in b_d}
    O = np.concatenate(obs)
    S = np.concatenate(sim)
    S_mult = np.concatenate([s * (mo[d] / ms[d]) for s, d in zip(sim, dates)])
    S_add = np.concatenate([s + (mo[d] - ms[d]) for s, d in zip(sim, dates)])

    mse = T['MSE']
    return {
        'n_tiles': len(tiles),
        'MSE': mse, 'T1': T['T1'], 'T1a_fecha': T1a, 'T1b_entre_tiles': T1b,
        'T2': T['T2'], 'T3': T['T3'],
        'escala_pct_T1_T2': 100 * (T['T1'] + T['T2']) / mse,
        'escala_pct_T1a_T2': 100 * (T1a + T['T2']) / mse,
        'T1a_pct': 100 * T1a / mse, 'T1b_pct': 100 * T1b / mse,
        'R2_raw': r2(O, S), 'R2_oraculo_mult': r2(O, S_mult), 'R2_oraculo_add': r2(O, S_add),
        'sesgo_medio_tiles': float(np.mean(b)),            # como mean_bias de E1/E2
        'sesgo_pixeles': float(S.mean() - O.mean()),
    }


def year_hs(path):
    """Suma y numero de pixeles validos con nieve de todos los tiles del .npz."""
    z = np.load(path)
    t = np.asarray(z['targets'], dtype=np.float64)
    m = np.isfinite(t) & (t > SNOW)
    if 'valids' in z.files:
        m &= np.asarray(z['valids']) > 0.5
    return float(t[m].sum()), int(m.sum())


def summ(v):
    v = np.asarray(v, float)
    return {'media': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/e2_scale_checks.json')
    args = ap.parse_args()

    res, hs = {}, {}
    for fold in args.folds:
        res[fold] = {}
        for seed in SEEDS:
            p = npz_path(fold, seed, args.seed7_dir, args.seeds_dir)
            print(f'  fold {fold} semilla {seed}: {os.path.basename(p)}', flush=True)
            res[fold][seed] = analyse(p)
            if seed == '7':
                hs[fold] = year_hs(p)

    out = {'descripcion': __doc__.split('Uso')[0].strip(), 'por_fold': {}}
    print('\n1. Separacion de T1 (media [min-max] de 3 semillas, fraccion del MSE)')
    print(f"  {'fold':<6}{'T1a fecha':>18}{'T1b tiles':>18}{'T2':>16}{'T3':>18}"
          f"{'escala T1+T2':>18}{'escala T1a+T2':>18}")
    n50_old = n50_new = 0
    for fold in args.folds:
        r = res[fold]
        f = {k: summ([r[s][k] for s in SEEDS]) for k in
             ('MSE', 'T1', 'T1a_fecha', 'T1b_entre_tiles', 'T2', 'T3', 'T1a_pct', 'T1b_pct',
              'escala_pct_T1_T2', 'escala_pct_T1a_T2', 'R2_raw', 'R2_oraculo_mult',
              'R2_oraculo_add')}
        f['por_semilla'] = r
        out['por_fold'][fold] = f
        n50_old += sum(r[s]['escala_pct_T1_T2'] > 50 for s in SEEDS)
        n50_new += sum(r[s]['escala_pct_T1a_T2'] > 50 for s in SEEDS)
        pct = lambda k: 100 * f[k]['media'] / f['MSE']['media']
        print(f"  {fold:<6}{pct('T1a_fecha'):>8.1f}%"
              f"{'':>9}{pct('T1b_entre_tiles'):>8.1f}%{'':>9}{pct('T2'):>7.1f}%{'':>8}"
              f"{pct('T3'):>8.1f}%{'':>9}"
              f"{f['escala_pct_T1_T2']['media']:>6.1f} [{f['escala_pct_T1_T2']['min']:.1f}-{f['escala_pct_T1_T2']['max']:.1f}]"
              f"  {f['escala_pct_T1a_T2']['media']:>6.1f} [{f['escala_pct_T1a_T2']['min']:.1f}-{f['escala_pct_T1a_T2']['max']:.1f}]")
    nrun = len(args.folds) * len(SEEDS)
    print(f'  Ejecuciones con escala > 50%: T1+T2 {n50_old} de {nrun}; T1a+T2 {n50_new} de {nrun}')
    out['escala_mayor_50'] = {'T1_T2': n50_old, 'T1a_T2': n50_new, 'de': nrun}

    print('\n2. Oraculo por fecha, R2 (media [min-max] de 3 semillas)')
    for fold in args.folds:
        f = out['por_fold'][fold]
        print(f"  {fold}: sin corregir {f['R2_raw']['media']:.3f}   multiplicativo "
              f"{f['R2_oraculo_mult']['media']:.3f} [{f['R2_oraculo_mult']['min']:.3f}-{f['R2_oraculo_mult']['max']:.3f}]"
              f"   aditivo {f['R2_oraculo_add']['media']:.3f} [{f['R2_oraculo_add']['min']:.3f}-{f['R2_oraculo_add']['max']:.3f}]")
    for k in ('R2_raw', 'R2_oraculo_mult', 'R2_oraculo_add'):
        out[f'media_4_folds_{k}'] = {s: round(float(np.mean([res[f][s][k] for f in args.folds])), 4)
                                     for s in SEEDS}
    print(f"  media de los folds, semilla 7: multiplicativo {out['media_4_folds_R2_oraculo_mult']['7']:.3f}"
          f"   aditivo {out['media_4_folds_R2_oraculo_add']['7']:.3f}")

    if set(FOLDS) <= set(args.folds):
        print('\n3. Anomalia exacta (todos los pixeles con nieve del anyo)')
        hs_year = {y: s / n for y, (s, n) in hs.items()}
        out['hs_anual_m'] = {y: round(v, 4) for y, v in hs_year.items()}
        anom, ok = {}, 0
        for fold in FOLDS:
            tr = [y for y in FOLDS if y not in (fold, VAL_YEAR)]
            m_years = float(np.mean([hs_year[y] for y in tr]))
            m_pool = sum(hs[y][0] for y in tr) / sum(hs[y][1] for y in tr)
            a = 100 * (hs_year[fold] - m_years) / m_years
            a_pool = 100 * (hs_year[fold] - m_pool) / m_pool
            signos = {s: int(np.sign(res[fold][s]['sesgo_medio_tiles'])) for s in SEEDS}
            coinc = sum(sg == -int(np.sign(a)) for sg in signos.values())
            ok += coinc
            anom[fold] = {'anomalia_pct': round(a, 1), 'anomalia_pct_pixeles_juntos': round(a_pool, 1),
                          'anomalia_muestreo_pct': ANOM_MUESTREO[fold],
                          'signo_sesgo_por_semilla': signos, 'signo_opuesto_en': coinc}
            print(f"  {fold}: HS {hs_year[fold]:.3f} m  anomalia {a:+.1f}% (pixeles juntos {a_pool:+.1f}%,"
                  f" muestreo {ANOM_MUESTREO[fold]:+d}%)  sesgo opuesto en {coinc}/3")
        print(f'  Signo del sesgo opuesto al de la anomalia: {ok} de 12')
        out['anomalia'] = anom
        out['signo_opuesto_total'] = {'coinciden': ok, 'de': 12}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
