r"""
Seleccion de la figura de mapas en terreno ya sobrevolado (LOYO-E2, seccion 5.1)
================================================================================

Solo SELECCIONA (no dibuja). Para los folds 2021 y 2022 aplica la regla fijada
antes de ver los mapas y guarda en analysis/fig_loyo_seleccion.json todo lo
necesario para la figura y para revisarla.

Regla (decisiones del 13-oct-2026)
----------------------------------
0. Semilla: en cada fold, la de R2 sin corregir mediano entre 7, 42 y 123
   (analysis/e2_scale_checks.json, tabla LOYO).
1. Fecha: solo fechas de test con senal de cobertura (la correccion E5 la
   necesita). Se ordenan por T1a de la fecha = b_d^2, con b_d = media predicha -
   media observada sobre los pixeles con nieve de la fecha. Se toma la mediana;
   con un numero par, la de menor T1a de las dos centrales. Se da tambien la
   contribucion al MSE (W_d b_d^2), solo como informacion.
2. Tesela: dentro de esa fecha, teselas con al menos un 50 % de pixeles con
   nieve (como la Fig. 5.2), ordenadas por el R2 de la red tal cual; mediana con
   la misma regla para el caso par.
Mascara de pixeles: la de B (tab:surveyed): HS > 0.01 m, dato valido y
climatologia de los anyos de train disponible; al menos 10 pixeles por tesela
(scripts/climatology_baseline.py).
Correccion E5: aditiva, pred + (media estimada por el regresor - media predicha
de la fecha), con el regresor ridge (lambda = 1) sobre snow_pct ajustado con las
fechas de train del fold y la media de todas sus teselas
(scripts/e5_regressor_allmeans.fit_regressor). Es el "nivel del regresor" de B.

2022-12-01 (fecha de baja varianza): la regla se aplica tal cual; si la fecha
elegida es esa, se da tambien la seleccion sin ella para que se decida.

Para la figura (figscripts/fig_mapas_loyo.py del articulo) guarda ademas los
campos de las dos teselas elegidas en --fig-npz (observada, red tal cual y
mascara) y, para el pie, el R2 del fold con la semilla elegida en tres
versiones sobre la mascara de B: tal cual, nivel del regresor (E5) y nivel
perfecto de la fecha. Se comprueban contra analysis/eB_climatology_v2.json y
analysis/e5_regresor_dependientes.json.

Nivel perfecto de la fecha = media observada - media predicha sobre todas las
teselas de la fecha (como en el articulo). El "mejor desplazamiento de la
tesela" (su propia media) se guarda aparte, solo como diagnostico.

No dibuja nada ni modifica ficheros existentes. Escribe solo --out y --fig-npz.

Uso (desde la raiz del repo)
----------------------------
    python scripts/fig_loyo_seleccion.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_test import date_of, ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX   # noqa: E402
from e5_regressor_allmeans import TileCache, means_by_date_masks, fit_regressor     # noqa: E402
from climatology_baseline import build_climatology                                   # noqa: E402

TILE = 256
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
LOW_VAR_DATE = '20221201'
MIN_SNOW_FRAC = 0.5


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def median_pick(items, key):
    """Ordena por key; mediana, o la menor de las dos centrales si n es par."""
    s = sorted(items, key=key)
    n = len(s)
    return s[(n - 1) // 2], s


def r2(o, p):
    ss = float(np.sum((o - o.mean()) ** 2))
    return 1.0 - float(np.sum((o - p) ** 2)) / ss if ss > 0 else float('nan')


def npz_path(args, fold, s):
    if s == '7':
        return os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                            f'resunetpp_v3_loyo{fold}_predictions.npz')
    return os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                        f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')


def select_tile(cands, d, delta):
    rows = [c for c in cands if c['fecha'] == d and c['frac_nieve'] >= MIN_SNOW_FRAC
            and np.isfinite(c['r2_tal_cual'])]
    if not rows:
        return None, 0
    sel, _ = median_pick(rows, key=lambda c: c['r2_tal_cual'])
    return sel, len(rows)


def main():
    ap = argparse.ArgumentParser(description='Seleccion de la figura LOYO (sin dibujar)')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--folds', nargs='+', default=['2021', '2022'])
    ap.add_argument('--out', default='analysis/fig_loyo_seleccion.json')
    ap.add_argument('--fig-npz', default='analysis/fig_mapas_loyo.npz')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()          # noqa: E712
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)
    cache = TileCache(args.masks)
    sc = json.load(open('analysis/e2_scale_checks.json', encoding='utf-8'))['por_fold']

    out = {'descripcion': 'Seleccion (regla fijada de antemano) de la figura de mapas LOYO-E2 en '
                          'terreno ya sobrevolado; ver el docstring de scripts/fig_loyo_seleccion.py',
           'regla': {'semilla': 'R2 sin corregir mediano del fold (e2_scale_checks.json)',
                     'fecha': 'fechas de test con senal de cobertura; orden por T1a = b_d^2; mediana, '
                              'la menor de las dos centrales si n es par',
                     'tesela': f'frac. nieve >= {MIN_SNOW_FRAC}; orden por R2 de la red tal cual; '
                               'mediana con la misma regla',
                     'mascara': 'B: HS > 0.01 m, valido, climatologia disponible, >= 10 pixeles',
                     'correccion_E5': 'aditiva con el regresor ridge (lambda 1) sobre snow_pct, '
                                      'media de todas las teselas de train'},
           'folds': {}}

    fig_fields = {}
    for fold in args.folds:
        r2s = {s: sc[fold]['por_semilla'][s]['R2_raw'] for s in SEEDS}
        seed = sorted(SEEDS, key=lambda s: r2s[s])[1]
        print('=' * 72)
        print(f'Fold {fold}: R2 sin corregir por semilla {r2s} -> semilla {seed}', flush=True)

        clim, _ = build_climatology(geo, fold, args.masks, shape)
        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        trm, _, _ = means_by_date_masks(tr_ids, cache, None)
        model, var_prior, tr_dates = fit_regressor(trm, sig)
        train_years = sorted({d[:4] for d in tr_dates})

        z = np.load(npz_path(args, fold, seed))
        ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
        vals = z['valids'] if 'valids' in z.files else None

        items, acc = [], {}
        for i, t in enumerate(ids):
            t = str(t); d = date_of(t)
            if d not in sig.index or t not in pos:
                continue
            r, c = pos[t]
            cp = clim[r:r + TILE, c:c + TILE]
            tg = tgts[i]
            m = (tg > SNOW_THRESHOLD) & np.isfinite(tg)
            if vals is not None:
                m &= vals[i] > 0.5
            if cp.shape != m.shape:
                continue
            m &= np.isfinite(cp)
            if m.sum() < MIN_VALID_PX:
                continue
            o, p = tg[m].astype(np.float64), preds[i][m].astype(np.float64)
            a = acc.setdefault(d, [0.0, 0.0, 0])
            a[0] += o.sum(); a[1] += p.sum(); a[2] += o.size
            items.append((i, t, d, o, p, float(m.sum()) / (TILE * TILE)))

        dates = sorted(acc)
        n_tot = sum(acc[d][2] for d in dates)
        mse_tot = sum(float(np.sum((p - o) ** 2)) for _, _, _, o, p, _ in items) / n_tot
        prior = {d: float(v) for d, v in zip(dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}
        info = []
        for d in dates:
            mo, mp, n = acc[d][0] / acc[d][2], acc[d][1] / acc[d][2], acc[d][2]
            b = mp - mo
            info.append({'fecha': d, 'T1a_b2': r6(b ** 2), 'sesgo_b_m': r6(b),
                         'T1a_contribucion_pct_MSE': r6(100 * (n / n_tot) * b ** 2 / mse_tot),
                         'media_obs_m': r6(mo), 'media_pred_m': r6(mp), 'media_regresor_m': r6(prior[d]),
                         'delta_E5_m': r6(prior[d] - mp), 'n_pixeles': n})
        delta = {x['fecha']: prior[x['fecha']] - (acc[x['fecha']][1] / acc[x['fecha']][2]) for x in info}

        orac = {x['fecha']: -x['sesgo_b_m'] for x in info}       # media obs - media pred de la fecha
        cands = []
        acc_f = {k: [0.0, 0.0, 0.0, 0] for k in ('tal_cual', 'E5', 'nivel_perfecto_fecha')}
        for i, t, d, o, p, fr in items:
            for k, q in (('tal_cual', p), ('E5', p + delta[d]), ('nivel_perfecto_fecha', p + orac[d])):
                a = acc_f[k]; a[0] += o.sum(); a[1] += (o ** 2).sum(); a[2] += ((o - q) ** 2).sum(); a[3] += o.size
            cands.append({'tile_id': t, 'fecha': d, 'frac_nieve': fr, 'idx': i,
                          'r2_tal_cual': r2(o, p), 'r2_E5': r2(o, p + delta[d]),
                          'r2_nivel_perfecto_fecha': r2(o, p + orac[d]),
                          'r2_mejor_desplazamiento_tesela': r2(o, p + (o.mean() - p.mean())),
                          'desplazamiento_E5_m': delta[d], 'desplazamiento_nivel_perfecto_m': orac[d]})
        fold_r2 = {k: 1 - a[2] / (a[1] - a[0] ** 2 / a[3]) for k, a in acc_f.items()}
        eb = json.load(open('analysis/eB_climatology_v2.json', encoding='utf-8'))[fold]['red_por_semilla'][seed]
        dep = json.load(open('analysis/e5_regresor_dependientes.json', encoding='utf-8'))
        regr_ref = dep['parte1_B']['por_variante']['todas'][fold]['red_por_semilla'][seed]
        ctrl = max(abs(fold_r2['tal_cual'] - eb['r2_raw']), abs(fold_r2['nivel_perfecto_fecha'] - eb['r2_oraculo']),
                   abs(fold_r2['E5'] - regr_ref))

        def pick(pool):
            dsel, ordered = median_pick(pool, key=lambda x: x['T1a_b2'])
            tsel, ncand = select_tile(cands, dsel['fecha'], delta)
            return dsel, ordered, tsel, ncand

        dsel, ordered, tsel, ncand = pick(info)
        res = {'semilla': seed, 'r2_fold_por_semilla': {s: r6(v) for s, v in r2s.items()},
               'npz': os.path.basename(npz_path(args, fold, seed)),
               'regresor': {'fechas_train': tr_dates, 'anyos_train': train_years,
                            'sin_anyo_test': fold not in train_years, 'sin_2024': '2024' not in train_years,
                            'var_prior_LOO': r6(var_prior)},
               'fechas_ordenadas_por_T1a': [x for x in ordered],
               'fecha_elegida': dsel['fecha'], 'T1a_fecha_elegida': dsel['T1a_b2'],
               'n_teselas_candidatas': ncand,
               'r2_fold_semilla_mascara_B': {k: r6(v) for k, v in fold_r2.items()},
               'r2_fold_fuentes': {'tal_cual': 'analysis/eB_climatology_v2.json (r2_raw)',
                                   'E5': "analysis/e5_regresor_dependientes.json (parte1_B 'todas')",
                                   'nivel_perfecto_fecha': 'analysis/eB_climatology_v2.json (r2_oraculo)'},
               'control_r2_fold_max_dif': ctrl}
        if tsel:
            res['tesela_elegida'] = {k: (r6(v) if isinstance(v, float) else v)
                                     for k, v in tsel.items() if k != 'idx'}
        if dsel['fecha'] == LOW_VAR_DATE:
            alt = [x for x in info if x['fecha'] != LOW_VAR_DATE]
            d2, _, t2, n2 = pick(alt)
            res['alternativa_sin_' + LOW_VAR_DATE] = {
                'fecha_elegida': d2['fecha'], 'T1a_fecha_elegida': d2['T1a_b2'], 'n_teselas_candidatas': n2,
                'tesela_elegida': {k: (r6(v) if isinstance(v, float) else v) for k, v in t2.items() if k != 'idx'} if t2 else None}
        out['folds'][fold] = res

        print(f"  R2 del fold (semilla {seed}, mascara B): tal cual {fold_r2['tal_cual']:.4f}  "
              f"E5 {fold_r2['E5']:.4f}  nivel perfecto {fold_r2['nivel_perfecto_fecha']:.4f}  "
              f"(control {ctrl:.1e})")
        print(f"  regresor: train {train_years} ({len(tr_dates)} fechas); sin el anyo de test: "
              f"{res['regresor']['sin_anyo_test']}, sin 2024: {res['regresor']['sin_2024']}")
        print('  fechas por T1a (b^2):')
        for x in ordered:
            mark = '  <- elegida' if x['fecha'] == dsel['fecha'] else ''
            print(f"    {x['fecha']}  b {x['sesgo_b_m']:+.3f} m  T1a {x['T1a_b2']:.4f}  "
                  f"({x['T1a_contribucion_pct_MSE']:.1f} % del MSE)  delta E5 {x['delta_E5_m']:+.3f} m{mark}")
        if tsel:
            print(f"  tesela ({ncand} candidatas con >= 50 % de nieve): {tsel['tile_id']}  "
                  f"nieve {100 * tsel['frac_nieve']:.0f} %  R2 tal cual {tsel['r2_tal_cual']:.3f}  "
                  f"E5 {tsel['r2_E5']:.3f}  nivel perfecto de la fecha {tsel['r2_nivel_perfecto_fecha']:.3f}  "
                  f"(mejor desplazamiento de la tesela {tsel['r2_mejor_desplazamiento_tesela']:.3f})")
            print(f"  desplazamientos: E5 {tsel['desplazamiento_E5_m']:+.3f} m, nivel perfecto "
                  f"{tsel['desplazamiento_nivel_perfecto_m']:+.3f} m")
            i = tsel['idx']
            tg, pr = tgts[i].astype(np.float32), preds[i].astype(np.float32)
            r, c = pos[tsel['tile_id']]
            cp = clim[r:r + TILE, c:c + TILE]
            mk = (tg > SNOW_THRESHOLD) & np.isfinite(tg) & np.isfinite(cp)
            if vals is not None:
                mk &= vals[i] > 0.5
            fig_fields[fold] = {'observado': np.where(mk, tg, np.nan), 'red': np.where(mk, pr, np.nan), 'mask': mk}
        if 'alternativa_sin_' + LOW_VAR_DATE in res:
            a = res['alternativa_sin_' + LOW_VAR_DATE]
            print(f"  [la fecha elegida es {LOW_VAR_DATE}] sin ella: {a['fecha_elegida']}, {a['tesela_elegida']}")
        del z, ids, preds, tgts, vals, items

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    flat = {}
    for fold, ff in fig_fields.items():
        for k, v in ff.items():
            flat[f'{k}_{fold}'] = v
    np.savez_compressed(args.fig_npz, **flat)
    print(f'\nGuardado en: {args.out} y {args.fig_npz}')


if __name__ == '__main__':
    main()
