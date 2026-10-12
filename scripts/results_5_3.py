r"""
Cifras canonicas para Resultados 5.3 (sin GPU)
==============================================

Reune en analysis/cifras_canonicas_5_3.json las cifras de la seccion 5.3, cada
una con su fichero de origen; media [min-max] de las 3 semillas donde las hay.

    1_E3        doble cabeza frente a base, LOYO-E2, semilla 7: R2 por fold y
                medio, folds que mejoran, fraccion de escala (T1+T2)/MSE.
    2_E4b       fold 2025, brazos base / dual / dualscale, train submuestreado
                (2284 teselas) y completo (3866): R2 y SPAEF SIN corregir,
                fraccion de escala y solapes de los rangos entre semillas.
    3_E5        regresor corregido (media de todas las teselas) sobre LOYO-E2:
                por fold y semilla R2 sin corregir y corregido, ganancia, MAE
                del regresor y de la red al estimar la media de cada fecha, y
                si la regla acierta.
    4_A         experimento A v2 (regresor corregido, ruido 0,10 m, muestreo
                libre): curvas, P10/P50/P90 por campana y N90 de los tres
                metodos, copiados de analysis/e5_regresor_dependientes.json;
                referencias (sin corregir, solo satelite, nivel perfecto)
                acumuladas y por campana, recalculadas aqui.

Que lee cada bloque
-------------------
    1, 3        solo JSON ya guardados.
    2           JSON ya guardados; la fraccion de escala del train COMPLETO no
                estaba calculada: se obtiene con analyse() de
                analysis/e1_error_decomposition.py sobre los .npz (--e4b-dir),
                igual que analysis/e4b_decomposition_s*.json para el
                submuestreado (que se recalcula tambien como control).
    4           curvas, campanas y N90: JSON. Referencias por campana: no se
                guardaron en ningun JSON (point_measurements.py solo las
                imprimia), asi que se recalculan con las mismas mascaras, los
                mismos .npz y el mismo regresor que la parte 2 de
                scripts/e5_regressor_dependents.py, sin la simulacion de
                Monte Carlo (las referencias no dependen de ella). Control: los
                valores acumulados deben coincidir con los del JSON.

Sin --e4b-dir no se calcula la fraccion de escala del train completo; sin
--seed7-dir/--seeds-dir no se calculan las referencias por campana de A. En
ambos casos el campo queda a null con un aviso.

Las cifras se guardan con 6 decimales (porcentajes con 4) para que al citarlas
con 3 (o 1) no haya doble redondeo. Los metrics.json de origen ya vienen con 4.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/results_5_3.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas ^
        --e4b-dir F:\uniovi\mapunet\experimentos\E4b_loyo_tres_brazos ^
        --slope-npz <parte1> <parte2>
"""

import argparse
import glob
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'analysis'))

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
ARMS = ['base', 'dual', 'dualscale']
DATASETS = {'sub': '', 'full': '_full'}
N_TRAIN = {'sub': 2284, 'full': 3866}
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'

# ficheros de origen
E3_BASE = 'results/norm_v3/loyo/resunetpp_v3_loyo{f}/resunetpp_v3_loyo{f}_metrics.json'
E3_DUAL = 'results/norm_v3/loyo_dual/resunetpp_dual_loyo{f}/resunetpp_dual_loyo{f}_metrics.json'
E3_DEC_BASE = 'analysis/e2_decomposition.json'
E3_DEC_DUAL = 'analysis/e3_dual_decomposition.json'
E4B_MET = 'results/norm_v3/loyo_e4b_{arm}/e4b_{arm}{ds}_loyo2025_s{s}/e4b_{arm}{ds}_loyo2025_s{s}_metrics.json'
E4B_DEC = 'analysis/e4b_decomposition_s{s}.json'
E4B_NPZ = os.path.join('loyo_e4b_{arm}', 'e4b_{arm}{ds}_loyo2025_s{s}',
                       'e4b_{arm}{ds}_loyo2025_s{s}_predictions.npz')
E5_JSON = 'analysis/e5_regresor_todas_teselas.json'
A_JSON = 'analysis/e5_regresor_dependientes.json'
VAR = 'todas'                  # regresor corregido
A_KEY = '{pat}|todos|0.1'      # muestreo libre, ruido 0,10 m


def load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def mmm(vals, nd=6):
    v = np.asarray(vals, float)
    return {'media': round(float(v.mean()), nd), 'min': round(float(v.min()), nd),
            'max': round(float(v.max()), nd), 'por_semilla': [round(float(x), nd) for x in v]}


def solapan(a, b):
    return bool(a['min'] <= b['max'] and b['min'] <= a['max'])


def fold_row(rows, fold):
    """Etiquetas de e2/e3: 'loyo2023 (-38%)', 'dual2023 (-38%)'."""
    return next(r for r in rows if fold in r['label'].split()[0])


# ======================================================================
# 1. E3
# ======================================================================
def block_e3():
    dec_b, dec_d = load(E3_DEC_BASE), load(E3_DEC_DUAL)
    pf, ctrl = {}, []
    for f in FOLDS:
        mb, md = load(E3_BASE.format(f=f)), load(E3_DUAL.format(f=f))
        db, dd = fold_row(dec_b, f), fold_row(dec_d, f)
        ctrl += [abs(mb['R2'] - db['R2_raw']), abs(md['R2'] - dd['R2_raw'])]
        pf[f] = {'n_teselas_test': mb['SPAEF_n_tiles'],
                 'R2_base': round(db['R2_raw'], 6), 'R2_doble_cabeza': round(dd['R2_raw'], 6),
                 'dif_R2': round(dd['R2_raw'] - db['R2_raw'], 6), 'mejora': bool(dd['R2_raw'] > db['R2_raw']),
                 'SPAEF_base_sin_corregir': round(mb['SPAEF'], 6),
                 'SPAEF_doble_cabeza_sin_corregir': round(md['SPAEF'], 6),
                 'escala_pct_base': round(db['scale_pct'], 4),
                 'escala_pct_doble_cabeza': round(dd['scale_pct'], 4),
                 'dif_escala_pct': round(dd['scale_pct'] - db['scale_pct'], 4)}
    return {
        'fuentes': {'SPAEF (y control del R2)': [E3_BASE.replace('{f}', '<fold>'), E3_DUAL.replace('{f}', '<fold>')],
                    'R2 (R2_raw, float64) y fraccion de escala': [E3_DEC_BASE + ' (base, s7)',
                                                                  E3_DEC_DUAL + ' (doble cabeza, s7)']},
        'definiciones': {'R2': 'acumulado sobre pixeles con nieve del test, sin corregir el nivel',
                         'escala': '(T1+T2)/MSE en %, ponderada por pixeles (analyse() de '
                                   'analysis/e1_error_decomposition.py)'},
        'semilla': 7, 'por_fold': pf,
        'media_4_folds': {'R2_base': round(float(np.mean([pf[f]['R2_base'] for f in FOLDS])), 6),
                          'R2_doble_cabeza': round(float(np.mean([pf[f]['R2_doble_cabeza'] for f in FOLDS])), 6),
                          'escala_pct_base': round(float(np.mean([pf[f]['escala_pct_base'] for f in FOLDS])), 4),
                          'escala_pct_doble_cabeza': round(float(np.mean([pf[f]['escala_pct_doble_cabeza'] for f in FOLDS])), 4)},
        'folds_que_mejoran_R2': [f for f in FOLDS if pf[f]['mejora']],
        'folds_donde_sube_la_escala': [f for f in FOLDS if pf[f]['dif_escala_pct'] > 0],
        'control_R2_metrics_vs_descomposicion_max_dif': float(max(ctrl))}


# ======================================================================
# 2. E4b
# ======================================================================
def scale_from_npz(e4b_dir, arm, ds, s):
    from e1_error_decomposition import analyse, load_npz
    p = os.path.join(e4b_dir, E4B_NPZ.format(arm=arm, ds=DATASETS[ds], s=s))
    if not os.path.exists(p):
        cand = [x for x in glob.glob(os.path.join(os.path.dirname(p), '*_predictions.npz'))
                if '_last' not in os.path.basename(x)]
        if not cand:
            print(f'  [aviso] falta {p}')
            return None
        p = cand[0]
    print(f'  descomposicion {os.path.basename(p)} ...', flush=True)
    r = analyse(load_npz(p), 'snow', f'{arm}{DATASETS[ds]} s{s}')
    return {'escala_pct': r['scale_pct'], 'R2_raw': r['R2_raw'], 'n_teselas': r['n_tiles_used'],
            'reconstruccion_ok': r['reconstruction_ok']}


def block_e4b(e4b_dir):
    dec = {s: load(E4B_DEC.format(s=s)) for s in SEEDS}
    out, ctrl_r2, ctrl_dec = {}, [], []
    for ds in DATASETS:
        for arm in ARMS:
            ms = [load(E4B_MET.format(arm=arm, ds=DATASETS[ds], s=s)) for s in SEEDS]
            row = {'n_teselas_train': N_TRAIN[ds], 'n_teselas_test': ms[0]['SPAEF_n_tiles'],
                   'R2_sin_corregir': mmm([m['R2'] for m in ms]),
                   'R2_fuente': 'metrics.json (4 decimales)',
                   'SPAEF_sin_corregir': mmm([m['SPAEF'] for m in ms])}
            esc_json = None
            if ds == 'sub':
                rr = [next(r for r in dec[s] if r['label'] == f'{arm} s{s}') for s in SEEDS]
                esc_json = [r['scale_pct'] for r in rr]
                ctrl_r2 += [abs(r['R2_raw'] - m['R2']) for r, m in zip(rr, ms)]
                row['escala_pct'] = mmm(esc_json, 4)
                row['R2_sin_corregir'] = mmm([r['R2_raw'] for r in rr])
                row['R2_fuente'] = E4B_DEC.replace('{s}', '<semilla>') + ' (R2_raw, float64)'
                row['escala_fuente'] = E4B_DEC.replace('{s}', '<semilla>')
            if e4b_dir:
                calc = [scale_from_npz(e4b_dir, arm, ds, s) for s in SEEDS]
                if all(c is not None for c in calc):
                    ctrl_r2 += [abs(c['R2_raw'] - m['R2']) for c, m in zip(calc, ms)]
                    if ds == 'sub':
                        ctrl_dec += [abs(c['escala_pct'] - e) for c, e in zip(calc, esc_json)]
                    else:
                        row['escala_pct'] = mmm([c['escala_pct'] for c in calc], 4)
                        row['R2_sin_corregir'] = mmm([c['R2_raw'] for c in calc])
                        row['R2_fuente'] = 'R2_raw (float64) de la descomposicion calculada aqui'
                        row['escala_fuente'] = ('analyse() de analysis/e1_error_decomposition.py sobre '
                                                + E4B_NPZ.replace('{s}', '<semilla>').format(
                                                    arm=arm, ds=DATASETS[ds], s='<semilla>')
                                                + ' (calculada aqui)')
            if 'escala_pct' not in row:
                row['escala_pct'] = None
                row['escala_fuente'] = 'no calculada: falta --e4b-dir'
            out[f'{arm}|{ds}'] = row

    solapes = {}
    for met in ('R2_sin_corregir', 'SPAEF_sin_corregir', 'escala_pct'):
        L = []
        for ds in DATASETS:
            for i, a in enumerate(ARMS):
                for b in ARMS[i + 1:]:
                    x, y = out[f'{a}|{ds}'][met], out[f'{b}|{ds}'][met]
                    if x and y:
                        L.append({'par': f'{a} - {b} ({ds})', 'solapan': solapan(x, y)})
        for arm in ARMS:
            x, y = out[f'{arm}|sub'][met], out[f'{arm}|full'][met]
            if x and y:
                L.append({'par': f'{arm}: sub - full', 'solapan': solapan(x, y)})
        solapes[met] = L
    return {
        'fuentes': {'SPAEF (y R2 si no hay descomposicion)': E4B_MET.replace('{s}', '<semilla>').replace('{arm}', '<brazo>')
                                        .replace('{ds}', "<'' | '_full'>"),
                    'fraccion de escala': 'ver escala_fuente en cada brazo'},
        'definiciones': {'datos': 'fold 2025; 23 fechas en el CSV = 13 de train (2021-2023) + 5 de '
                                  'val (2024) + 5 de test (2025); test: 220 teselas y 5 fechas; '
                                  'sub = train submuestreado (2284 teselas, '
                                  'dataset_e4b_loyo2025.csv), full = completo (3866, '
                                  'dataset_e4b_full_loyo2025.csv)',
                         'SPAEF': 'media por tesela, SIEMPRE sin corregir el nivel (hallazgo 13)',
                         'R2': 'acumulado sobre pixeles con nieve, sin corregir; R2_raw en float64 de '
                               'la descomposicion (como la tabla LOYO de 5.1); el de metrics.json '
                               'difiere en <2e-4 y queda como control',
                         'solapan': 'los intervalos [min, max] de las 3 semillas se tocan'},
        'por_brazo': out, 'solapes': solapes,
        'control_R2_max_dif': float(max(ctrl_r2)) if ctrl_r2 else None,
        'control_escala_sub_recalculada_vs_json_max_dif': float(max(ctrl_dec)) if ctrl_dec else None}


# ======================================================================
# 3. E5 (regresor corregido) sobre LOYO-E2
# ======================================================================
def block_e5():
    j = load(E5_JSON)
    pf, todos = {}, []
    for f in FOLDS:
        rows = {}
        for r in j['punto2_e5'][f]:
            s = r['experimento'].rsplit('_s', 1)[1] if '_s' in r['experimento'].split('loyo')[-1] else '7'
            v = r['variantes'][VAR]
            rows[s] = {'experimento': r['experimento'], 'n_test_fechas': r['n_test_fechas'],
                       'R2_sin_corregir': round(r['r2_sin_recal'], 6),
                       'R2_corregido': round(v['r2_pred_add_cobertura'], 6),
                       'ganancia': round(v['ganancia_r2_principal'], 6),
                       'R2_nivel_perfecto': round(r['r2_oraculo_add'], 6),
                       'MAE_regresor_m': round(v['mae_media_cobertura'], 6),
                       'MAE_red_m': round(r['mae_media_red'], 6),
                       'regla_predice': v['criterio_predice'], 'regla_acierta': bool(v['criterio_acierta']),
                       'mejora': bool(v['ganancia_r2_principal'] > 0)}
            todos.append((s, rows[s]))
        pf[f] = {s: rows[s] for s in SEEDS if s in rows}
    g = [r['ganancia'] for _, r in todos]
    return {
        'fuente': E5_JSON + " (punto2_e5, variante 'todas')",
        'definiciones': {'corregido': 'nivel de cada fecha = prediccion del ridge sobre la cobertura '
                                      'satelital (snow_pct), ajustado con la media de TODAS las '
                                      'teselas de cada fecha de train',
                         'ganancia': 'R2 corregido - R2 sin corregir',
                         'MAE': 'error absoluto medio al estimar la HS media de las fechas de test',
                         'regla': "predice 'ayuda' si MAE del regresor < MAE de la red",
                         'aviso': 'solo fechas de test admisibles (n_test_fechas): en 2021 son 4 de 6, '
                                  'asi que el R2 sin corregir de 2021 no es el de la tabla LOYO'},
        'por_fold': pf,
        'resumen': {'n_modelos': len(todos),
                    'ganancia_media': round(float(np.mean(g)), 6),
                    'ganancia_min_max': [round(float(min(g)), 6), round(float(max(g)), 6)],
                    'modelos_que_mejoran': sum(r['mejora'] for _, r in todos),
                    'aciertos_regla': sum(r['regla_acierta'] for _, r in todos),
                    'aciertos_regla_semilla_7': sum(r['regla_acierta'] for s, r in todos if s == '7'),
                    'aciertos_regla_semillas_42_123': sum(r['regla_acierta'] for s, r in todos if s != '7')}}


# ======================================================================
# 4. A: referencias por campana (recalculo sin Monte Carlo)
# ======================================================================
def a_references(args):
    if args.slope_npz:
        sys.modules.setdefault('rasterio', types.ModuleType('rasterio'))
    import point_measurements as pm
    from climatology_baseline import build_climatology
    from recalibration_test import date_of, ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX
    from e5_regressor_allmeans import TileCache, means_by_date_masks, fit_regressor

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()          # noqa: E712
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    cache = TileCache(args.masks)
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)
    if args.slope_npz:
        slope = np.vstack([np.load(p)['slope'] for p in args.slope_npz])
    else:
        slope = pm.slope_map(args.dem or pm.DEM_PATH)

    per_run = {'climatologia': [], 'red': []}
    per_date = {'climatologia': [], 'red': []}
    for fold in args.folds:
        print(f'A, fold {fold} ...', flush=True)
        clim, _ = build_climatology(geo, fold, args.masks, shape)
        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        trm, _, _ = means_by_date_masks(tr_ids, cache, None)
        model, _, _ = fit_regressor(trm, sig)
        paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                   f'resunetpp_v3_loyo{fold}_predictions.npz')}
        for s in ('42', '123'):
            paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                    f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')
        clim_hecha = False
        for s in SEEDS:
            if not os.path.exists(paths[s]):
                print(f'  [aviso] falta la semilla {s}: {paths[s]}'); continue
            z = np.load(paths[s])
            ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
            vals = z['valids'] if 'valids' in z.files else None
            # mismo bucle que scripts/e5_regressor_dependents.py (parte 2)
            rng = np.random.default_rng(int(s))
            net, cli = {}, {}
            so = so2 = 0.0; n_tot = 0
            for i, t in enumerate(ids):
                t = str(t); d = date_of(t)
                if d not in sig.index or t not in pos:
                    continue
                r, c = pos[t]
                cp = clim[r:r + TILE, c:c + TILE]
                sl = slope[r:r + TILE, c:c + TILE]
                tg = tgts[i]
                m = (tg > SNOW_THRESHOLD) & np.isfinite(tg) & np.isfinite(cp) & np.isfinite(sl)
                if vals is not None:
                    m &= vals[i] > 0.5
                if m.sum() < MIN_VALID_PX:
                    continue
                o = tg[m].astype(np.float64)
                so += o.sum(); so2 += (o ** 2).sum(); n_tot += o.size
                net.setdefault(d, pm.DateStats()).add(o, preds[i][m].astype(np.float64), sl[m], rng)
                if not clim_hecha:
                    cli.setdefault(d, pm.DateStats()).add(o, cp[m].astype(np.float64), sl[m], rng)
            for st in list(net.values()) + list(cli.values()):
                st.finish()
            dates = sorted(net)
            prior = {d: float(v) for d, v in zip(
                dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}
            todo = [('red', net, s)] + ([] if clim_hecha else [('climatologia', cli, '-')])
            for pat, stats, sem in todo:
                zero = {d: np.zeros(1) for d in stats}
                orac = {d: np.array([st.s1 / st.n]) for d, st in stats.items()}
                sat = {d: np.array([prior[d] - st.mean_p]) for d, st in stats.items()}
                per_run[pat].append({
                    'fold': fold, 'semilla': sem,
                    'sin': float(pm.r2_global(stats, zero, so, so2, n_tot)[0]),
                    'satelite': float(pm.r2_global(stats, sat, so, so2, n_tot)[0]),
                    'oraculo': float(pm.r2_global(stats, orac, so, so2, n_tot)[0])})
                for d in sorted(stats):
                    st = stats[d]
                    per_date[pat].append({'fold': fold, 'semilla': sem, 'fecha': d,
                                          'sin': float(st.r2(0.0)),
                                          'satelite': float(st.r2(prior[d] - st.mean_p)),
                                          'oraculo': float(st.r2(st.s1 / st.n))})
            clim_hecha = True
            print(f'  semilla {s}: {len(dates)} fechas', flush=True)
            del z, ids, preds, tgts, vals, net, cli
    return per_run, per_date, pm.REPS


def block_a(args):
    dep = load(A_JSON)['parte2_A']
    N = dep['N']
    metodos = dep['metodos']
    R = dep['resumen'][VAR]
    refs = None
    if args.seed7_dir and args.seeds_dir:
        refs = a_references(args)
    out = {}
    ctrl = []
    for pat in ('climatologia', 'red'):
        r = R[A_KEY.format(pat=pat)]
        blk = {
            'R2_acumulado': {m: {str(n): round(v, 6) for n, v in zip(N, r['curvas'][m])} for m in metodos},
            'campanas_P10_P50_P90': {m: {str(n): [round(x, 6) for x in q]
                                         for n, q in zip(N, r['campanas_p10_p50_p90'][m])} for m in metodos},
            'N90': r['N90'],
            'referencias': {'acumulado': {'sin_corregir': round(r['sin'], 6),
                                          'solo_satelite': round(r['satelite'], 6),
                                          'nivel_perfecto': round(r['oraculo'], 6)},
                            'campanas_P10_P50_P90': None}}
        if refs is not None:
            per_run, per_date, reps = refs
            for k_json, k in (('sin', 'sin'), ('satelite', 'satelite'), ('oraculo', 'oraculo')):
                ctrl.append(abs(float(np.mean([x[k] for x in per_run[pat]])) - r[k_json]))
            camp = {}
            for k, name in (('sin', 'sin_corregir'), ('satelite', 'solo_satelite'), ('oraculo', 'nivel_perfecto')):
                v = np.array([x[k] for x in per_date[pat]])
                camp[name] = [round(float(q), 6) for q in np.percentile(np.repeat(v, reps), [10, 50, 90])]
                camp[name + '_sin_replicar'] = [round(float(q), 6) for q in np.percentile(v, [10, 50, 90])]
            blk['referencias']['campanas_P10_P50_P90'] = camp
            blk['referencias']['n_campanas'] = len(per_date[pat])
            blk['referencias']['por_campana'] = [{k: (round(v, 6) if isinstance(v, float) else v)
                                                  for k, v in x.items()} for x in per_date[pat]]
            blk['referencias']['por_fold_y_semilla'] = [{k: (round(v, 6) if isinstance(v, float) else v)
                                                         for k, v in x.items()} for x in per_run[pat]]
        out[pat] = blk
    res = {
        'fuentes': {'curvas, campanas, N90, referencias acumuladas':
                        A_JSON + f" (parte2_A.resumen['{VAR}']['<patron>|todos|0.1'])",
                    'referencias por campana': 'recalculadas aqui (mismo bucle que la parte 2 de '
                                               'scripts/e5_regressor_dependents.py, sin Monte Carlo)'},
        'definiciones': {
            'configuracion': 'regresor corregido (media de todas las teselas), ruido 0,10 m, '
                             'muestreo libre (pool todos), 200 repeticiones por fecha, folds LOYO-E2; '
                             'red: 3 semillas x 4 folds; climatologia: 4 folds',
            'metodos': {'ingenuo': 'media de los N puntos - media predicha',
                        'patron': 'media de (medida - patron) en los N puntos',
                        'patron+sat': 'patron combinado con el regresor satelital por varianza inversa'},
            'R2_acumulado': 'media sobre folds (y semillas) del R2 acumulado de cada fold',
            'campanas': 'R2 de cada fecha corregida; P10/P50/P90 sobre todas las fechas, folds, '
                        'semillas y repeticiones',
            'referencias_por_campana': 'una entrada por fecha repetida 200 veces (mismo peso que las '
                                       'campanas corregidas, como en point_measurements.py); '
                                       '*_sin_replicar: una entrada por fecha',
            'N90': 'primer N con R2 acumulado - sin >= 0,9 (nivel perfecto - sin); None = >100'},
        'N': N, 'metodos': metodos, 'por_patron': out,
        'satelite_frente_a_patron': dep['satelite_frente_a_patron'][VAR],
        'var_prior_por_fold': {f: v[VAR] for f, v in dep['var_prior_por_fold'].items()},
        'control_referencias_acumuladas_vs_json_max_dif': float(max(ctrl)) if ctrl else None}
    if refs is None:
        res['aviso'] = 'referencias por campana no calculadas: faltan --seed7-dir y --seeds-dir'
    return res


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description='Cifras canonicas de Resultados 5.3')
    ap.add_argument('--seed7-dir', default=None)
    ap.add_argument('--seeds-dir', default=None)
    ap.add_argument('--e4b-dir', default=None)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--dem', default=None)
    ap.add_argument('--slope-npz', nargs='+', default=None)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/cifras_canonicas_5_3.json')
    args = ap.parse_args()
    t0 = time.time()

    out = {'descripcion': 'Cifras canonicas de Resultados 5.3; cada bloque lleva su fuente. '
                          'Media, minimo y maximo de las semillas 7, 42 y 123 (por_semilla en ese '
                          'orden) donde las hay. SPAEF siempre sin corregir el nivel (hallazgo 13).'}
    out['1_E3_doble_cabeza'] = block_e3()
    out['2_E4b'] = block_e4b(args.e4b_dir)
    out['3_E5_regresor_corregido'] = block_e5()
    out['4_A_medidas_puntuales'] = block_a(args)

    # ---------------------------------------------------------- impresion
    e3 = out['1_E3_doble_cabeza']
    print('=' * 72); print('1. E3 (semilla 7)   R2 base  R2 dual  escala base  escala dual')
    for f, r in e3['por_fold'].items():
        print(f"   {f}            {r['R2_base']:>7.3f}  {r['R2_doble_cabeza']:>7.3f}"
              f"  {r['escala_pct_base']:>10.1f}  {r['escala_pct_doble_cabeza']:>10.1f}")
    m = e3['media_4_folds']
    print(f"   media           {m['R2_base']:>7.3f}  {m['R2_doble_cabeza']:>7.3f}"
          f"  {m['escala_pct_base']:>10.1f}  {m['escala_pct_doble_cabeza']:>10.1f}")
    print(f"   mejoran: {e3['folds_que_mejoran_R2']}  | control {e3['control_R2_metrics_vs_descomposicion_max_dif']:.1e}")

    e4 = out['2_E4b']
    print('=' * 72); print('2. E4b (fold 2025)  R2 [min-max]            SPAEF sin corr.          escala %')
    for k, r in e4['por_brazo'].items():
        a, b, c = r['R2_sin_corregir'], r['SPAEF_sin_corregir'], r['escala_pct']
        cs = f"{c['media']:.1f} [{c['min']:.1f}-{c['max']:.1f}]" if c else '-'
        print(f"   {k:<16}{a['media']:.3f} [{a['min']:.3f}-{a['max']:.3f}]   "
              f"{b['media']:.3f} [{b['min']:.3f}-{b['max']:.3f}]   {cs}")
    for met, L in e4['solapes'].items():
        print(f"   solapes {met}: " + '; '.join(f"{x['par']} {'si' if x['solapan'] else 'NO'}" for x in L))
    print(f"   control R2 {e4['control_R2_max_dif']} | escala sub {e4['control_escala_sub_recalculada_vs_json_max_dif']}")

    e5 = out['3_E5_regresor_corregido']
    print('=' * 72); print('3. E5   fold sem  sin     corr    gan     MAEreg  MAEred  regla')
    for f, rs in e5['por_fold'].items():
        for s, r in rs.items():
            print(f"        {f} {s:>4} {r['R2_sin_corregir']:>6.3f} {r['R2_corregido']:>7.3f} {r['ganancia']:>+7.3f}"
                  f" {r['MAE_regresor_m']:>7.3f} {r['MAE_red_m']:>7.3f}  {r['regla_predice']:<9} "
                  f"{'acierta' if r['regla_acierta'] else 'falla'}")
    print(f"   resumen: {e5['resumen']}")

    A = out['4_A_medidas_puntuales']
    print('=' * 72); print('4. A')
    for pat, b in A['por_patron'].items():
        print(f'   {pat}: N90 {b["N90"]} | acumulado {b["referencias"]["acumulado"]}')
        print(f"   {'N':>5}" + ''.join(f'{m:>12}' for m in A['metodos']) + '   P10/P50/P90 patron')
        for n in A['N']:
            q = b['campanas_P10_P50_P90']['patron'][str(n)]
            print(f'   {n:>5}' + ''.join(f"{b['R2_acumulado'][m][str(n)]:>12.3f}" for m in A['metodos'])
                  + f'   {q[0]:>7.3f} {q[1]:>6.3f} {q[2]:>6.3f}')
        if b['referencias']['campanas_P10_P50_P90']:
            for k, q in b['referencias']['campanas_P10_P50_P90'].items():
                print(f'   {k:<30}{q[0]:>7.3f} {q[1]:>6.3f} {q[2]:>6.3f}')
    print(f"   control referencias acumuladas: {A['control_referencias_acumuladas_vs_json_max_dif']}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'\nGuardado en: {args.out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
