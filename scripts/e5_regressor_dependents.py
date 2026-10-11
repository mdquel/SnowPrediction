r"""
Todo lo que usa el regresor de E5, recalculado con la media de TODAS las teselas
================================================================================

Continuacion de scripts/e5_regressor_allmeans.py (11-oct-2026). El regresor de
E5 (ridge sobre la cobertura satelital) se ajusta con la HS media de cada fecha
de train, y recalibration_test.mean_from_masks la calculaba con las 60
PRIMERAS teselas (franja norte; sobrestima +0.146 m de media). Lo usan:

    E5   recalibration_test.py      (E2/E2b: ya en e5_regresor_todas_teselas.json)
                                    (E4b fold 2025: AQUI, parte 3)
    B    climatology_baseline.py    version "nivel del regresor"  (parte 1)
    A    point_measurements.py      prior y peso del satelite      (parte 2)

Ningun otro script del repo importa mean_from_masks ni el ridge
(patch_recalibration_seeds.py solo parcheaba el codigo de E5).

Para cada resultado se calculan las DOS variantes en la misma pasada:
    '60'     media de train con las 60 primeras teselas (lo actual)
    'todas'  media de train con todas las teselas (corregido)
La variante '60' debe reproducir exactamente los JSON actuales:
    B   analysis/eB_climatology_v2.json
    A   analysis/eA_point_measurements_v2.json  (TODAS las claves del resumen)
    E5  analysis/e5_recalibration.json (base) y e5_recalibration_dual.json (dual),
        solo E4b submuestreado; el full y el brazo dualscale no tenian JSON.
Los generadores aleatorios de A se crean igual que en point_measurements.py.

Partes
------
1. B: R2 de red (3 semillas) y climatologia con el nivel del regresor, por
   fold y media de folds; tambien tal cual y nivel perfecto (no cambian).
2. A completo: curvas de los tres metodos para todos los N, N90, P10/P50/P90
   por campana, sensibilidad a pendiente (25/30/35) y ruido (0/0.10/0.20), y
   desde que N deja de ayudar el satelite (patron+sat frente a patron).
3. E5 sobre E4b, fold 2025, brazos base/dual/dualscale, submuestreado (2284
   teselas) y completo (3866): R2 sin corregir, corregido, oraculo y fraccion
   del techo.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/e5_regressor_dependents.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas ^
        --e4b-dir F:\uniovi\mapunet\experimentos\E4b_loyo_tres_brazos ^
        --slope-npz <parte1> <parte2>

    --slope-npz evita rasterio (bloqueado en el PC del trabajo); sin el se
    calcula la pendiente desde el DEM como en A.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_test import (date_of, ridge_fit, ridge_predict, load_npz,   # noqa: E402
                                SNOW_THRESHOLD, MIN_VALID_PX)
from e5_regressor_allmeans import (TileCache, means_by_date_masks,            # noqa: E402
                                   fit_regressor, e5_model, VARIANTS)

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
E4B_CSV = {'sub': 'dataset_v4_ms_sx200/loyo_e4b/dataset_e4b_loyo2025.csv',
           'full': 'dataset_v4_ms_sx200/loyo_e4b/dataset_e4b_full_loyo2025.csv'}
E4B_NPZ = {'sub': os.path.join('loyo_e4b_{arm}', 'e4b_{arm}_loyo2025_s*', '*_predictions.npz'),
           'full': os.path.join('loyo_e4b_{arm}', 'e4b_{arm}_full_loyo2025_s*', '*_predictions.npz')}
ARMS = ['base', 'dual', 'dualscale']


# ----------------------------------------------------------------------
# B - replica de climatology_baseline.evaluate_seed con varios modelos
# ----------------------------------------------------------------------
def new_acc():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0}


def add(acc, o, p):
    acc['n'] += o.size
    acc['so'] += o.sum()
    acc['so2'] += (o ** 2).sum()
    acc['sr'] += ((o - p) ** 2).sum()


def r2acc(acc):
    ss = acc['so2'] - acc['so'] ** 2 / acc['n']
    return 1.0 - acc['sr'] / ss if ss > 0 else float('nan')


def evaluate_seed_B(ids, preds, tgts, vals, pos, clim, sig, models):
    """climatology_baseline.py l. 129-194, linea a linea, con un modelo por variante."""
    dates = np.array([date_of(t) for t in ids])
    keep = np.where(np.isin(dates, list(sig.index)))[0]
    items = []
    base_px = comun_px = 0
    for i in keep:
        t = str(ids[i])
        if t not in pos:
            continue
        r, c = pos[t]
        cp = clim[r:r + TILE, c:c + TILE]
        tg = tgts[i]
        m = (tg > SNOW_THRESHOLD) & np.isfinite(tg)
        if vals is not None:
            m &= vals[i] > 0.5
        base_px += int(m.sum())
        if cp.shape != m.shape:
            continue
        m &= np.isfinite(cp)
        comun_px += int(m.sum())
        if m.sum() >= MIN_VALID_PX:
            items.append((i, dates[i], r, c, m))

    acc_d = {}
    for i, d, r, c, m in items:
        cp = clim[r:r + TILE, c:c + TILE]
        s = acc_d.setdefault(d, [0.0, 0.0, 0.0, 0])
        s[0] += float(tgts[i][m].sum()); s[1] += float(preds[i][m].sum())
        s[2] += float(cp[m].sum()); s[3] += int(m.sum())
    real = {d: v[0] / v[3] for d, v in acc_d.items()}
    mred = {d: v[1] / v[3] for d, v in acc_d.items()}
    mcli = {d: v[2] / v[3] for d, v in acc_d.items()}
    tdates = sorted(real)
    regr = {var: {d: float(v) for d, v in zip(
        tdates, ridge_predict(mdl, sig.loc[tdates, ['snow_pct']].to_numpy(float)))}
        for var, mdl in models.items()}

    vers = ['raw', 'orac'] + [f'regr_{v}' for v in models]
    accs = {(p, v): new_acc() for p in ('red', 'clim') for v in vers}
    corrs = {'red': [], 'clim': []}
    for i, d, r, c, m in items:
        o = tgts[i][m].astype(np.float64)
        pat = {'red': preds[i][m].astype(np.float64),
               'clim': clim[r:r + TILE, c:c + TILE][m].astype(np.float64)}
        medias = {'red': mred, 'clim': mcli}
        for p, x in pat.items():
            if o.std() > 1e-9 and x.std() > 1e-9:
                corrs[p].append(float(np.corrcoef(o, x)[0, 1]))
            add(accs[(p, 'raw')], o, x)
            add(accs[(p, 'orac')], o, x + (real[d] - medias[p][d]))
            for var in models:
                add(accs[(p, f'regr_{var}')], o, x + (regr[var][d] - medias[p][d]))

    out = {}
    for p in ('red', 'clim'):
        out[p] = {'r2_raw': r2acc(accs[(p, 'raw')]), 'r2_oraculo': r2acc(accs[(p, 'orac')]),
                  'corr': float(np.mean(corrs[p]))}
        for var in models:
            out[p][f'r2_regresor_{var}'] = r2acc(accs[(p, f'regr_{var}')])
    out['cobertura'] = comun_px / max(base_px, 1)
    return out


def mmm(v):
    v = np.asarray(v, float)
    return {'media': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='B, A y E5-E4b con la media de todas las teselas')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--e4b-dir', required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--dem', default=None)
    ap.add_argument('--slope-npz', nargs='+', default=None)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--skip-ba', action='store_true', help='solo la parte 3 (E4b)')
    ap.add_argument('--skip-e4b', action='store_true', help='solo las partes 1 y 2')
    ap.add_argument('--b-ref', default='analysis/eB_climatology_v2.json')
    ap.add_argument('--a-ref', default='analysis/eA_point_measurements_v2.json')
    ap.add_argument('--e4b-ref', nargs='*', default=['analysis/e5_recalibration.json',
                                                     'analysis/e5_recalibration_dual.json'])
    ap.add_argument('--out', default='analysis/e5_regresor_dependientes.json')
    args = ap.parse_args()
    t0 = time.time()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    cache = TileCache(args.masks)
    out = {'descripcion': 'B, A y E5 sobre E4b con el regresor de E5 ajustado con la media de '
                          'las 60 primeras teselas (actual) y con la de todas (corregido).',
           'otros_usos_del_regresor': 'ninguno: solo recalibration_test.py (E5), '
                                      'climatology_baseline.py (B) y point_measurements.py (A) '
                                      'importan mean_from_masks / ridge_fit.'}

    # ==================================================================
    # Partes 1 y 2: B y A (comparten climatologia, medias de train y .npz)
    # ==================================================================
    if not args.skip_ba:
        if args.slope_npz:
            sys.modules.setdefault('rasterio', types.ModuleType('rasterio'))
        import point_measurements as pm
        from climatology_baseline import build_climatology
        geo = pd.read_csv(args.georef, dtype={'date': str})
        pos = {t: (int(r), int(c)) for t, r, c in
               geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
        shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)
        if args.slope_npz:
            print(f'Pendientes precalculadas: {args.slope_npz}', flush=True)
            slope = np.vstack([np.load(p)['slope'] for p in args.slope_npz])
        else:
            print('Calculando pendientes desde el DEM...', flush=True)
            slope = pm.slope_map(args.dem or pm.DEM_PATH)

        B = {v: {} for v in VARIANTS}
        B_shared = {}
        A = {v: {p: {k: {str(nz): [] for nz in pm.NOISES} for k in pm.POOLS}
                 for p in ('climatologia', 'red')} for v in VARIANTS}
        var_prior = {}

        for fold in args.folds:
            print('=' * 72); print(f'Fold {fold}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
            clim, _ = build_climatology(geo, fold, args.masks, shape)
            df = pd.read_csv(FOLD_CSV.format(fold=fold))
            tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
            reg = {}
            for var, mx in VARIANTS.items():
                trm, _, _ = means_by_date_masks(tr_ids, cache, mx)
                reg[var] = fit_regressor(trm, sig)          # (modelo, var_prior, fechas)
            var_prior[fold] = {var: reg[var][1] for var in VARIANTS}
            models = {var: reg[var][0] for var in VARIANTS}

            paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                       f'resunetpp_v3_loyo{fold}_predictions.npz')}
            for s in ('42', '123'):
                paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                        f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')
            b_seed, b_clim, b_cob = {}, None, None
            clim_hecha = False
            for s in SEEDS:
                if not os.path.exists(paths[s]):
                    print(f'  [aviso] falta la semilla {s}: {paths[s]}'); continue
                print(f'  semilla {s} ...', flush=True)
                z = np.load(paths[s])
                ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
                vals = z['valids'] if 'valids' in z.files else None

                # ---- B ----
                rb = evaluate_seed_B(ids, preds, tgts, vals, pos, clim, sig, models)
                b_seed[s] = rb['red']
                if b_clim is None:
                    b_clim, b_cob = rb['clim'], rb['cobertura']

                # ---- A (point_measurements.py l. 220-258) ----
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
                for var in VARIANTS:
                    model, vp, _ = reg[var]
                    prior = {d: float(v) for d, v in zip(
                        dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}
                    for k in pm.POOLS:
                        for nz in pm.NOISES:
                            if k != 'todos' and nz != pm.NOISES[0]:
                                continue
                            A[var]['red'][k][str(nz)].append(pm.run_pattern(
                                net, so, so2, n_tot, prior, vp, nz, k,
                                np.random.default_rng(1000 + int(s))))
                            if not clim_hecha:
                                A[var]['climatologia'][k][str(nz)].append(pm.run_pattern(
                                    cli, so, so2, n_tot, prior, vp, nz, k,
                                    np.random.default_rng(99)))
                clim_hecha = True
                del z, ids, preds, tgts, vals, net, cli

            if not b_seed:
                continue
            B_shared[fold] = {'cobertura': b_cob,
                              'red_raw': mmm([v['r2_raw'] for v in b_seed.values()]),
                              'red_oraculo': mmm([v['r2_oraculo'] for v in b_seed.values()]),
                              'clim_raw': b_clim['r2_raw'], 'clim_oraculo': b_clim['r2_oraculo']}
            for var in VARIANTS:
                B[var][fold] = {'red_por_semilla': {s: v[f'r2_regresor_{var}'] for s, v in b_seed.items()},
                                'red': mmm([v[f'r2_regresor_{var}'] for v in b_seed.values()]),
                                'climatologia': b_clim[f'r2_regresor_{var}']}
            print(f"  B regresor  red 60 {B['60'][fold]['red']['media']:.3f} -> todas "
                  f"{B['todas'][fold]['red']['media']:.3f} | clim 60 {B['60'][fold]['climatologia']:.3f}"
                  f" -> todas {B['todas'][fold]['climatologia']:.3f}", flush=True)

        # ---- resumen B ----
        for var in VARIANTS:
            fl = list(B[var])
            B[var]['MEDIA_FOLDS'] = {'n_folds': len(fl),
                                     'red': float(np.mean([B[var][f]['red']['media'] for f in fl])),
                                     'climatologia': float(np.mean([B[var][f]['climatologia'] for f in fl])),
                                     'folds_red_gana': sum(B[var][f]['red']['media'] > B[var][f]['climatologia']
                                                           for f in fl)}
        out['parte1_B'] = {'por_variante': B, 'no_cambia': B_shared}
        if os.path.exists(args.b_ref):
            ref = json.load(open(args.b_ref, encoding='utf-8'))
            ds = []
            for f, v in ref.items():
                if f not in B['60']:
                    continue
                ds.append(abs(v['climatologia']['r2_regresor'] - B['60'][f]['climatologia']))
                for s, x in v['red_por_semilla'].items():
                    ds.append(abs(x['r2_regresor'] - B['60'][f]['red_por_semilla'][s]))
                ds.append(abs(v['climatologia']['r2_raw'] - B_shared[f]['clim_raw']))
                ds.append(abs(v['climatologia']['r2_oraculo'] - B_shared[f]['clim_oraculo']))
            out['control_B'] = {'valores_comparados': len(ds), 'max_diferencia': max(ds) if ds else None}

        # ---- resumen A (mismo calculo que point_measurements.resumen) ----
        def resumen(runs):
            sin = np.mean([r['sin'] for r in runs]); orac = np.mean([r['oraculo'] for r in runs])
            sat = np.mean([r['satelite'] for r in runs])
            curvas = {m: [float(np.mean([r['curvas'][m][k] for r in runs])) for k in range(len(pm.N_LIST))]
                      for m in pm.METHODS}
            camp = {m: [np.concatenate([np.asarray(r['campanas'][m][k]) for r in runs])
                        for k in range(len(pm.N_LIST))] for m in pm.METHODS}
            return {'sin': float(sin), 'oraculo': float(orac), 'satelite': float(sat), 'curvas': curvas,
                    'campanas_p10_p50_p90': {m: [np.percentile(c, [10, 50, 90]).tolist() for c in camp[m]]
                                             for m in pm.METHODS},
                    'N90': {m: pm.n90(sin, orac, curvas[m]) for m in pm.METHODS},
                    'fechas_sin_pool': sum(r['fechas_sin_pool'] for r in runs)}

        Ares = {var: {} for var in VARIANTS}
        for var in VARIANTS:
            for pat in ('climatologia', 'red'):
                for k in pm.POOLS:
                    for nz, runs in A[var][pat][k].items():
                        if runs:
                            Ares[var][f'{pat}|{k}|{nz}'] = resumen(runs)
        # satelite: patron+sat menos patron, por N (principal: todos, 0.1)
        sat_eff = {}
        for var in VARIANTS:
            sat_eff[var] = {}
            for pat in ('climatologia', 'red'):
                cv = Ares[var][f'{pat}|todos|{pm.NOISES[0]}']['curvas']
                dif = [a - b for a, b in zip(cv['patron+sat'], cv['patron'])]
                deja = next((N for N, x in zip(pm.N_LIST, dif) if x <= 0), None)
                sat_eff[var][pat] = {'dif_por_N': dict(zip(map(str, pm.N_LIST), dif)),
                                     'primer_N_sin_ayuda': deja}
        out['parte2_A'] = {'N': pm.N_LIST, 'metodos': pm.METHODS, 'principal': pm.PRINCIPAL,
                           'resumen': Ares, 'satelite_frente_a_patron': sat_eff,
                           'var_prior_por_fold': var_prior}
        if os.path.exists(args.a_ref):
            ref = json.load(open(args.a_ref, encoding='utf-8'))['resumen']
            ds = []
            for key, o in ref.items():
                mine = Ares['60'].get(key)
                if mine is None:
                    continue
                ds += [abs(mine[x] - o[x]) for x in ('sin', 'oraculo', 'satelite')]
                for m in pm.METHODS:
                    ds += [abs(a - b) for a, b in zip(mine['curvas'][m], o['curvas'][m])]
                    for a, b in zip(mine['campanas_p10_p50_p90'][m], o['campanas_p10_p50_p90'][m]):
                        ds += [abs(x - y) for x, y in zip(a, b)]
            out['control_A'] = {'valores_comparados': len(ds), 'max_diferencia': max(ds) if ds else None}

    # ==================================================================
    # Parte 3: E5 sobre E4b, fold 2025
    # ==================================================================
    if not args.skip_e4b:
        print('=' * 72); print(f'E5 sobre E4b  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        E = {}
        for ds_name, csv in E4B_CSV.items():
            df = pd.read_csv(csv)
            tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
            train_dates_csv = sorted({date_of(t) for t in tr_ids})
            tm_by_var = {}
            for var, mx in VARIANTS.items():
                means, _, _ = means_by_date_masks(tr_ids, cache, mx)
                td = [d for d in train_dates_csv if d in means and d in sig.index]
                tm_by_var[var] = (td, np.array([means[d] for d in td]))
            n_train_tiles = len(tr_ids)
            for arm in ARMS:
                pattern = os.path.join(args.e4b_dir, E4B_NPZ[ds_name].format(arm=arm))
                paths = [p for p in sorted(glob.glob(pattern)) if '_last' not in os.path.basename(p)]
                if not paths:
                    print(f'  [aviso] sin .npz: {pattern}'); continue
                rows = []
                for p in paths:
                    name = os.path.basename(p).replace('_predictions.npz', '')
                    print(f'  {name} ...', flush=True)
                    rows.append(e5_model(load_npz(p), sig, tm_by_var, name))
                res = {'n_teselas_train': n_train_tiles, 'modelos': rows}
                sin = np.mean([r['r2_sin_recal'] for r in rows])
                orac = np.mean([r['r2_oraculo_add'] for r in rows])
                for var in VARIANTS:
                    corr = np.mean([r['variantes'][var]['r2_pred_add_cobertura'] for r in rows])
                    res[var] = {'r2_sin': float(sin), 'r2_corregido': float(corr), 'r2_oraculo': float(orac),
                                'corregido_min_max': [float(min(r['variantes'][var]['r2_pred_add_cobertura'] for r in rows)),
                                                      float(max(r['variantes'][var]['r2_pred_add_cobertura'] for r in rows))],
                                'fraccion_techo': float((corr - sin) / (orac - sin)) if orac > sin else None,
                                'mae_regresor': float(np.mean([r['variantes'][var]['mae_media_cobertura'] for r in rows])),
                                'mae_red': float(np.mean([r['mae_media_red'] for r in rows])),
                                'criterio_acierta': f"{sum(r['variantes'][var]['criterio_acierta'] for r in rows)} de {len(rows)}"}
                E[f'{arm}|{ds_name}'] = res
        out['parte3_E5_E4b'] = E
        ref = {}
        for p in args.e4b_ref:
            if os.path.exists(p):
                j = json.load(open(p, encoding='utf-8'))
                rows_ref = j.get('rows', []) + [r for rr in j.get('folds', {}).values() for r in rr]
                for r in rows_ref:
                    ref[r['experimento']] = r
        ds = []
        for key, res in E.items():
            for r in res['modelos']:
                o = ref.get(r['experimento'])
                if o is None:
                    continue
                v = r['variantes']['60']
                ds += [abs(r['r2_sin_recal'] - o['r2_sin_recal']), abs(r['r2_oraculo_add'] - o['r2_oraculo_add']),
                       abs(v['r2_pred_add_cobertura'] - o['r2_pred_add_cobertura']),
                       abs(v['mae_media_cobertura'] - o['mae_media_cobertura'])]
        out['control_E4b'] = {'valores_comparados': len(ds), 'max_diferencia': max(ds) if ds else None}

    # ==================================================================
    # Impresion
    # ==================================================================
    print('=' * 72)
    if 'parte1_B' in out:
        print('PARTE 1 - B, nivel del regresor (red: media [min-max] de 3 semillas)')
        print(f"  {'fold':<12}{'red 60':>24}{'red todas':>24}{'clim 60':>9}{'clim todas':>11}")
        for f in B['60']:
            if f == 'MEDIA_FOLDS':
                continue
            a, b = B['60'][f]['red'], B['todas'][f]['red']
            print(f"  {f:<12}{a['media']:>8.3f} [{a['min']:.3f}-{a['max']:.3f}]"
                  f"{b['media']:>9.3f} [{b['min']:.3f}-{b['max']:.3f}]"
                  f"{B['60'][f]['climatologia']:>9.3f}{B['todas'][f]['climatologia']:>11.3f}")
        for var in VARIANTS:
            m = B[var]['MEDIA_FOLDS']
            print(f"  MEDIA [{var}]: red {m['red']:.3f} | clim {m['climatologia']:.3f} | "
                  f"red gana en {m['folds_red_gana']} de {m['n_folds']}")
        print(f"  control (60 = JSON actual): {out.get('control_B')}")

        print('\nPARTE 2 - A')
        for var in VARIANTS:
            print(f'  [{var}] N90 (todos, 10 cm):')
            for pat in ('climatologia', 'red'):
                r = Ares[var][f'{pat}|todos|{pm.NOISES[0]}']
                print(f"    {pat:<13}" + ', '.join(f"{m} {r['N90'][m] or '>100'}" for m in pm.METHODS))
        k10 = pm.N_LIST.index(10)
        print('  P10 / P50 / P90 por campana, N=10, patron+sat (todos, 10 cm):')
        for var in VARIANTS:
            for pat in ('climatologia', 'red'):
                p = Ares[var][f'{pat}|todos|{pm.NOISES[0]}']['campanas_p10_p50_p90']['patron+sat'][k10]
                print(f'    [{var}] {pat:<13}{p[0]:>7.3f}{p[1]:>7.3f}{p[2]:>7.3f}')
        print('  Sensibilidad, patron+sat, N=10: R2 medio (P10)')
        casos = [(k, str(pm.NOISES[0])) for k in pm.POOLS] + [('todos', str(nz)) for nz in pm.NOISES[1:]]
        for var in VARIANTS:
            for k, nz in casos:
                line = f'    [{var}] {k + ", " + str(int(float(nz) * 100)) + " cm":<18}'
                for pat in ('climatologia', 'red'):
                    r = Ares[var][f'{pat}|{k}|{nz}']
                    line += (f"{pat[:4]} {r['curvas']['patron+sat'][k10]:.3f} "
                             f"({r['campanas_p10_p50_p90']['patron+sat'][k10][0]:.3f})   ")
                print(line)
        print('  Satelite: (patron+sat) - (patron) por N; primer N sin ayuda')
        for var in VARIANTS:
            for pat in ('climatologia', 'red'):
                e = sat_eff[var][pat]
                print(f"    [{var}] {pat:<13}" + ' '.join(f"{N}:{x:+.3f}" for N, x in e['dif_por_N'].items())
                      + f"   -> {e['primer_N_sin_ayuda']}")
        print(f"  control (60 = JSON actual): {out.get('control_A')}")

    if 'parte3_E5_E4b' in out:
        print('\nPARTE 3 - E5 sobre E4b, fold 2025 (media de semillas)')
        print(f"  {'brazo|datos':<18}{'sin':>7}{'corr 60':>9}{'techo 60':>9}{'corr todas':>11}"
              f"{'techo todas':>12}{'oraculo':>9}{'criterio 60/todas':>19}")
        for key, res in out['parte3_E5_E4b'].items():
            a, b = res['60'], res['todas']
            fa = f"{100 * a['fraccion_techo']:.0f}%" if a['fraccion_techo'] is not None else '-'
            fb = f"{100 * b['fraccion_techo']:.0f}%" if b['fraccion_techo'] is not None else '-'
            print(f"  {key:<18}{a['r2_sin']:>7.3f}{a['r2_corregido']:>9.3f}{fa:>9}"
                  f"{b['r2_corregido']:>11.3f}{fb:>12}{a['r2_oraculo']:>9.3f}"
                  f"{a['criterio_acierta'] + ' / ' + b['criterio_acierta']:>19}")
        print(f"  control (60 = JSON actual): {out.get('control_E4b')}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'\nGuardado en: {args.out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
