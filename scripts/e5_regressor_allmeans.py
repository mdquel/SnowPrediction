r"""
E5 y A con el regresor ajustado sobre la media de TODAS las teselas (sin GPU)
=============================================================================

Problema que se comprueba
-------------------------
En entrenamiento, la HS media objetivo de cada fecha se calcula con
recalibration_test.mean_from_masks, que solo lee las 60 PRIMERAS teselas de
cada fecha (max_per_date=60). Las teselas vienen ordenadas por fila, asi que
esas 60 son la franja norte de la zona volada, no una muestra aleatoria. En
test, en cambio, la media real se toma con todas las teselas.

Que hace
--------
1. Para cada fold y cada fecha de train admisible: media con las 60 primeras
   teselas frente a la media con todas (solo pixeles con nieve, mismo filtro
   que mean_from_masks). Diferencia media, maxima y signo.
2. E5 (E2, 4 folds x semillas 7, 42, 123): regresor reajustado con la media de
   todas las teselas. Por fold y semilla: MAE del regresor y de la red al
   estimar la media de test, ganancia de R2 de la correccion (principal:
   cobertura, aditiva) y acierto del criterio. Se calculan las DOS variantes
   (60 y todas) en la misma pasada.
3. A, configuracion principal (patron+sat, muestreo libre, ruido 10 cm): R2
   con 1, 3 y 10 puntos para climatologia y red, con las dos variantes. Con
   el regresor cambian dos cosas: la media predicha (prior) y su error de
   validacion cruzada (var_prior), que fija el peso del satelite.

Control de reproduccion
-----------------------
La variante "60" debe reproducir EXACTAMENTE las cifras actuales:
    E5 -> analysis/e5_recalibration_e2_s7.json (semilla 7) y
          analysis/e5_recalibration_e2seeds.json (s42, s123)
    A  -> analysis/eA_point_measurements_v2.json
Los generadores aleatorios de A se crean igual que en point_measurements.py
y no dependen del regresor, asi que los puntos muestreados son los mismos en
las dos variantes: la unica diferencia es la media de train.

Todo lo que se puede se importa (recalibration_test, point_measurements,
climatology_baseline). La media por fecha se replica aqui con una cache por
tesela para no leer dos veces cada mascara; la replica sigue linea a linea a
mean_from_masks (l. 95-125).

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/e5_regressor_allmeans.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas

    Si al ejecutar A se paso --dem, pasar el mismo --dem aqui.
    Sin rasterio: --slope-npz <parte1> <parte2> con la pendiente calculada
    fuera por point_measurements.slope_map sobre el mismo DEM, partida por
    filas (se apilan en orden; mismos valores).
    Solo E5 (sin DEM ni rasterio): anyadir --skip-a
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_test import (date_of, ridge_fit, ridge_predict,       # noqa: E402
                                load_npz, means_by_date, evaluate,
                                SIGNAL_SETS, SNOW_THRESHOLD, MIN_VALID_PX)

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
VARIANTS = {'60': 60, 'todas': None}       # None = sin limite
N_SHOW = [1, 3, 10]
NOISE = 0.10                               # principal de A
POOL = 'todos'                             # principal de A
PRINCIPAL_SET = 'cobertura'                # principal de E5


# ----------------------------------------------------------------------
# 1. Media por fecha desde las mascaras
# ----------------------------------------------------------------------
class TileCache:
    """(suma, n) de pixeles con nieve por tesela; None si no cuenta."""
    def __init__(self, masks_dir):
        self.masks_dir = masks_dir
        self.c = {}

    def get(self, t):
        t = str(t)
        if t not in self.c:
            path = os.path.join(self.masks_dir, t)
            if not os.path.exists(path):
                self.c[t] = None
            else:
                m = np.load(path).ravel()
                m = m[np.isfinite(m) & (m > SNOW_THRESHOLD)]
                self.c[t] = None if m.size < MIN_VALID_PX else (float(m.sum()), int(m.size))
        return self.c[t]


def means_by_date_masks(tile_ids, cache, max_per_date):
    """Replica de recalibration_test.mean_from_masks con cache.

    Mismo orden, mismo corte ts[:max_per_date] (cuentan tambien las teselas
    que luego se descartan) y mismo filtro de pixeles.
    """
    por_fecha = {}
    for t in tile_ids:
        por_fecha.setdefault(date_of(t), []).append(t)
    acc, usadas = {}, {}
    for d, ts in por_fecha.items():
        sel = ts if max_per_date is None else ts[:max_per_date]
        for t in sel:
            v = cache.get(t)
            if v is None:
                continue
            s, n = acc.get(d, (0.0, 0))
            acc[d] = (s + v[0], n + v[1])
            usadas[d] = usadas.get(d, 0) + 1
    means = {d: s / n for d, (s, n) in acc.items() if n > 0}
    return means, {d: len(ts) for d, ts in por_fecha.items()}, usadas


def row_of(t):
    # 20210304_lidar_tile_<fila>_<col>.npy
    parts = str(t).replace('.npy', '').split('_')
    return int(parts[-2])


def fit_regressor(trm, sig, cols=('snow_pct',)):
    """Mismo ajuste que point_measurements.py l. 203-209 (y E5 'cobertura')."""
    trd = [d for d in sorted(trm) if d in sig.index]
    X = sig.loc[trd, list(cols)].to_numpy(float)
    y = np.array([trm[d] for d in trd])
    cv = [ridge_predict(ridge_fit(np.delete(X, j, 0), np.delete(y, j)), X[j:j + 1])[0] - y[j]
          for j in range(len(trd))]
    return ridge_fit(X, y), float(np.mean(np.square(cv))), trd


def summ(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None
    i = int(np.argmax(np.abs(v)))
    return {'media': float(v.mean()), 'media_abs': float(np.abs(v).mean()),
            'max_abs': float(abs(v[i])), 'signo_max': '+' if v[i] > 0 else '-',
            'n_pos': int((v > 0).sum()), 'n_neg': int((v < 0).sum()), 'n': int(v.size)}


# ----------------------------------------------------------------------
# 2. E5 para un modelo, las dos variantes
# ----------------------------------------------------------------------
def e5_model(data, sig, train_means_by_var, name):
    """Replica de recalibration_test.run_fold (version del repo) para un .npz.

    Modifica 'data' en sitio (filtra a fechas con senal), como el original.
    """
    keep = np.isin(data['dates'], list(sig.index))
    for k in ('tile_ids', 'preds', 'targets', 'mask', 'dates'):
        data[k] = data[k][keep]
    if data['valids'] is not None:
        data['valids'] = data['valids'][keep]

    test_real = means_by_date(data, 'targets')
    pred_raw = means_by_date(data, 'preds')
    test_dates = [d for d in sorted(test_real) if d in sig.index]
    y_true = np.array([test_real[d] for d in test_dates])
    mae_red = float(np.mean([abs(pred_raw[d] - test_real[d]) for d in test_dates if d in pred_raw]))
    r2_sin = evaluate(data, pred_raw, {}, 'none')

    out = {'experimento': name, 'n_test_fechas': len(test_dates),
           'r2_sin_recal': r2_sin,
           'r2_oraculo_add': evaluate(data, pred_raw, test_real, 'add'),
           'r2_oraculo_mult': evaluate(data, pred_raw, test_real, 'mult'),
           'mae_media_red': mae_red, 'variantes': {}}
    for var, (train_dates, y_tr) in train_means_by_var.items():
        row = {'n_train_fechas': len(train_dates)}
        for set_name, cols in SIGNAL_SETS.items():
            X_tr = sig.loc[train_dates, cols].to_numpy(dtype=np.float64)
            X_te = sig.loc[test_dates, cols].to_numpy(dtype=np.float64)
            y_hat = ridge_predict(ridge_fit(X_tr, y_tr, alpha=1.0), X_te)
            pm = {d: float(v) for d, v in zip(test_dates, y_hat)}
            row[f'mae_media_{set_name}'] = float(np.mean(np.abs(y_hat - y_true)))
            row[f'r2_pred_add_{set_name}'] = evaluate(data, pred_raw, pm, 'add')
            row[f'r2_pred_mult_{set_name}'] = evaluate(data, pred_raw, pm, 'mult')
        mr = row[f'mae_media_{PRINCIPAL_SET}']
        g = row[f'r2_pred_add_{PRINCIPAL_SET}'] - r2_sin
        predice = mr < mae_red
        row['ganancia_r2_principal'] = g
        row['criterio_predice'] = 'ayuda' if predice else 'empeora'
        row['criterio_acierta'] = bool((predice and g > 0) or (not predice and g <= 0))
        out['variantes'][var] = row
    return out


# ----------------------------------------------------------------------
# 3. A para un modelo (estadisticos ya construidos), las dos variantes
# ----------------------------------------------------------------------
def a_curves(run):
    from point_measurements import N_LIST
    return {'sin': run['sin'], 'oraculo': run['oraculo'], 'satelite': run['satelite'],
            'patron+sat': {str(N): run['curvas']['patron+sat'][N_LIST.index(N)] for N in N_SHOW},
            'campanas_patron+sat_N10': run['campanas']['patron+sat'][N_LIST.index(10)]}


def a_summary(runs):
    from point_measurements import N_LIST
    k10 = N_LIST.index(10)
    res = {'sin': float(np.mean([r['sin'] for r in runs])),
           'oraculo': float(np.mean([r['oraculo'] for r in runs])),
           'satelite': float(np.mean([r['satelite'] for r in runs])),
           'patron+sat': {str(N): float(np.mean([r['patron+sat'][str(N)] for r in runs])) for N in N_SHOW},
           'P10_N10': float(np.percentile(np.concatenate(
               [np.asarray(r['campanas_patron+sat_N10']) for r in runs]), 10)),
           'n_runs': len(runs)}
    return res


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='E5 y A con la media de todas las teselas')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--dem', default=None, help='el mismo que se uso en A (por defecto, el de A)')
    ap.add_argument('--slope-npz', nargs='+', default=None,
                    help='pendiente ya calculada con point_measurements.slope_map (evita rasterio)')
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--skip-a', action='store_true', help='solo puntos 1 y 2')
    ap.add_argument('--e5-ref', nargs='*', default=['analysis/e5_recalibration_e2_s7.json',
                                                    'analysis/e5_recalibration_e2seeds.json'])
    ap.add_argument('--a-ref', default='analysis/eA_point_measurements_v2.json')
    ap.add_argument('--out', default='analysis/e5_regresor_todas_teselas.json')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    cache = TileCache(args.masks)

    if not args.skip_a:
        if args.slope_npz:
            # point_measurements importa rasterio al cargarse, pero con la
            # pendiente precalculada no hace falta: se registra un modulo
            # vacio para poder importar DateStats y run_pattern sin rasterio.
            import types
            sys.modules.setdefault('rasterio', types.ModuleType('rasterio'))
        from point_measurements import DateStats, run_pattern, slope_map, DEM_PATH
        from climatology_baseline import build_climatology
        geo = pd.read_csv(args.georef, dtype={'date': str})
        pos = {t: (int(r), int(c)) for t, r, c in
               geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
        shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)
        if args.slope_npz:
            print(f'Pendientes precalculadas: {args.slope_npz}', flush=True)
            # en varias partes por filas (limite de tamano de fichero): se apilan
            slope = np.vstack([np.load(p)['slope'] for p in args.slope_npz])
        else:
            print('Calculando pendientes desde el DEM...', flush=True)
            slope = slope_map(args.dem or DEM_PATH)

    out = {'descripcion': 'Regresor de E5 ajustado con la media de las 60 primeras teselas '
                          '(actual) frente a la media de todas las teselas de cada fecha de train.',
           'punto1_medias_train': {}, 'punto2_e5': {}, 'punto3_A': {}}
    a_runs = {v: {'climatologia': [], 'red': []} for v in VARIANTS}
    all_diffs = []

    for fold in args.folds:
        print('=' * 72); print(f'Fold {fold}', flush=True)
        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()

        # ---- punto 1 ----
        means, n_tiles, used = {}, None, {}
        for var, mx in VARIANTS.items():
            means[var], n_tiles, used[var] = means_by_date_masks(tr_ids, cache, mx)
        filas = {}
        for t in tr_ids:
            filas.setdefault(date_of(t), []).append(row_of(t))
        fechas = [d for d in sorted(means['todas']) if d in sig.index and d in means['60']]
        rows = []
        for d in fechas:
            m60, mall = means['60'][d], means['todas'][d]
            rows.append({'fecha': d, 'n_teselas_train': n_tiles[d],
                         'n_usadas_60': used['60'].get(d, 0), 'n_usadas_todas': used['todas'].get(d, 0),
                         'fila_max_60': int(max(filas[d][:60])), 'fila_max_todas': int(max(filas[d])),
                         'media_60': m60, 'media_todas': mall, 'dif_m': m60 - mall,
                         'dif_pct': 100 * (m60 - mall) / mall})
        dif = [r['dif_m'] for r in rows if r['n_teselas_train'] > 60]
        all_diffs.extend(dif)
        out['punto1_medias_train'][fold] = {'fechas': rows, 'resumen_fechas_con_mas_de_60': summ(dif),
                                            'n_fechas_sin_cambio': sum(r['n_teselas_train'] <= 60 for r in rows)}
        print(f"  {'fecha':<10}{'teselas':>8}{'media60':>9}{'todas':>8}{'dif m':>8}{'dif %':>8}")
        for r in rows:
            print(f"  {r['fecha']:<10}{r['n_teselas_train']:>8}{r['media_60']:>9.3f}"
                  f"{r['media_todas']:>8.3f}{r['dif_m']:>+8.3f}{r['dif_pct']:>+8.1f}")

        # medias de train por variante, como en run_fold (fechas del CSV con media y senal)
        train_dates_csv = sorted({date_of(t) for t in tr_ids})
        tm_by_var = {}
        for var in VARIANTS:
            td = [d for d in train_dates_csv if d in means[var] and d in sig.index]
            tm_by_var[var] = (td, np.array([means[var][d] for d in td]))

        # regresor de A por variante
        if not args.skip_a:
            regA = {var: fit_regressor(means[var], sig) for var in VARIANTS}
            clim, _ = build_climatology(geo, fold, args.masks, shape)

        paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                   f'resunetpp_v3_loyo{fold}_predictions.npz')}
        for s in ('42', '123'):
            paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                    f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')
        out['punto2_e5'][fold] = []
        clim_hecha = False
        for s in SEEDS:
            if not os.path.exists(paths[s]):
                print(f'  [aviso] falta la semilla {s}: {paths[s]}'); continue
            name = os.path.basename(paths[s]).replace('_predictions.npz', '')
            print(f'  {name} ...', flush=True)
            data = load_npz(paths[s])

            # ---- punto 3 (antes de filtrar data) - replica de point_measurements.py ----
            if not args.skip_a:
                rng = np.random.default_rng(int(s))
                ids, preds, tgts, vals = data['tile_ids'], data['preds'], data['targets'], data['valids']
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
                    net.setdefault(d, DateStats()).add(o, preds[i][m].astype(np.float64), sl[m], rng)
                    if not clim_hecha:
                        cli.setdefault(d, DateStats()).add(o, cp[m].astype(np.float64), sl[m], rng)
                for st in list(net.values()) + list(cli.values()):
                    st.finish()
                dates = sorted(net)
                for var in VARIANTS:
                    model, var_prior, _ = regA[var]
                    prior = {d: float(v) for d, v in zip(
                        dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}
                    a_runs[var]['red'].append(a_curves(run_pattern(
                        net, so, so2, n_tot, prior, var_prior, NOISE, POOL,
                        np.random.default_rng(1000 + int(s)))))
                    if not clim_hecha:
                        a_runs[var]['climatologia'].append(a_curves(run_pattern(
                            cli, so, so2, n_tot, prior, var_prior, NOISE, POOL,
                            np.random.default_rng(99))))
                clim_hecha = True
                del net, cli

            # ---- punto 2 ----
            res = e5_model(data, sig, tm_by_var, name)
            out['punto2_e5'][fold].append(res)
            del data
            for var in VARIANTS:
                v = res['variantes'][var]
                print(f"    {var:<6} MAE regr {v['mae_media_cobertura']:.3f}  MAE red "
                      f"{res['mae_media_red']:.3f}  ganancia {v['ganancia_r2_principal']:+.3f}  "
                      f"predice {v['criterio_predice']:<8} acierta {'SI' if v['criterio_acierta'] else 'NO'}")

        if not args.skip_a:
            out['punto3_A'].setdefault('var_prior_por_fold', {})[fold] = {
                var: regA[var][1] for var in VARIANTS}

    # ---------------- resumenes ----------------
    out['punto1_resumen_global'] = summ(all_diffs)

    crit = {}
    for var in VARIANTS:
        c = {'semilla_7': [0, 0], 'semillas_42_123': [0, 0]}
        for fold, rows in out['punto2_e5'].items():
            for r in rows:
                k = 'semillas_42_123' if r['experimento'].endswith(('_s42', '_s123')) else 'semilla_7'
                c[k][1] += 1
                c[k][0] += int(r['variantes'][var]['criterio_acierta'])
        crit[var] = {k: f'{a} de {b}' for k, (a, b) in c.items()}
    out['punto2_criterio'] = crit

    # control de reproduccion E5
    ref = {}
    for p in args.e5_ref:
        if os.path.exists(p):
            j = json.load(open(p, encoding='utf-8'))
            for fold, rows in j.get('folds', {}).items():
                for r in rows:
                    ref[(fold, r['experimento'])] = r
    checks = []
    for fold, rows in out['punto2_e5'].items():
        for r in rows:
            o = ref.get((fold, r['experimento']))
            if o is None:
                continue
            v = r['variantes']['60']
            checks.append(max(abs(v['r2_pred_add_cobertura'] - o['r2_pred_add_cobertura']),
                              abs(v['mae_media_cobertura'] - o['mae_media_cobertura']),
                              abs(r['r2_sin_recal'] - o['r2_sin_recal'])))
    out['control_e5'] = {'modelos_comparados': len(checks),
                         'max_diferencia_variante_60_vs_json': max(checks) if checks else None}

    if not args.skip_a:
        out['punto3_A']['resumen'] = {var: {pat: a_summary(a_runs[var][pat]) for pat in ('climatologia', 'red')}
                                      for var in VARIANTS}
        if os.path.exists(args.a_ref):
            j = json.load(open(args.a_ref, encoding='utf-8'))['resumen']
            from point_measurements import N_LIST
            ctrl = {}
            for pat in ('climatologia', 'red'):
                o = j.get(f'{pat}|{POOL}|{NOISE}')
                if o is None:
                    continue
                mine = out['punto3_A']['resumen']['60'][pat]
                ds = [abs(mine['sin'] - o['sin']), abs(mine['satelite'] - o['satelite'])]
                ds += [abs(mine['patron+sat'][str(N)] - o['curvas']['patron+sat'][N_LIST.index(N)])
                       for N in N_SHOW]
                ctrl[pat] = max(ds)
            out['control_A'] = {'max_diferencia_variante_60_vs_json': ctrl}

    # ---------------- impresion ----------------
    print('=' * 72)
    print('PUNTO 1 - media 60 primeras menos media todas (fechas con > 60 teselas)')
    for fold, v in out['punto1_medias_train'].items():
        s_ = v['resumen_fechas_con_mas_de_60']
        if s_:
            print(f"  {fold}: media {s_['media']:+.3f} m | media abs {s_['media_abs']:.3f} | "
                  f"max {s_['signo_max']}{s_['max_abs']:.3f} | +{s_['n_pos']} / -{s_['n_neg']} "
                  f"| sin cambio {v['n_fechas_sin_cambio']}")
    g = out['punto1_resumen_global']
    if g:
        print(f"  GLOBAL: media {g['media']:+.3f} m | media abs {g['media_abs']:.3f} | "
              f"max {g['signo_max']}{g['max_abs']:.3f} | +{g['n_pos']} / -{g['n_neg']} (n={g['n']})")

    print('\nPUNTO 2 - E5 principal (cobertura, aditiva)')
    print(f"  {'modelo':<30}{'MAE red':>8}" + ''.join(f'{"MAE " + v:>11}{"gan " + v:>11}{"ok":>4}'
                                                    for v in VARIANTS))
    for fold, rows in out['punto2_e5'].items():
        for r in rows:
            line = f"  {r['experimento']:<30}{r['mae_media_red']:>8.3f}"
            for var in VARIANTS:
                v = r['variantes'][var]
                line += (f"{v['mae_media_cobertura']:>11.3f}{v['ganancia_r2_principal']:>+11.3f}"
                         f"{'SI' if v['criterio_acierta'] else 'NO':>4}")
            print(line)
    for var, c in crit.items():
        print(f"  criterio [{var}]: semilla 7 {c['semilla_7']} | semillas 42 y 123 {c['semillas_42_123']}")
    ce = out['control_e5']
    print(f"  control (variante 60 = JSON actual): {ce['modelos_comparados']} modelos, "
          f"diferencia maxima {ce['max_diferencia_variante_60_vs_json']}")

    if not args.skip_a:
        print('\nPUNTO 3 - A principal (patron+sat, muestreo libre, 10 cm)')
        print(f"  {'patron / variante':<24}{'sin':>7}{'sat':>7}" +
              ''.join(f'{"N=" + str(N):>7}' for N in N_SHOW) + f"{'P10@10':>8}{'oraculo':>9}")
        for pat in ('climatologia', 'red'):
            for var in VARIANTS:
                s_ = out['punto3_A']['resumen'][var][pat]
                print(f"  {pat + ' / ' + var:<24}{s_['sin']:>7.3f}{s_['satelite']:>7.3f}" +
                      ''.join(f"{s_['patron+sat'][str(N)]:>7.3f}" for N in N_SHOW) +
                      f"{s_['P10_N10']:>8.3f}{s_['oraculo']:>9.3f}")
        print(f"  control (variante 60 = JSON actual): {out.get('control_A')}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
