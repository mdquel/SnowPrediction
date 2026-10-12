r"""
Cifras canonicas de Resultados 5.2 y datos de la figura de mapas (sin GPU)
=========================================================================

Escribe analysis/cifras_canonicas_5_2.json y analysis/fig05_tesela.npz. No
modifica ningun JSON existente.

1. Experimento B (LOYO-E2, fechas con senal de cobertura). Red (semillas 7,
   42, 123) frente a la climatologia de los anyos de entrenamiento del fold.
   R2 y SPAEF (a) sin corregir, (b) con nivel perfecto aditivo y (c) con el
   nivel del regresor de E5 corregido (media de train con todas las
   teselas); correlacion de patron por tesela; fechas, teselas y pixeles
   evaluados por fold. Mascara y calculo de R2: climatology_baseline.py
   (evaluate_seed) linea a linea; se anade el SPAEF por tesela, igual que
   en C (utils.metrics.compute_spaef, media sin ponderar).
   Control: R2 tal cual / nivel perfecto = analysis/eB_climatology_v2.json;
   R2 con el regresor corregido = analysis/e5_regresor_dependientes.json.

2. Experimento C (split espacial, 702 teselas, mascara de C). Techo
   (climatologia de los demas anyos), red y RF justo, sin corregir y con
   nivel perfecto: R2, SPAEF, correlacion por tesela y fraccion del techo.
   Mismo calculo que scripts/spatial_models_vs_climatology.py (funciones
   importadas o replicadas linea a linea). Se anade la climatologia sin
   corregir (R2 y SPAEF), que aquel script no guardaba.
   Control: = analysis/eC_net_vs_rf_allpix.json.

3. Descomposicion T1a / T1b / T2 / T3 del split espacial en la mascara de C
   (red, RF y climatologia sin corregir), como scripts/decomp_oracle_checks.py.

4. Tesela de la figura de mapas, con la regla fijada antes de verla: entre
   todas las tesela-fecha del test espacial (mascara de C), la que tenga el
   R2 por tesela de la red (semilla 42, nivel perfecto) mas cercano a la
   mediana de TODAS las tesela-fecha, con al menos el 50 % de los 256 x 256
   pixeles de la tesela con nieve (dentro de la mascara de C). Se guardan
   los cuatro campos con nivel perfecto (observado, climatologia, red s42,
   RF s42) en analysis/fig05_tesela.npz, con el R2 y el SPAEF de cada uno en
   esa tesela.

Uso (desde la raiz del repo)
----------------------------
    python scripts/results_5_2.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas ^
        --spatial-net <npz s7> <npz s42> <npz s123> ^
        --spatial-rf  <npz s7> <npz s42> <npz s123>
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'analysis'))

# utils/metrics.py importa sklearn al cargarse, pero compute_spaef solo usa
# numpy. En el PC del trabajo el Control de aplicaciones de Windows bloquea
# las DLL de scipy (que sklearn carga), asi que si sklearn no se puede
# importar se registra un modulo vacio con esos nombres. No cambia ningun
# calculo: este script no llama a ninguna funcion de sklearn.
try:
    import sklearn.metrics  # noqa: F401
except Exception:
    import types
    _sk = types.ModuleType('sklearn')
    _skm = types.ModuleType('sklearn.metrics')
    for _n in ('mean_absolute_error', 'mean_squared_error', 'r2_score'):
        setattr(_skm, _n, None)
    _sk.metrics = _skm
    sys.modules['sklearn'] = _sk
    sys.modules['sklearn.metrics'] = _skm
    print('[aviso] sklearn no disponible: se usa un modulo vacio (no afecta al calculo)')

from recalibration_test import date_of, ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX   # noqa: E402
from e5_regressor_allmeans import TileCache, means_by_date_masks, fit_regressor        # noqa: E402
from climatology_baseline import build_climatology                                     # noqa: E402
from spatial_models_vs_climatology import build_year_climatology                       # noqa: E402
from utils.metrics import compute_spaef                                                # noqa: E402
from e1_error_decomposition import decompose_tile                                      # noqa: E402

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'


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


def r2_arr(o, p):
    ss = float(np.sum((o - o.mean()) ** 2))
    return 1.0 - float(np.sum((o - p) ** 2)) / ss if ss > 0 else float('nan')


# Decimales de salida: 4 (y 2 en porcentajes) por defecto, como el JSON
# original; con --nd 6 se guardan 6 (y 4 en porcentajes) en un fichero aparte.
ND = 4
ND_PCT = 2


def set_nd(nd):
    global ND, ND_PCT
    ND, ND_PCT = nd, nd - 2


def mmm(v, nd=None):
    nd = ND if nd is None else nd
    v = np.asarray(v, float)
    return {'media': round(float(v.mean()), nd), 'min': round(float(v.min()), nd),
            'max': round(float(v.max()), nd), 'por_semilla': [round(float(x), nd) for x in v]}


def spaef_mean(lst):
    v = [x for x in lst if not np.isnan(x)]
    return float(np.mean(v)), len(v)


# ======================================================================
# 1. B
# ======================================================================
def evaluate_B(ids, preds, tgts, vals, pos, clim, sig, model):
    """climatology_baseline.evaluate_seed, linea a linea, + SPAEF y recuentos."""
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
    regr = {d: float(v) for d, v in zip(
        tdates, ridge_predict(model, sig.loc[tdates, ['snow_pct']].to_numpy(float)))}

    accs = {(p, v): new_acc() for p in ('red', 'clim') for v in ('raw', 'orac', 'regr')}
    sp = {(p, v): [] for p in ('red', 'clim') for v in ('raw', 'orac', 'regr')}
    corrs = {'red': [], 'clim': []}
    n_px = 0
    for i, d, r, c, m in items:
        o = tgts[i][m].astype(np.float64)
        n_px += o.size
        pat = {'red': preds[i][m].astype(np.float64),
               'clim': clim[r:r + TILE, c:c + TILE][m].astype(np.float64)}
        medias = {'red': mred, 'clim': mcli}
        for p, x in pat.items():
            if o.std() > 1e-9 and x.std() > 1e-9:
                corrs[p].append(float(np.corrcoef(o, x)[0, 1]))
            versions = {'raw': x, 'orac': x + (real[d] - medias[p][d]),
                        'regr': x + (regr[d] - medias[p][d])}
            for v, xx in versions.items():
                add(accs[(p, v)], o, xx)
                sp[(p, v)].append(compute_spaef(o, xx))

    out = {}
    for p in ('red', 'clim'):
        out[p] = {'corr': float(np.mean(corrs[p]))}
        for v in ('raw', 'orac', 'regr'):
            out[p][f'r2_{v}'] = r2acc(accs[(p, v)])
            out[p][f'spaef_{v}'], out[p][f'spaef_n_{v}'] = spaef_mean(sp[(p, v)])
    out['n_fechas'] = len(tdates)
    out['n_teselas'] = len(items)
    out['n_pixeles'] = n_px
    out['cobertura_climatologia'] = comun_px / max(base_px, 1)
    return out


# ======================================================================
# 2-4. C
# ======================================================================
def c_items(z, pos, train_cov, clim_patch):
    """Mascara de spatial_models_vs_climatology.evaluate (linea a linea)."""
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None
    items = []
    for i, t in enumerate(ids):
        t = str(t)
        if t not in pos:
            continue
        r, c = pos[t]
        if train_cov[r:r + TILE, c:c + TILE].any():
            continue
        cp = clim_patch(int(t[:4]), r, c)
        tg, pr = tgts[i], preds[i]
        m = (tg > SNOW_THRESHOLD) & np.isfinite(tg) & np.isfinite(pr) & np.isfinite(cp)
        if vals is not None:
            m &= vals[i] > 0.5
        if m.sum() >= MIN_VALID_PX:
            items.append((i, t, t[:8], cp, m))
    return items, preds, tgts


def decomp(rows):
    """T1a/T1b/T2/T3 en % del MSE, ponderado por pixeles (decomp_oracle_checks)."""
    n = np.array([r[1] for r in rows], float); w = n / n.sum()
    b = np.array([r[2] for r in rows]); dl = np.array([r[0] for r in rows])
    T1, T2, T3, MSE = (float(np.sum(w * np.array([r[k] for r in rows]))) for k in (3, 4, 5, 6))
    b_d = {d: float(np.sum(n[dl == d] * b[dl == d]) / n[dl == d].sum()) for d in np.unique(dl)}
    bd = np.array([b_d[d] for d in dl])
    T1a, T1b = float(np.sum(w * bd ** 2)), float(np.sum(w * (b - bd) ** 2))
    return {'T1a_pct': 100 * T1a / MSE, 'T1b_pct': 100 * T1b / MSE, 'T2_pct': 100 * T2 / MSE,
            'T3_pct': 100 * T3 / MSE, 'escala_T1_T2_pct': 100 * (T1 + T2) / MSE,
            'suma_pct': 100 * (T1a + T1b + T2 + T3) / MSE, 'n_teselas': len(rows)}


def evaluate_C(z, pos, train_cov, clim_patch, keep_tiles=False):
    items, preds, tgts = c_items(z, pos, train_cov, clim_patch)
    acc_d = {}
    for i, t, d, cp, m in items:
        s = acc_d.setdefault(d, [0.0, 0.0, 0.0, 0])
        s[0] += float(tgts[i][m].sum()); s[1] += float(preds[i][m].sum())
        s[2] += float(cp[m].sum()); s[3] += int(m.sum())
    real = {d: v[0] / v[3] for d, v in acc_d.items()}
    med = {'mod': {d: v[1] / v[3] for d, v in acc_d.items()},
           'clim': {d: v[2] / v[3] for d, v in acc_d.items()}}

    keys = ('mod_raw', 'mod_orac', 'clim_raw', 'clim_orac')
    accs = {k: new_acc() for k in keys}
    sp = {k: [] for k in keys}
    corr = {'mod': [], 'clim': []}
    drows = {'mod': [], 'clim': []}
    per_tile = []
    for i, t, d, cp, m in items:
        o = tgts[i][m].astype(np.float64)
        pm = preds[i][m].astype(np.float64)
        pc = cp[m].astype(np.float64)
        pm_o = pm + (real[d] - med['mod'][d])
        pc_o = pc + (real[d] - med['clim'][d])
        xs = {'mod_raw': pm, 'mod_orac': pm_o, 'clim_raw': pc, 'clim_orac': pc_o}
        for k, x in xs.items():
            add(accs[k], o, x)
            v = compute_spaef(o, x)
            if not np.isnan(v):
                sp[k].append(v)
        for k, x in (('mod', pm), ('clim', pc)):
            dt = decompose_tile(o, x)
            if dt is not None:
                corr[k].append(dt['rho'])
                drows[k].append((d, dt['n'], dt['mu_s'] - dt['mu_o'], dt['t1'], dt['t2'], dt['t3'], dt['mse']))
        if keep_tiles:
            per_tile.append({'tile_id': t, 'fecha': d, 'n_nieve': int(m.sum()),
                             'frac_nieve': float(m.sum()) / (TILE * TILE),
                             'r2_orac': r2_arr(o, pm_o)})
    out = {'n_tiles': len(items), 'n_pixeles': int(sum(m.sum() for *_, m in items)),
           'r2_raw': r2acc(accs['mod_raw']), 'r2_orac': r2acc(accs['mod_orac']),
           'spaef_raw': float(np.mean(sp['mod_raw'])), 'spaef_orac': float(np.mean(sp['mod_orac'])),
           'techo_r2': r2acc(accs['clim_orac']), 'techo_spaef': float(np.mean(sp['clim_orac'])),
           'clim_r2_raw': r2acc(accs['clim_raw']), 'clim_spaef_raw': float(np.mean(sp['clim_raw'])),
           'corr_mod': float(np.mean(corr['mod'])), 'corr_clim': float(np.mean(corr['clim'])),
           'decomp_mod': decomp(drows['mod']), 'decomp_clim_raw': decomp(drows['clim'])}
    out['frac_r2'] = out['r2_orac'] / out['techo_r2']
    out['frac_spaef'] = out['spaef_orac'] / out['techo_spaef']
    out['frac_r2_raw'] = out['r2_raw'] / out['clim_r2_raw']
    out['frac_spaef_raw'] = out['spaef_raw'] / out['clim_spaef_raw']
    return out, per_tile, (items, preds, tgts, real, med)


def field_orac(item, preds, real, med, which):
    """Campo completo de la tesela con nivel perfecto; NaN fuera de la mascara."""
    i, t, d, cp, m = item
    if which == 'clim':
        f = cp + (real[d] - med['clim'][d])
    else:
        f = preds[i].astype(np.float64) + (real[d] - med['mod'][d])
    return np.where(m, f, np.nan)


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description='Cifras de Resultados 5.2 y tesela de la figura')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--spatial-net', nargs=3, required=True, help='npz red espacial s7 s42 s123')
    ap.add_argument('--spatial-rf', nargs=3, required=True, help='npz RF justo espacial s7 s42 s123')
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--skip-b', action='store_true')
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/cifras_canonicas_5_2.json')
    ap.add_argument('--fig-npz', default='analysis/fig05_tesela.npz')
    ap.add_argument('--nd', type=int, default=4,
                    help='decimales de salida (6 para cifras_canonicas_5_2_6d.json)')
    args = ap.parse_args()
    set_nd(args.nd)
    t0 = time.time()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    H, W = int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE
    out = {'descripcion': 'Cifras canonicas de Resultados 5.2 (scripts/results_5_2.py). '
                          'Red: semillas 7, 42, 123 (por_semilla en ese orden).'}

    # ------------------------------------------------------------ B
    if not args.skip_b:
        cache = TileCache(args.masks)
        B, ctrl = {}, []
        refB = json.load(open('analysis/eB_climatology_v2.json', encoding='utf-8')) \
            if os.path.exists('analysis/eB_climatology_v2.json') else {}
        refD = json.load(open('analysis/e5_regresor_dependientes.json', encoding='utf-8')) \
            if os.path.exists('analysis/e5_regresor_dependientes.json') else {}
        for fold in args.folds:
            print(f'B fold {fold}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
            clim, _ = build_climatology(geo, fold, args.masks, (H, W))
            df = pd.read_csv(FOLD_CSV.format(fold=fold))
            tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
            trm, _, _ = means_by_date_masks(tr_ids, cache, None)       # todas las teselas
            model = fit_regressor(trm, sig)[0]
            paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                       f'resunetpp_v3_loyo{fold}_predictions.npz')}
            for s in ('42', '123'):
                paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                        f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')
            per, cl = {}, None
            for s in SEEDS:
                z = np.load(paths[s])
                r = evaluate_B(z['tile_ids'], z['preds'], z['targets'],
                               z['valids'] if 'valids' in z.files else None, pos, clim, sig, model)
                per[s] = r
                if cl is None:
                    cl = r
                del z
            red = {k: mmm([per[s]['red'][k] for s in SEEDS])
                   for k in ('r2_raw', 'r2_orac', 'r2_regr', 'spaef_raw', 'spaef_orac', 'spaef_regr', 'corr')}
            B[fold] = {'n_fechas': cl['n_fechas'], 'n_teselas': cl['n_teselas'], 'n_pixeles': cl['n_pixeles'],
                       'cobertura_climatologia': round(cl['cobertura_climatologia'], ND),
                       'red': red,
                       'climatologia': {k: round(cl['clim'][k], ND) for k in
                                        ('r2_raw', 'r2_orac', 'r2_regr', 'spaef_raw', 'spaef_orac', 'spaef_regr', 'corr')}}
            if fold in refB:
                for s in SEEDS:
                    o = refB[fold]['red_por_semilla'][s]
                    ctrl += [abs(o['r2_raw'] - per[s]['red']['r2_raw']), abs(o['r2_oraculo'] - per[s]['red']['r2_orac']),
                             abs(o['corr'] - per[s]['red']['corr'])]
                ctrl += [abs(refB[fold]['climatologia']['r2_raw'] - cl['clim']['r2_raw']),
                         abs(refB[fold]['climatologia']['r2_oraculo'] - cl['clim']['r2_orac'])]
            dep = refD.get('parte1_B', {}).get('por_variante', {}).get('todas', {}).get(fold)
            if dep:
                for s in SEEDS:
                    ctrl.append(abs(dep['red_por_semilla'][s] - per[s]['red']['r2_regr']))
                ctrl.append(abs(dep['climatologia'] - cl['clim']['r2_regr']))
            print(f"  red R2 raw/orac/regr {red['r2_raw']['media']:.3f}/{red['r2_orac']['media']:.3f}/"
                  f"{red['r2_regr']['media']:.3f}  clim {cl['clim']['r2_raw']:.3f}/{cl['clim']['r2_orac']:.3f}/"
                  f"{cl['clim']['r2_regr']:.3f}", flush=True)
        keys = ('r2_raw', 'r2_orac', 'r2_regr', 'spaef_raw', 'spaef_orac', 'spaef_regr', 'corr')
        B['MEDIA_4_FOLDS'] = {
            'red': {k: round(float(np.mean([B[f]['red'][k]['media'] for f in args.folds])), ND) for k in keys},
            'climatologia': {k: round(float(np.mean([B[f]['climatologia'][k] for f in args.folds])), ND) for k in keys}}
        out['1_B'] = {'fuente': 'scripts/results_5_2.py (mascara y R2 de scripts/climatology_baseline.py; '
                                'regresor de E5 con la media de train de todas las teselas)',
                      'definiciones': 'raw: sin corregir; orac: nivel perfecto aditivo; regr: nivel del '
                                      'regresor corregido; SPAEF medio por tesela; corr: media por tesela',
                      'por_fold': B,
                      'control': {'valores': len(ctrl), 'max_diferencia': max(ctrl) if ctrl else None,
                                  'contra': 'analysis/eB_climatology_v2.json (raw, orac, corr) y '
                                            'analysis/e5_regresor_dependientes.json (regr)'}}
        print(f"  control B: {out['1_B']['control']}", flush=True)

    # ------------------------------------------------------------ C
    print(f'C: climatologia por anyo  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    geo['year'] = geo['date'].str[:4].astype(int)
    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in geo.loc[geo['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True
    ysum, ycnt, tsum, tcnt = build_year_climatology(geo, args.masks, H, W)

    def clim_patch(year, r, c):
        s = tsum[r:r + TILE, c:c + TILE] - ysum[year][r:r + TILE, c:c + TILE]
        n = tcnt[r:r + TILE, c:c + TILE] - ycnt[year][r:r + TILE, c:c + TILE]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    C = {'red': [], 'rf': []}
    keep = {}
    for label, paths in (('red', args.spatial_net), ('rf', args.spatial_rf)):
        for p in paths:
            print(f'  {label}: {os.path.basename(p)}', flush=True)
            z = np.load(p)
            is42 = '_s42' in os.path.basename(p)
            r, per_tile, ctx = evaluate_C(z, pos, train_cov, clim_patch, keep_tiles=(label == 'red' and is42))
            r['archivo'] = os.path.basename(p)
            C[label].append(r)
            if is42:
                keep[label] = (per_tile, ctx)
            else:
                del z

    def blk(label):
        rs = C[label]
        return {k: mmm([x[k] for x in rs]) for k in
                ('r2_raw', 'r2_orac', 'spaef_raw', 'spaef_orac', 'corr_mod',
                 'frac_r2', 'frac_spaef', 'frac_r2_raw', 'frac_spaef_raw')}
    t = C['red'][0]
    techo = {'r2_orac': round(t['techo_r2'], ND), 'spaef_orac': round(t['techo_spaef'], ND),
             'r2_raw': round(t['clim_r2_raw'], ND), 'spaef_raw': round(t['clim_spaef_raw'], ND),
             'corr': round(t['corr_clim'], ND)}
    ref = json.load(open('analysis/eC_net_vs_rf_allpix.json', encoding='utf-8')) \
        if os.path.exists('analysis/eC_net_vs_rf_allpix.json') else {}
    cc = []
    for label in ('red', 'rf'):
        for a, b in zip(C[label], ref.get(label, [])):
            cc += [abs(a[k] - b[k]) for k in ('r2_raw', 'r2_orac', 'spaef_raw', 'spaef_orac',
                                               'techo_r2', 'techo_spaef')]
    out['2_C'] = {'fuente': 'scripts/results_5_2.py (mismo calculo que scripts/spatial_models_vs_climatology.py)',
                  'mascara': '702 teselas sin fuga; pixeles con nieve, dato valido, prediccion finita y '
                             'climatologia de los demas anyos',
                  'n_teselas': t['n_tiles'], 'n_pixeles': t['n_pixeles'],
                  'techo_climatologia': techo, 'red': blk('red'), 'rf_justo': blk('rf'),
                  'control': {'valores': len(cc), 'max_diferencia': max(cc) if cc else None,
                              'contra': 'analysis/eC_net_vs_rf_allpix.json'}}
    print(f"  control C: {out['2_C']['control']}", flush=True)

    # ------------------------------------------------------------ decomposition
    def dblk(label, key='decomp_mod'):
        return {k: mmm([x[key][k] for x in C[label]], ND_PCT) for k in
                ('T1a_pct', 'T1b_pct', 'T2_pct', 'T3_pct', 'escala_T1_T2_pct', 'suma_pct')}
    out['3_descomposicion_espacial_mascara_C'] = {
        'fuente': 'scripts/results_5_2.py (como scripts/decomp_oracle_checks.py, en la mascara de C)',
        'red': dblk('red'), 'rf_justo': dblk('rf'),
        'climatologia_sin_corregir': {k: round(C['red'][0]['decomp_clim_raw'][k], ND_PCT)
                                      for k in ('T1a_pct', 'T1b_pct', 'T2_pct', 'T3_pct', 'escala_T1_T2_pct')},
        'nota': 'analysis/decomp_espacial.json usa la mascara del hallazgo 8 (sin exigir climatologia)'}

    # ------------------------------------------------------------ figure tile
    per_tile, (items, preds, tgts, real, med) = keep['red']
    r2s = np.array([p['r2_orac'] for p in per_tile])
    med_all = float(np.nanmedian(r2s))
    cand = [k for k, p in enumerate(per_tile) if p['frac_nieve'] >= 0.5 and np.isfinite(p['r2_orac'])]
    k_sel = min(cand, key=lambda k: abs(per_tile[k]['r2_orac'] - med_all))
    sel = per_tile[k_sel]
    it = items[k_sel]
    _, ctx_rf = keep['rf']
    items_rf, preds_rf, tgts_rf, real_rf, med_rf = ctx_rf
    k_rf = next(j for j, x in enumerate(items_rf) if x[1] == sel['tile_id'])
    it_rf = items_rf[k_rf]
    if not np.array_equal(it_rf[4], it[4]):
        sys.exit('STOP: la mascara de la tesela elegida difiere entre red y RF.')
    i = it[0]; m = it[4]
    obs = np.where(m, tgts[i].astype(np.float64), np.nan)
    fields = {'observado': obs,
              'climatologia': field_orac(it, preds, real, med, 'clim'),
              'red_s42': field_orac(it, preds, real, med, 'mod'),
              'rf_s42': field_orac(it_rf, preds_rf, real_rf, med_rf, 'mod')}
    o = obs[m]
    met = {k: {'R2': r2_arr(o, f[m]), 'SPAEF': float(compute_spaef(o, f[m]))}
           for k, f in fields.items() if k != 'observado'}
    np.savez_compressed(args.fig_npz, mask=m, **fields)
    out['4_tesela_figura'] = {
        'regla': 'R2 por tesela de la red s42 con nivel perfecto mas cercano a la mediana de todas las '
                 'tesela-fecha del test espacial (mascara de C), con >= 50 % de los 256 x 256 pixeles '
                 'con nieve dentro de la mascara',
        'mediana_r2_todas': round(med_all, ND), 'n_tesela_fecha': len(per_tile),
        'n_candidatas_50pct': len(cand),
        'tile_id': sel['tile_id'], 'fecha': sel['fecha'], 'frac_nieve': round(sel['frac_nieve'], ND),
        'r2_red_s42_orac': round(sel['r2_orac'], ND),
        'metricas_tesela_nivel_perfecto': {k: {kk: round(vv, ND) for kk, vv in v.items()} for k, v in met.items()},
        'campos': args.fig_npz}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False,
                  default=lambda x: x.item() if hasattr(x, 'item') else str(x))

    # ------------------------------------------------------------ print
    print('=' * 72)
    if '1_B' in out:
        print('B (red media de 3 semillas | climatologia): R2 raw/orac/regr ; SPAEF raw/orac/regr ; corr')
        for f in args.folds + ['MEDIA_4_FOLDS']:
            b = out['1_B']['por_fold'][f]
            if f == 'MEDIA_4_FOLDS':
                rr, cl, extra = b['red'], b['climatologia'], ''
                g = lambda d, k: d[k]
            else:
                rr, cl = b['red'], b['climatologia']
                extra = f"  ({b['n_fechas']} fechas, {b['n_teselas']} teselas)"
                g = lambda d, k: d[k]['media'] if isinstance(d[k], dict) else d[k]
            print(f"  {f:<14} red " + ' '.join(f"{g(rr, k):.3f}" for k in
                  ('r2_raw', 'r2_orac', 'r2_regr', 'spaef_raw', 'spaef_orac', 'spaef_regr', 'corr')) +
                  ' | clim ' + ' '.join(f"{cl[k]:.3f}" for k in
                  ('r2_raw', 'r2_orac', 'r2_regr', 'spaef_raw', 'spaef_orac', 'spaef_regr', 'corr')) + extra)
        print(f"  control B: {out['1_B']['control']['max_diferencia']}")
    c2 = out['2_C']
    print(f"C techo: {c2['techo_climatologia']}")
    for lab in ('red', 'rf_justo'):
        b = c2[lab]
        print(f"  {lab:<9} R2 raw {b['r2_raw']['media']:.3f} orac {b['r2_orac']['media']:.3f} | SPAEF raw "
              f"{b['spaef_raw']['media']:.3f} orac {b['spaef_orac']['media']:.3f} | frac R2 "
              f"{100 * b['frac_r2']['media']:.1f}% [{100 * b['frac_r2']['min']:.1f}-{100 * b['frac_r2']['max']:.1f}]"
              f" SPAEF {100 * b['frac_spaef']['media']:.1f}% [{100 * b['frac_spaef']['min']:.1f}-"
              f"{100 * b['frac_spaef']['max']:.1f}]")
    print(f"  control C: {c2['control']['max_diferencia']}")
    d3 = out['3_descomposicion_espacial_mascara_C']['red']
    print('Descomposicion red (mascara C): ' + ', '.join(f"{k} {v['media']:.1f}" for k, v in d3.items()))
    t4 = out['4_tesela_figura']
    print(f"Tesela figura: {t4['tile_id']} ({t4['fecha']}), R2 red s42 {t4['r2_red_s42_orac']:.3f}, "
          f"mediana {t4['mediana_r2_todas']:.3f}, nieve {100 * t4['frac_nieve']:.0f}%")
    print(f"  metricas: {t4['metricas_tesela_nivel_perfecto']}")
    print(f'Guardado en: {args.out} y {args.fig_npz}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
