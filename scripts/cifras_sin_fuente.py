r"""
Cifras del articulo que no tenian JSON de respaldo (sin GPU)
============================================================

Calcula y guarda en analysis/cifras_sin_fuente.json, cada una con su
definicion y su origen, las cifras que el articulo citaba sin un JSON que
las respaldara (revision del 12 de octubre de 2026):

    1. r entre el tamano del entrenamiento y la anomalia del anyo de test en
       los folds LOYO sin submuestrear (04_methods.tex:313). Misma logica que
       scripts/generate_loyo_configs.py (se importan sus funciones).
    2. Fraccion de escala con la forma de Murphy (1988):
           MSE = (mu_s - mu_o)^2 + (sigma_s - rho sigma_o)^2 + sigma_o^2 (1 - rho^2)
       escala = (sesgo + sesgo condicional) / MSE, por tesela y ponderada por
       pixeles, con los filtros de analysis/e1_error_decomposition.py; junto a
       la forma de Gupta et al. (2009) que usa el articulo (04_methods.tex:554).
       Se calcula para E1 (temporal y espacial, semilla 7) y para los 12
       modelos LOYO-E2.
    3. SPAEF del RF antes y despues de corregir la formula (04_methods.tex:821):
       metrics.json de results/rf_v6_s*/ (despues) y las copias de
       RF_v6_backups_pre_recalculo/ (antes). La copia de la semilla 42 esta
       truncada: el SPAEF se lee con una expresion regular.
    4. Pixeles validos excluidos de las metricas (HS <= 0.01 m) en el test
       espacial (702 teselas sin fuga, y 730) y en el de 2025, en total y
       por fecha (04_methods.tex:27-29).
    5. Cota del desfase sub-pixel (03_study_area_data.tex:316,
       04_methods.tex:825): mismo calculo que scripts/check_subpixel_shift.py,
       pero el origen de cada vuelo se lee de las etiquetas GeoTIFF sin
       rasterio (bloqueado en el PC del trabajo). Comprobado contra rasterio:
       mismos desplazamientos.
    6. R2 del RF temporal a precision completa desde sus .npz (los
       metrics.json guardan 4 decimales; 05_results.tex:16 y 38).
    7. SPAEF de E4b a precision completa en los dos casos en que el valor de
       metrics.json (4 decimales) acaba en 5 y no decide el redondeo a 3
       (05_results.tex:344 y 349). Se recalcula desde los .npz de
       predicciones con el procedimiento de training/evaluate.py (funcion
       spaef_like_evaluate) y se comprueba que redondeado a 4 da el
       metrics.json.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/cifras_sin_fuente.py ^
        --seed7-dir F:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir F:\uniovi\mapunet\experimentos\E2b_semillas ^
        --e1-dir F:\uniovi\mapunet\experimentos\E1_descomposicion ^
        --rf-temporal-dir F:\uniovi\mapunet\experimentos\E_rf_temporal_allpix\train ^
        --rf-backups F:\uniovi\mapunet\experimentos\RF_v6_backups_pre_recalculo ^
        --e4b-dir F:\uniovi\mapunet\experimentos\E4b_loyo_tres_brazos
"""

import argparse
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'analysis'))
from e1_error_decomposition import load_npz, tile_mask, decompose_tile, MIN_VALID_PIXELS  # noqa: E402

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
SNOW = 0.01


def r6(x):
    return None if x is None else round(float(x), 6)


# ----------------------------------------------------------------------
# 1. r tamano de entrenamiento - anomalia
# ----------------------------------------------------------------------
def block_r(args):
    import generate_loyo_configs as g
    df = pd.read_csv(g.CSV_IN)
    df = df[df['source'] == 'lidar'].reset_index(drop=True)
    hs = g.mean_hs_by_year(df, g.MASKS_DIR)            # muestra de 150 teselas por anyo, seed 0
    rows = []
    for y in g.TEST_YEARS:
        fold, val_year = g.build_fold(df, y, train_cap=None)
        train_years = [a for a in g.ALL_YEARS if a not in (y, val_year)]
        n_train = int((fold['exp_temporal_split'] == 'train').sum())
        hs_train = float(np.mean([hs[a] for a in train_years]))
        rows.append({'fold': str(y), 'n_train_sin_submuestrear': n_train,
                     'anomalia_muestreo_pct': r6(100 * (hs[y] - hs_train) / hs_train)})
    sc = json.load(open(os.path.join(_REPO, 'analysis', 'e2_scale_checks.json'), encoding='utf-8'))
    for r in rows:
        r['anomalia_exacta_pct'] = sc['anomalia'][r['fold']]['anomalia_pct']
    n = [r['n_train_sin_submuestrear'] for r in rows]
    a_m = [abs(r['anomalia_muestreo_pct']) for r in rows]
    a_e = [abs(r['anomalia_exacta_pct']) for r in rows]
    return {
        'definicion': 'Pearson entre |anomalia| y tamano del train de cada fold LOYO SIN el '
                      'submuestreo a 2659 teselas (como scripts/generate_loyo_configs.py, '
                      'que lo imprime pero no lo guardaba)',
        'por_fold': rows,
        'r_abs_anomalia_muestreo': r6(np.corrcoef(a_m, n)[0, 1]),
        'r_abs_anomalia_exacta': r6(np.corrcoef(a_e, n)[0, 1]),
        'n_folds': len(rows),
        'fuente': 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv, masks/ y analysis/e2_scale_checks.json'}


# ----------------------------------------------------------------------
# 2. Murphy frente a Gupta
# ----------------------------------------------------------------------
def murphy_gupta(path):
    d = load_npz(path)
    acc = np.zeros(6)          # mse, t1, t2_gupta, t3_gupta, cond_bias_murphy, resid_murphy
    w = 0
    for i in range(len(d['tile_ids'])):
        v = d['valids'][i] if d['valids'] is not None else None
        m = tile_mask(d['targets'][i], v, 'snow')
        if m.sum() < MIN_VALID_PIXELS:
            continue
        dec = decompose_tile(d['targets'][i][m], d['preds'][i][m])
        if dec is None:
            continue
        cb = (dec['sd_s'] - dec['rho'] * dec['sd_o']) ** 2
        rs = dec['sd_o'] ** 2 * (1 - dec['rho'] ** 2)
        acc += dec['n'] * np.array([dec['mse'], dec['t1'], dec['t2'], dec['t3'], cb, rs])
        w += dec['n']
    acc /= w
    mse, t1, t2, t3, cb, rs = acc
    return {'escala_gupta_pct': r6(100 * (t1 + t2) / mse), 'escala_murphy_pct': r6(100 * (t1 + cb) / mse),
            'reconstruccion_murphy_rel': r6(abs(t1 + cb + rs - mse) / mse),
            'n_pixeles': int(w), 'npz': os.path.basename(path)}


def block_murphy(args):
    out = {}
    e1t = os.path.join(args.e1_dir, 'resunetpp_v3_sp00_s7_predictions.npz')
    e1s = os.path.join(args.e1_dir, 'spatial', 'resunetpp_v3_spatial_sp00_s7',
                       'resunetpp_v3_spatial_sp00_s7_predictions.npz')
    for name, p in (('E1_temporal_test2025_s7', e1t), ('E1_espacial_730_teselas_s7', e1s)):
        if os.path.exists(p):
            print(f'  Murphy {name} ...', flush=True)
            out[name] = murphy_gupta(p)
    loyo = {}
    for f in FOLDS:
        loyo[f] = {}
        for s in SEEDS:
            p = (os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{f}', f'resunetpp_v3_loyo{f}_predictions.npz')
                 if s == '7' else
                 os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{f}_s{s}',
                              f'resunetpp_v3_loyo{f}_s{s}_predictions.npz'))
            if os.path.exists(p):
                print(f'  Murphy LOYO {f} s{s} ...', flush=True)
                loyo[f][s] = murphy_gupta(p)
        if loyo[f]:
            loyo[f]['media'] = {k: r6(np.mean([loyo[f][s][k] for s in SEEDS if s in loyo[f]]))
                                for k in ('escala_gupta_pct', 'escala_murphy_pct')}
    out['LOYO_E2'] = loyo
    out['definicion'] = ('Murphy (1988): MSE = (mu_s-mu_o)^2 + (sigma_s - rho sigma_o)^2 + '
                         'sigma_o^2 (1-rho^2); escala = dos primeros terminos. Gupta et al. (2009): '
                         'MSE = (mu_s-mu_o)^2 + (sigma_s-sigma_o)^2 + 2 sigma_s sigma_o (1-rho); '
                         'escala = dos primeros. Por tesela, ponderado por pixeles, pixeles con '
                         'nieve, filtros de analysis/e1_error_decomposition.py')
    out['nota'] = ('El 69,4 % citado en 04_methods.tex:554 como "split temporal (E1)" es el fold '
                   'LOYO 2025 con la semilla 7 (Gupta 59,5 %), no E1 temporal; el 28,3 % si es E1 '
                   'espacial (730 teselas, Gupta 37,3 %).')
    return out


# ----------------------------------------------------------------------
# 3. SPAEF del RF antes y despues de corregir la formula
# ----------------------------------------------------------------------
def block_rf_spaef(args):
    rows = {}
    for s in SEEDS:
        new = json.load(open(os.path.join(_REPO, 'results', f'rf_v6_s{s}', f'rf_v6_s{s}_metrics.json'),
                             encoding='utf-8'))['test_metrics']
        bk = os.path.join(args.rf_backups, f'rf_v6_s{s}_metrics_backup_pre_recompute.json')
        txt = open(bk, encoding='utf-8').read()
        mt = re.search(r'"SPAEF"\s*:\s*(-?[0-9.]+)', txt)
        try:
            json.loads(txt); integro = True
        except ValueError:
            integro = False
        rows[s] = {'SPAEF_antes': float(mt.group(1)) if mt else None, 'SPAEF_despues': new['SPAEF'],
                   'R2': new['R2'], 'copia_antes_integra': integro}
    ant = [rows[s]['SPAEF_antes'] for s in SEEDS if rows[s]['SPAEF_antes'] is not None]
    des = [rows[s]['SPAEF_despues'] for s in SEEDS]
    return {'definicion': 'RF v6 temporal (pipeline anterior: solo nieve, train + val), test 2025. '
                          'Antes: SPAEF con histogramas de valores; despues: forma canonica con '
                          'z-scores (hallazgo 3). Mismas predicciones (R2 identico).',
            'por_semilla': rows, 'media_antes': r6(np.mean(ant)), 'media_despues': r6(np.mean(des)),
            'precision_origen': '4 decimales (metrics.json)',
            'fuente': 'results/rf_v6_s*/rf_v6_s*_metrics.json; '
                      'RF_v6_backups_pre_recalculo/rf_v6_s*_metrics_backup_pre_recompute.json'}


# ----------------------------------------------------------------------
# 4. Pixeles excluidos (HS <= 0.01 m entre los validos)
# ----------------------------------------------------------------------
def bare_fraction(path, keep=None):
    z = np.load(path)
    ids, tg, va = z['tile_ids'], z['targets'], z['valids']      # una sola lectura de cada array
    tot = bare = 0
    by_date, nt = {}, 0
    for i, t in enumerate(ids):
        t = str(t)
        if keep is not None and not keep(t):
            continue
        nt += 1
        v = (va[i] > 0.5) & np.isfinite(tg[i])
        b = v & (tg[i] <= SNOW)
        tot += int(v.sum()); bare += int(b.sum())
        a = by_date.setdefault(t[:8], [0, 0]); a[0] += int(v.sum()); a[1] += int(b.sum())
    return {'n_teselas': nt, 'pct_excluidos': r6(100 * bare / tot),
            'por_fecha_pct': {d: r6(100 * b / a) for d, (a, b) in sorted(by_date.items())}}


def block_bare(args):
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    H, W = int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE
    cov = np.zeros((H, W), dtype=bool)
    for r, c in geo.loc[geo['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        cov[r:r + TILE, c:c + TILE] = True

    def clean(t):
        if t not in pos:
            return False
        r, c = pos[t]
        return not cov[r:r + TILE, c:c + TILE].any()

    sp = os.path.join(args.e1_dir, 'spatial', 'resunetpp_v3_spatial_sp00_s7',
                      'resunetpp_v3_spatial_sp00_s7_predictions.npz')
    tp = os.path.join(args.e1_dir, 'resunetpp_v3_sp00_s7_predictions.npz')
    print('  pixeles excluidos ...', flush=True)
    return {'definicion': 'pixeles validos con HS <= 0.01 m (fuera de las metricas, que usan solo '
                          'nieve), en % de los validos; los objetivos son los mismos en todas las '
                          'semillas y modelos',
            'espacial_702_teselas': bare_fraction(sp, clean),
            'espacial_730_teselas': bare_fraction(sp),
            'temporal_test_2025': bare_fraction(tp),
            'nota': 'El 3,7 % citado corresponde a las 730 teselas; con las 702 sin fuga es 3,8 %.',
            'fuente': os.path.basename(sp) + ', ' + os.path.basename(tp) + ', data/tile_georef.csv'}


# ----------------------------------------------------------------------
# 5. Desfase sub-pixel, sin rasterio
# ----------------------------------------------------------------------
def geotiff_bounds(path):
    """(left, top, res_x, res_y) desde las etiquetas TIFF (ModelPixelScale y ModelTiepoint)."""
    with open(path, 'rb') as f:
        e = '<' if f.read(2) == b'II' else '>'
        big = struct.unpack(e + 'H', f.read(2))[0] == 43
        if big:
            f.read(4); off = struct.unpack(e + 'Q', f.read(8))[0]
        else:
            off = struct.unpack(e + 'I', f.read(4))[0]
        f.seek(off)
        n = struct.unpack(e + ('Q' if big else 'H'), f.read(8 if big else 2))[0]
        size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8, 17: 8}
        fmt = {1: 'B', 2: 's', 3: 'H', 4: 'I', 5: 'II', 11: 'f', 12: 'd', 16: 'Q', 17: 'q'}
        tags = {}
        for _ in range(n):
            if big:
                tag, typ, cnt = struct.unpack(e + 'HHQ', f.read(12)); raw = f.read(8); lim = 8
            else:
                tag, typ, cnt = struct.unpack(e + 'HHI', f.read(8)); raw = f.read(4); lim = 4
            if tag not in (33550, 33922):
                continue
            nb = size[typ] * cnt
            if nb > lim:
                here = f.tell(); f.seek(struct.unpack(e + ('Q' if big else 'I'), raw)[0])
                data = f.read(nb); f.seek(here)
            else:
                data = raw[:nb]
            tags[tag] = struct.unpack(e + fmt[typ] * cnt, data)
    sx, sy = tags[33550][0], tags[33550][1]
    i, j, _, x, y, _ = tags[33922][:6]
    return x - i * sx, y + j * sy, sx, sy


def block_subpixel(args):
    import check_subpixel_shift as cs        # importa rasterio: se registra un modulo vacio si falta
    snow_dir, dem_path = cs.SNOW_DIR, cs.DEM_PATH
    dl, dt, dsx, dsy = geotiff_bounds(dem_path)
    frac = {}
    for fn in sorted(os.listdir(snow_dir)):
        m = re.match(r'SD_(\d{8})_1m\.tif$', fn)
        if m:
            l, t, _, _ = geotiff_bounds(os.path.join(snow_dir, fn))
            ro, co = (dt - t) / dsy, (l - dl) / dsx       # = rasterio dem.window(*sd.bounds)
            frac[m.group(1)] = (ro - math.floor(ro), co - math.floor(co))
    df = pd.read_csv(cs.CSV_MAIN)
    df['date'] = df['tile_id'].str[:8]
    tot_f, tot_1 = cs.new_acc(), cs.new_acc()
    por_fecha = {}
    print('  desfase sub-pixel ...', flush=True)
    for d, g in df.groupby('date'):
        if d not in frac:
            continue
        fr, fc = frac[d]
        a_f, a_1 = cs.new_acc(), cs.new_acc()
        for t in g['tile_id'].head(args.max_tiles):
            path = os.path.join(cs.MASKS_DIR, t)
            if not os.path.exists(path):
                continue
            mm = np.load(path).astype(np.float64)
            mm = np.where(np.isfinite(mm) & (mm > -100), mm, np.nan)
            s, o = cs.shift_bilinear(mm, fr, fc)
            ok = np.isfinite(s) & np.isfinite(o) & (o > cs.SNOW_THRESHOLD)
            if ok.sum() > 100:
                cs.acc_r2(a_f, o[ok], s[ok]); cs.acc_r2(tot_f, o[ok], s[ok])
            s1, o1 = mm[:-1, :-1], mm[1:, 1:]
            ok1 = np.isfinite(s1) & np.isfinite(o1) & (o1 > cs.SNOW_THRESHOLD)
            if ok1.sum() > 100:
                cs.acc_r2(a_1, o1[ok1], s1[ok1]); cs.acc_r2(tot_1, o1[ok1], s1[ok1])
        if a_f['n']:
            por_fecha[d] = {'desplazamiento_fila': r6(fr), 'desplazamiento_col': r6(fc),
                            'R2_desfase_real': r6(cs.r2_of(a_f)), 'R2_un_pixel': r6(cs.r2_of(a_1))}
    peor = min(por_fecha, key=lambda k: por_fecha[k]['R2_desfase_real'])
    return {'definicion': 'R2 de cada mapa de nieve frente a si mismo desplazado la fraccion de '
                          'origen de su fecha (interpolacion bilineal); cota superior del dano. '
                          f'{args.max_tiles} teselas por fecha, como scripts/check_subpixel_shift.py',
            'R2_total_desfase_real': r6(cs.r2_of(tot_f)), 'R2_total_un_pixel': r6(cs.r2_of(tot_1)),
            'peor_fecha': peor, 'R2_peor_fecha': por_fecha[peor]['R2_desfase_real'],
            'por_fecha': por_fecha,
            'fuente': 'Articulo 1/Data/izas/LiDAR (SD_*_1m.tif, DEMbigIzas_1m.tif: solo cabeceras), '
                      'dataset_v4_ms_sx200/masks'}


# ----------------------------------------------------------------------
# 6. R2 del RF temporal a precision completa
# ----------------------------------------------------------------------
def block_rf_r2(args):
    rows = {}
    for s in SEEDS:
        p = os.path.join(args.rf_temporal_dir, f'rf_temporal_allpix_train_s{s}',
                         f'rf_temporal_allpix_train_s{s}_predictions.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        O, S = [], []
        for t, pr, v in zip(z['targets'], z['preds'], z['valids']):
            m = (v > 0.5) & (t > SNOW)            # como rf_spatial_allpix.predict_test
            if m.sum() >= 10:
                O.append(t[m].astype(np.float64)); S.append(pr[m].astype(np.float64))
        o, x = np.concatenate(O), np.concatenate(S)
        rows[s] = r6(1 - np.sum((o - x) ** 2) / np.sum((o - o.mean()) ** 2))
    v = list(rows.values())
    return {'definicion': 'R2 acumulado sobre pixeles con nieve, test 2025 (como predict_test)',
            'por_semilla': rows, 'media': r6(np.mean(v)) if v else None,
            'min': r6(min(v)) if v else None, 'max': r6(max(v)) if v else None,
            'fuente': 'E_rf_temporal_allpix/train/rf_temporal_allpix_train_s*_predictions.npz'}


# ----------------------------------------------------------------------
# 7. SPAEF de E4b a precision completa (empates del redondeo)
# ----------------------------------------------------------------------
E4B_EMPATES = [
    # (brazo, datos, cifra, semillas que entran, texto)
    ('dualscale', 'full', 'min', ['7'], '05_results.tex:349, "Con senales", 3866 teselas, SPAEF minimo'),
    ('dual', 'sub', 'media', ['7', '42', '123'], '05_results.tex:344, "Doble cabeza", 2284 teselas, SPAEF medio'),
]


def _compute_spaef():
    # utils/metrics.py importa sklearn al cargarse (bloqueado en el PC del trabajo);
    # compute_spaef solo usa numpy, asi que basta un modulo vacio si falla
    try:
        import sklearn.metrics  # noqa: F401
    except Exception:
        import types
        sk, skm = types.ModuleType('sklearn'), types.ModuleType('sklearn.metrics')
        for n in ('mean_absolute_error', 'mean_squared_error', 'r2_score'):
            setattr(skm, n, None)
        sk.metrics = skm
        sys.modules['sklearn'], sys.modules['sklearn.metrics'] = sk, skm
    from utils.metrics import compute_spaef
    return compute_spaef


def spaef_like_evaluate(path):
    """SPAEF medio por tesela como training/evaluate.py (l. 180-208): por tesela,
    pixeles con HS > 0.01 m, al menos 10, se descartan las teselas con SPAEF NaN;
    media sin ponderar. Devuelve (spaef, n_teselas)."""
    compute_spaef = _compute_spaef()
    z = np.load(path)
    tg, pr = z['targets'], z['preds']
    vals = []
    for t, p in zip(tg, pr):
        t, p = t.reshape(-1), p.reshape(-1)
        v = t > SNOW
        if v.sum() < 10:
            continue
        x = compute_spaef(t[v], p[v])
        if not np.isnan(x):
            vals.append(x)
    return float(np.mean(vals)), len(vals)


def block_spaef_e4b(args):
    out = {'definicion': 'SPAEF medio por tesela, sin corregir el nivel, recalculado desde los '
                         '.npz con el procedimiento de training/evaluate.py (spaef_like_evaluate); '
                         'metrics.json solo guarda 4 decimales',
           'casos': []}
    for arm, ds, cifra, seeds, texto in E4B_EMPATES:
        suf = '_full' if ds == 'full' else ''
        por = {}
        for s in seeds:
            name = f'e4b_{arm}{suf}_loyo2025_s{s}'
            npz = os.path.join(args.e4b_dir, f'loyo_e4b_{arm}', name, f'{name}_predictions.npz')
            met = os.path.join(_REPO, 'results', 'norm_v3', f'loyo_e4b_{arm}', name, f'{name}_metrics.json')
            print(f'  SPAEF {name} ...', flush=True)
            sp, nt = spaef_like_evaluate(npz)
            m = json.load(open(met, encoding='utf-8'))
            por[s] = {'SPAEF_metrics_json': m['SPAEF'], 'SPAEF_recalculado': r6(sp), 'n_teselas': nt,
                      'n_teselas_metrics_json': m['SPAEF_n_tiles'],
                      'coincide_a_4_decimales': abs(round(sp, 4) - m['SPAEF']) < 1e-9}
        agg = np.mean if cifra == 'media' else np.min
        out['casos'].append({
            'cifra': texto, 'brazo': arm, 'datos': ds, 'estadistico': cifra, 'semillas': seeds,
            'valor_metrics_json': round(float(agg([por[s]['SPAEF_metrics_json'] for s in seeds])), 6),
            'valor_recalculado': r6(agg([por[s]['SPAEF_recalculado'] for s in seeds])),
            'por_semilla': por,
            'procedimiento': 'training/evaluate.py, l. 180-208 (scripts/cifras_sin_fuente.py: spaef_like_evaluate)'})
    return out


def main():
    ap = argparse.ArgumentParser(description='Cifras sin JSON de respaldo')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--e1-dir', required=True)
    ap.add_argument('--rf-temporal-dir', required=True)
    ap.add_argument('--rf-backups', required=True)
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--max-tiles', type=int, default=60)
    ap.add_argument('--e4b-dir', default=None, help='carpeta E4b_loyo_tres_brazos (bloque 7)')
    ap.add_argument('--skip', nargs='*', default=[],
                    help='bloques a saltar: r murphy rf bare subpixel rfr2 spaef')
    ap.add_argument('--out', default='analysis/cifras_sin_fuente.json')
    args = ap.parse_args()
    sys.modules.setdefault('rasterio', __import__('types').ModuleType('rasterio'))
    t0 = time.time()
    out = {'descripcion': 'Cifras del articulo que no tenian JSON de respaldo (revision del '
                          '12-oct-2026). 6 decimales.'}
    blocks = [('r', '1_r_tamano_train_anomalia', block_r), ('murphy', '2_escala_murphy_frente_a_gupta', block_murphy),
              ('rf', '3_spaef_rf_antes_despues_formula', block_rf_spaef), ('bare', '4_pixeles_excluidos', block_bare),
              ('subpixel', '5_desfase_subpixel', block_subpixel), ('rfr2', '6_r2_rf_temporal_completo', block_rf_r2),
              ('spaef', '7_spaef_e4b_precision_completa', block_spaef_e4b)]
    for key, name, fn in blocks:
        if key in args.skip or (key == 'spaef' and not args.e4b_dir):
            continue
        print(f'{name}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        out[name] = fn(args)

    print('=' * 72)
    b = out.get('1_r_tamano_train_anomalia')
    if b:
        print(f"1. r |anomalia| - n_train: muestreo {b['r_abs_anomalia_muestreo']:+.3f}, "
              f"exacta {b['r_abs_anomalia_exacta']:+.3f}")
    b = out.get('2_escala_murphy_frente_a_gupta')
    if b:
        for k in ('E1_temporal_test2025_s7', 'E1_espacial_730_teselas_s7'):
            if k in b:
                print(f"2. {k}: Gupta {b[k]['escala_gupta_pct']:.1f} %  Murphy {b[k]['escala_murphy_pct']:.1f} %")
        for f, v in b['LOYO_E2'].items():
            if '7' in v:
                print(f"   LOYO {f} s7: Gupta {v['7']['escala_gupta_pct']:.1f} %  Murphy {v['7']['escala_murphy_pct']:.1f} %"
                      f"   | media 3 semillas: {v['media']['escala_gupta_pct']:.1f} / {v['media']['escala_murphy_pct']:.1f}")
    b = out.get('3_spaef_rf_antes_despues_formula')
    if b:
        print(f"3. SPAEF RF: antes {b['media_antes']:.4f}, despues {b['media_despues']:.4f}  {b['por_semilla']}")
    b = out.get('4_pixeles_excluidos')
    if b:
        for k in ('espacial_702_teselas', 'espacial_730_teselas', 'temporal_test_2025'):
            pf = b[k]['por_fecha_pct']
            top = sorted(pf, key=lambda d: -pf[d])[:2]
            print(f"4. {k}: {b[k]['pct_excluidos']:.2f} %  (" + ', '.join(f'{d} {pf[d]:.1f} %' for d in top) + ')')
    b = out.get('5_desfase_subpixel')
    if b:
        print(f"5. desfase: R2 total {b['R2_total_desfase_real']:.4f}, peor fecha {b['peor_fecha']} "
              f"{b['R2_peor_fecha']:.4f}, un pixel {b['R2_total_un_pixel']:.4f}")
    b = out.get('6_r2_rf_temporal_completo')
    if b:
        print(f"6. R2 RF temporal: {b['por_semilla']}  media {b['media']}  min {b['min']}")
    b = out.get('7_spaef_e4b_precision_completa')
    if b:
        for c in b['casos']:
            print(f"7. {c['cifra']}: metrics.json {c['valor_metrics_json']:.4f} -> recalculado "
                  f"{c['valor_recalculado']:.6f}  (teselas {[v['n_teselas'] for v in c['por_semilla'].values()]}, "
                  f"coincide a 4 dec: {all(v['coincide_a_4_decimales'] for v in c['por_semilla'].values())})")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
