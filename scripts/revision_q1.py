r"""
Analisis adicionales para la revision Q1 (sin GPU)
==================================================

Responde con medidas, no con redaccion, a tres objeciones de la revision del
14-oct-2026. No cambia ningun resultado del articulo: anyade evaluaciones que
hoy no se reportan. Escribe analysis/revision_q1.json.

    1_dominio_completo_y_extension   (objecion 1: "no se demuestra donde esta
       la nieve"). Dos partes:
       a) Dominio completo frente a solo nieve: R2, MAE, RMSE y sesgo sobre
          TODOS los pixeles validos (incluido el suelo desnudo) junto a los de
          solo nieve. Se leen de los metrics.json, que ya los guardan
          (R2_full, MAE_full, RMSE_full, Bias_full), para todas las semillas.
       b) Extension de nieve: matriz de confusion observado/predicho con el
          umbral del pipeline (0.01 m) y con 0.05, 0.10 y 0.25 m, sobre los
          pixeles validos. Da precision, exhaustividad (POD), especificidad,
          tasa de falsos positivos, F1, CSI y sesgo de frecuencia. Esto NO
          esta en ningun JSON: se calcula aqui desde los .npz.
       Control: las cifras a 0.01 m deben reproducir snow_precision,
       snow_recall y false_pos_rate de los metrics.json.

    2_componentes_spaef              (objecion 5: "SPAEF no debe ser la unica
       prueba de fidelidad espacial"). Descompone el SPAEF publicado en sus
       tres componentes por tesela: alpha (correlacion de Pearson), beta
       (cociente de coeficientes de variacion) y gamma (interseccion de
       histogramas de z-scores), con la notacion del articulo (Koch 2018).
       Misma mascara que el SPAEF publicado (pixeles con nieve de cada tesela,
       >= 10 pixeles). Control: recomponer SPAEF desde los componentes debe
       dar el valor del metrics.json.

    3_dispersion_por_fecha_C         (objecion 3: "las semillas no son
       validaciones espaciales independientes"). En el split espacial, R2 y
       SPAEF por FECHA (27 fechas, unidades independientes) para la red, el RF
       justo y la climatologia, sobre la mascara de C. Da mediana y
       percentiles entre fechas, y en cuantas fechas gana cada modelo. Es la
       incertidumbre "a partir de unidades relevantes" que pide la revision,
       y no sustituye a repetir el split con otras bandas.

Mascaras, umbrales y SPAEF son los del pipeline (training/evaluate.py,
utils/metrics.py, scripts/results_5_2.py); nada se redefine aqui.

Control (bloque 'control' de cada ejecucion en 1b_2_por_ejecucion): se buscan
TODOS los metrics.json del experimento, no solo el de al lado del .npz, porque
un experimento puede tener varios (entrenamientos distintos de la misma
configuracion). Se informa de cada uno con su timestamp y sus valores, de con
cual coincide el .npz y, si discrepan entre si, se deja un aviso explicito.
En el split temporal la semilla 7 se toma del entrenamiento que exporto
predicciones; el otro queda registrado en 1a .. temporal_red.nota_s7.

No modifica nada existente. Escribe solo el JSON de --out.

Uso (desde la raiz del repo)
----------------------------
    python scripts/revision_q1.py ^
        --spatial-net <s7.npz> <s42.npz> <s123.npz> ^
        --spatial-rf  <s7.npz> <s42.npz> <s123.npz> ^
        --temporal-net <s7.npz> ^
        --temporal-rf <s7.npz> <s42.npz> <s123.npz> ^
        [--seed7-dir ... --seeds-dir ... --con-loyo]

--con-loyo anyade los 12 .npz de LOYO a los bloques 1b y 2 (son unos 5 GB de
lectura; sin el, el bloque 1b cubre el split temporal y el espacial, que son
los de tab:baseline y tab:unsurveyed).
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
# numpy; en el PC del trabajo el Control de aplicaciones bloquea las DLL de
# scipy. Mismo apanyo que en scripts/results_5_2.py: no cambia ningun calculo.
try:
    import sklearn.metrics  # noqa: F401
except Exception:
    import types
    _sk, _skm = types.ModuleType('sklearn'), types.ModuleType('sklearn.metrics')
    for _n in ('mean_absolute_error', 'mean_squared_error', 'r2_score'):
        setattr(_skm, _n, None)
    _sk.metrics = _skm
    sys.modules['sklearn'], sys.modules['sklearn.metrics'] = _sk, _skm
    print('[aviso] sklearn no disponible: modulo vacio (no afecta al calculo)')

from utils.metrics import compute_spaef                                      # noqa: E402
from spatial_models_vs_climatology import build_year_climatology             # noqa: E402

TILE = 256
SNOW = 0.01                      # umbral snow-only del pipeline
THRESHOLDS = [0.01, 0.05, 0.10, 0.25]
MIN_PX = 10                      # teselas con menos pixeles con nieve: fuera del SPAEF
SEEDS = ['7', '42', '123']
FOLDS = ['2021', '2022', '2023', '2025']


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def mmm(v, nd=6):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not v.size:
        return None
    return {'media': round(float(v.mean()), nd), 'min': round(float(v.min()), nd),
            'max': round(float(v.max()), nd), 'por_semilla': [round(float(x), nd) for x in v]}


def r2(o, p):
    ss = float(np.sum((o - o.mean()) ** 2))
    return 1.0 - float(np.sum((o - p) ** 2)) / ss if ss > 0 else float('nan')


def spaef_components(o, s):
    """alpha, beta, gamma del SPAEF con la notacion del articulo (Koch 2018).

    Replica utils.metrics.compute_spaef paso a paso: alpha = correlacion,
    beta = cociente de CV, gamma = interseccion de histogramas de z-scores
    (100 clases). El recorte sim >= 0 es el del pipeline.
    """
    o = np.asarray(o, np.float64)
    s = np.maximum(np.asarray(s, np.float64), 0.0)
    if o.size < MIN_PX:
        return None
    alpha = float(np.corrcoef(o, s)[0, 1])
    mo, ms = float(o.mean()), float(s.mean())
    so, ss = float(o.std()), float(s.std())
    if not np.isfinite(alpha) or mo == 0.0 or ms == 0.0 or so == 0.0 or ss == 0.0:
        return None
    beta = (ss / ms) / (so / mo)
    oz, sz = (o - mo) / so, (s - ms) / ss
    lo, hi = min(oz.min(), sz.min()), max(oz.max(), sz.max())
    if hi <= lo:
        return None
    bins = np.linspace(lo, hi, 101)
    ho = np.histogram(oz, bins=bins)[0]; hs = np.histogram(sz, bins=bins)[0]
    ho = ho / (ho.sum() + 1e-10); hs = hs / (hs.sum() + 1e-10)
    gamma = float(np.sum(np.minimum(ho, hs)))
    # terminos al cuadrado: su SUMA es (1-SPAEF)^2, asi que promediar cada uno
    # por tesela da la descomposicion aditiva correcta. Promediar primero el
    # componente y elevar despues NO lo es (desigualdad de Jensen).
    t_a, t_b, t_g = (alpha - 1) ** 2, (beta - 1) ** 2, (gamma - 1) ** 2
    spaef = 1.0 - float(np.sqrt(t_a + t_b + t_g))
    return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'spaef': spaef,
            'term_alpha2': t_a, 'term_beta2': t_b, 'term_gamma2': t_g,
            'suma_terminos': t_a + t_b + t_g}


def met_get(met, *names):
    """Valor de la primera clave que exista, en la raiz o en 'test_metrics'.

    Los metrics.json no comparten nombres: la red usa R2/SPAEF, el RF espacial
    R2_snow_730/SPAEF_snow_730 y el RF v6 los anida en 'test_metrics'. Si no
    hay ninguna, devuelve None y esa comparacion se omite (el control nunca
    debe romper la ejecucion).
    """
    for d in (met, met.get('test_metrics', {}) if isinstance(met, dict) else {}):
        for n in names:
            if isinstance(d, dict) and n in d:
                return d[n]
    return None


def find_metrics(npz_path, roots=('results',)):
    """TODOS los metrics.json del experimento al que pertenece el .npz.

    Un mismo experimento puede tener mas de un metrics.json. Caso real:
    resunetpp_v3_sp00_s7 se entreno dos veces, en julio (results/norm_v3/
    lambda/resunetpp_v3_sp00_s7/, R2 0.2134) y en agosto con el config
    ..._preds.yaml para exportar predicciones (R2 0.2259, el del .npz). Buscar
    un unico fichero y quedarse con el que cuadre ocultaria precisamente la
    discrepancia que este control debe detectar, asi que se devuelven todos.
    """
    stem = os.path.basename(npz_path).replace('_predictions.npz', '')
    name = stem + '_metrics.json'
    aqui = os.path.dirname(npz_path)
    cands = [os.path.join(aqui, name), os.path.join(aqui, stem + '_preds', name)]
    for root in roots:
        if os.path.isdir(root):
            for dp, _dn, fn in os.walk(root):
                if name in fn:
                    cands.append(os.path.join(dp, name))
    vistos, salida = set(), []
    for c in cands:
        k = os.path.normpath(os.path.abspath(c))
        if k not in vistos and os.path.exists(k):
            vistos.add(k)
            salida.append(c)
    return salida


# claves equivalentes por metrica: la red usa R2/SPAEF, el RF espacial
# R2_snow_730/SPAEF_snow_730 y el RF v6 lo anida en 'test_metrics'
CLAVES = {'R2': ('R2', 'R2_snow_730'), 'R2_full': ('R2_full',),
          'MAE': ('MAE', 'MAE_snow_730'), 'RMSE': ('RMSE',),
          'SPAEF': ('SPAEF', 'SPAEF_snow_730'), 'SPAEF_n_tiles': ('SPAEF_n_tiles',)}


def control_metrics(cands, mio):
    """Compara lo recalculado del .npz con CADA metrics.json del experimento.

    Devuelve una fila por fichero (ruta, timestamp, valores y diferencias),
    senyala con que fichero coincide el .npz y, si hay varios que no dicen lo
    mismo, deja un aviso explicito en vez de elegir en silencio el que cuadra.
    """
    filas = []
    for c in cands:
        met = json.load(open(c, encoding='utf-8'))
        pares = [(mio.get(k), met_get(met, *CLAVES[k]), k)
                 for k in ('R2', 'R2_full', 'SPAEF', 'SPAEF_n_tiles')]
        dif = {k: r6(abs(round(float(a), 4) - float(b)))
               for a, b, k in pares if a is not None and b is not None}
        flo = {k: v for k, v in dif.items() if k != 'SPAEF_n_tiles'}
        filas.append({'ruta': c.replace('\\', '/'),
                      'timestamp': met.get('timestamp'),
                      'valores': {k: met_get(met, *v) for k, v in CLAVES.items()
                                  if met_get(met, *v) is not None},
                      'diferencias': dif,
                      'max_diferencia': r6(max(flo.values())) if flo else None,
                      'diferencia_n_teselas_spaef': dif.get('SPAEF_n_tiles')})
    ctrl = {'n_metrics_json': len(filas), 'metrics_json': filas}
    conm = [f for f in filas if f['max_diferencia'] is not None]
    if conm:
        mejor = min(conm, key=lambda f: f['max_diferencia'])
        ctrl['coincide_con'] = mejor['ruta']
        ctrl['max_diferencia_frente_a_metrics_json'] = mejor['max_diferencia']
        ctrl['diferencia_n_teselas_spaef'] = mejor['diferencia_n_teselas_spaef']
    # discrepancia entre ficheros del mismo experimento: entrenamientos distintos
    disc = {}
    for k in CLAVES:
        vs = {f['ruta']: f['valores'][k] for f in filas if k in f['valores']}
        if len(set(vs.values())) > 1:
            disc[k] = vs
    if disc:
        ctrl['metrics_json_multiples'] = disc
        ctrl['aviso'] = ('hay mas de un metrics.json para este experimento y no '
                         'dicen lo mismo: son entrenamientos distintos de la misma '
                         'configuracion. El .npz corresponde al senyalado en '
                         '"coincide_con"; la tabla del articulo debe citar ese mismo.')
    return ctrl


def confusion(obs, pred, thr):
    """Extension de nieve: observado y predicho por encima del umbral."""
    o, p = obs > thr, pred > thr
    tp = int(np.sum(o & p)); fp = int(np.sum(~o & p))
    fn = int(np.sum(o & ~p)); tn = int(np.sum(~o & ~p))
    d = lambda a, b: (a / b) if b else float('nan')          # noqa: E731
    return {'umbral_m': thr, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'n_obs_con_nieve': tp + fn, 'n_obs_desnudo': fp + tn,
            'precision': r6(d(tp, tp + fp)),
            'exhaustividad_POD': r6(d(tp, tp + fn)),
            'especificidad': r6(d(tn, tn + fp)),
            'tasa_falsos_positivos': r6(d(fp, fp + tn)),
            'F1': r6(d(2 * tp, 2 * tp + fp + fn)),
            'CSI': r6(d(tp, tp + fp + fn)),
            'sesgo_frecuencia': r6(d(tp + fp, tp + fn))}


# ----------------------------------------------------------------------
def scan(path, pos=None, train_cov=None, clim_patch=None):
    """Una pasada por un .npz: extension, componentes del SPAEF y, si se dan
    las piezas de C, R2 y SPAEF por fecha sobre la mascara de C."""
    z = np.load(path)
    ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None

    acc = {t: {k: 0 for k in ('TP', 'FP', 'FN', 'TN')} for t in THRESHOLDS}
    comp, por_fecha = [], {}
    so = so2 = sr = 0.0; n = 0                      # solo nieve
    sof = so2f = srf = 0.0; nf = 0                  # dominio completo
    for i, tid in enumerate(ids):
        tg, pr = tgts[i].astype(np.float64), preds[i].astype(np.float64)
        val = np.isfinite(tg) if vals is None else (vals[i] > 0.5) & np.isfinite(tg)
        if val.sum():
            o, p = tg[val], pr[val]
            sof += o.sum(); so2f += (o ** 2).sum(); srf += ((o - p) ** 2).sum(); nf += o.size
            for t in THRESHOLDS:
                c = confusion(o, p, t)
                for k in ('TP', 'FP', 'FN', 'TN'):
                    acc[t][k] += c[k]
        # R2 y SPAEF "solo nieve": mascara EXACTA de training/evaluate.py
        # (l. 168 y 185): tgt > 0.01, sin la mascara de validez, porque en los
        # .npz los pixeles sin dato ya valen 0. Asi se reproduce el metrics.json.
        m = tg > SNOW
        if m.sum() >= MIN_PX:
            o, p = tg[m], pr[m]
            so += o.sum(); so2 += (o ** 2).sum(); sr += ((o - p) ** 2).sum(); n += o.size
            c = spaef_components(o, p)
            if c:
                comp.append(c)
        # ---- bloque 3: mascara de C, agregado por fecha
        if pos is not None:
            t = str(tid)
            if t in pos:
                r, cc = pos[t]
                if not train_cov[r:r + TILE, cc:cc + TILE].any():
                    cp = clim_patch(int(t[:4]), r, cc)
                    mc = (tg > SNOW) & np.isfinite(tg) & np.isfinite(pr) & np.isfinite(cp)
                    if vals is not None:
                        mc &= vals[i] > 0.5
                    if mc.sum() >= MIN_PX:
                        d = por_fecha.setdefault(t[:8], {'o': [], 'p': [], 'c': [], 'sp': [], 'spc': []})
                        o, p, q = tg[mc], pr[mc], cp[mc].astype(np.float64)
                        d['o'].append(o); d['p'].append(p); d['c'].append(q)
                        for key, x in (('sp', p), ('spc', q)):
                            v = compute_spaef(o, x)
                            if np.isfinite(v):
                                d[key].append(float(v))

    def acc_r2(s_o, s_o2, s_r, nn):
        ss = s_o2 - s_o ** 2 / nn
        return 1.0 - s_r / ss if ss > 0 else float('nan')

    out = {'npz': os.path.basename(path), 'n_teselas': len(ids),
           'solo_nieve': {'n_pixeles': n, 'R2': r6(acc_r2(so, so2, sr, n)),
                          'RMSE': r6(np.sqrt(sr / n))},
           'dominio_completo': {'n_pixeles': nf, 'R2': r6(acc_r2(sof, so2f, srf, nf)),
                                'RMSE': r6(np.sqrt(srf / nf)),
                                'fraccion_desnudo': r6(1 - n / nf) if nf else None},
           'extension': [], 'componentes_spaef': None}
    for t in THRESHOLDS:
        a = acc[t]
        d = lambda x, y: r6(x / y) if y else None                 # noqa: E731
        out['extension'].append({
            'umbral_m': t, **{k: a[k] for k in ('TP', 'FP', 'FN', 'TN')},
            'n_obs_con_nieve': a['TP'] + a['FN'], 'n_obs_desnudo': a['FP'] + a['TN'],
            'precision': d(a['TP'], a['TP'] + a['FP']),
            'exhaustividad_POD': d(a['TP'], a['TP'] + a['FN']),
            'especificidad': d(a['TN'], a['TN'] + a['FP']),
            'tasa_falsos_positivos': d(a['FP'], a['FP'] + a['TN']),
            'F1': d(2 * a['TP'], 2 * a['TP'] + a['FP'] + a['FN']),
            'CSI': d(a['TP'], a['TP'] + a['FP'] + a['FN']),
            'sesgo_frecuencia': d(a['TP'] + a['FP'], a['TP'] + a['FN'])})
    cands = find_metrics(path)
    if cands:
        mio = {'R2': acc_r2(so, so2, sr, n), 'R2_full': acc_r2(sof, so2f, srf, nf)}
        if comp:
            mio['SPAEF'] = float(np.mean([c['spaef'] for c in comp]))
            mio['SPAEF_n_tiles'] = len(comp)
        out['control'] = control_metrics(cands, mio)
    if comp:
        out['componentes_spaef'] = {
            'n_teselas': len(comp),
            **{k: {'media': r6(np.mean([c[k] for c in comp])),
                   'p10': r6(np.percentile([c[k] for c in comp], 10)),
                   'p50': r6(np.percentile([c[k] for c in comp], 50)),
                   'p90': r6(np.percentile([c[k] for c in comp], 90))}
               for k in ('alpha', 'beta', 'gamma', 'spaef')}}
        # descomposicion aditiva: media POR TESELA de cada termino al cuadrado
        tm = {k: float(np.mean([c[k] for c in comp]))
              for k in ('term_alpha2', 'term_beta2', 'term_gamma2', 'suma_terminos')}
        out['componentes_spaef']['descomposicion_terminos'] = {
            'definicion': '(1-SPAEF)^2 = (1-alpha)^2 + (1-beta)^2 + (1-gamma)^2; se promedia '
                          'cada termino por tesela, no el componente antes de elevarlo',
            **{k: r6(v) for k, v in tm.items()},
            'pct_alpha2': r6(100 * tm['term_alpha2'] / tm['suma_terminos']),
            'pct_beta2': r6(100 * tm['term_beta2'] / tm['suma_terminos']),
            'pct_gamma2': r6(100 * tm['term_gamma2'] / tm['suma_terminos']),
            'termino_dominante': max(('alpha', 'term_alpha2'), ('beta', 'term_beta2'),
                                     ('gamma', 'term_gamma2'), key=lambda x: tm[x[1]])[0],
            'comprobacion_terminos_de_las_medias': {
                'nota': 'los mismos terminos calculados con las medias de alpha, beta y gamma; '
                        'solo para ver el sesgo de Jensen, NO es la descomposicion',
                'term_alpha2': r6((np.mean([c['alpha'] for c in comp]) - 1) ** 2),
                'term_beta2': r6((np.mean([c['beta'] for c in comp]) - 1) ** 2),
                'term_gamma2': r6((np.mean([c['gamma'] for c in comp]) - 1) ** 2)}}
    fechas = {}
    for d, v in sorted(por_fecha.items()):
        o = np.concatenate(v['o']); p = np.concatenate(v['p']); q = np.concatenate(v['c'])
        sd_o = float(o.std())
        fechas[d] = {'n_pixeles': int(o.size), 'n_teselas': len(v['o']),
                     'R2_modelo': r6(r2(o, p)), 'R2_climatologia': r6(r2(o, q)),
                     'SPAEF_modelo': r6(np.mean(v['sp'])) if v['sp'] else None,
                     'SPAEF_climatologia': r6(np.mean(v['spc'])) if v['spc'] else None,
                     # para distinguir suavizado de acierto: si el R2 se salva
                     # porque la prediccion es casi plana, el cociente de
                     # desviaciones tipicas cae muy por debajo de 1
                     'HS_media_observada_m': r6(float(o.mean())),
                     'sd_observada_m': r6(sd_o),
                     'sd_modelo_m': r6(float(p.std())),
                     'sd_climatologia_m': r6(float(q.std())),
                     'cociente_sd_modelo': r6(float(p.std()) / sd_o) if sd_o > 0 else None,
                     'cociente_sd_climatologia': r6(float(q.std()) / sd_o) if sd_o > 0 else None,
                     'sesgo_modelo_m': r6(float((p - o).mean())),
                     'sesgo_climatologia_m': r6(float((q - o).mean())),
                     'rho_modelo': r6(float(np.corrcoef(o, p)[0, 1])) if sd_o > 0 else None,
                     'rho_climatologia': r6(float(np.corrcoef(o, q)[0, 1])) if sd_o > 0 else None}
    if fechas:
        out['por_fecha_mascara_C'] = fechas
    del z, ids, preds, tgts, vals
    return out


def pct(v, q):
    return r6(np.percentile([x for x in v if x is not None and np.isfinite(x)], q))


def main():
    ap = argparse.ArgumentParser(description='Analisis adicionales para la revision Q1')
    ap.add_argument('--spatial-net', nargs=3, required=True)
    ap.add_argument('--spatial-rf', nargs=3, required=True)
    ap.add_argument('--temporal-net', nargs='+', default=[])
    ap.add_argument('--temporal-rf', nargs='*', default=[])
    ap.add_argument('--seed7-dir', default=None)
    ap.add_argument('--seeds-dir', default=None)
    ap.add_argument('--con-loyo', action='store_true', help='anyade los 12 .npz de LOYO (lento)')
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--out', default='analysis/revision_q1.json')
    args = ap.parse_args()
    t0 = time.time()

    # ---- piezas de la mascara de C (como scripts/results_5_2.py)
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in
           geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    H, W = int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE
    geo['year'] = geo['date'].str[:4].astype(int)
    train_cov = np.zeros((H, W), bool)
    for r, c in geo.loc[geo['exp_spatial_split'] == 'train',
                        ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True
    print(f'Climatologia por anyo  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    ysum, ycnt, tsum, tcnt = build_year_climatology(geo, args.masks, H, W)

    def clim_patch(year, r, c):
        s = tsum[r:r + TILE, c:c + TILE] - ysum[year][r:r + TILE, c:c + TILE]
        nn = tcnt[r:r + TILE, c:c + TILE] - ycnt[year][r:r + TILE, c:c + TILE]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(nn > 0, s / np.maximum(nn, 1), np.nan)

    jobs = []
    for lab, paths, con_c in (('espacial_red', args.spatial_net, True),
                              ('espacial_rf', args.spatial_rf, True),
                              ('temporal_red', args.temporal_net, False),
                              ('temporal_rf', args.temporal_rf, False)):
        for p in paths:
            jobs.append((lab, p, con_c))
    if args.con_loyo and args.seed7_dir and args.seeds_dir:
        for f in FOLDS:
            jobs.append((f'loyo{f}_red', os.path.join(
                args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{f}',
                f'resunetpp_v3_loyo{f}_predictions.npz'), False))
            for s in ('42', '123'):
                jobs.append((f'loyo{f}_red', os.path.join(
                    args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{f}_s{s}',
                    f'resunetpp_v3_loyo{f}_s{s}_predictions.npz'), False))

    res = {}
    for lab, p, con_c in jobs:
        if not os.path.exists(p):
            print(f'  [aviso] falta {p}'); continue
        print(f'{lab}: {os.path.basename(p)}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        res.setdefault(lab, []).append(
            scan(p, pos, train_cov, clim_patch) if con_c else scan(p))

    out = {'descripcion': 'Analisis adicionales para la revision Q1 (14-oct-2026): dominio '
                          'completo y extension de nieve, componentes del SPAEF y dispersion '
                          'por fecha en el split espacial. No cambia ningun resultado del '
                          'articulo. 6 decimales.',
           'definiciones': {
               'solo_nieve': 'pixeles validos con HS observada > 0.01 m (mascara de las metricas '
                             'del articulo)',
               'dominio_completo': 'todos los pixeles con dato valido, incluido el suelo desnudo',
               'extension': 'observado y predicho por encima del umbral; POD = TP/(TP+FN), '
                            'especificidad = TN/(TN+FP), CSI = TP/(TP+FP+FN), sesgo de '
                            'frecuencia = predichos/observados con nieve',
               'componentes_spaef': 'alpha correlacion, beta cociente de CV, gamma interseccion '
                                    'de histogramas de z-scores (notacion del articulo, Koch 2018); '
                                    'por tesela, sobre sus pixeles con nieve',
               'por_fecha': 'mascara de C (702 teselas sin fuga, pixeles con nieve, dato valido y '
                            'climatologia); R2 acumulado de la fecha y SPAEF medio por tesela'},
           '1b_2_por_ejecucion': res}

    # ---- 1a: dominio completo desde los metrics.json (todas las semillas)
    # s7 del split temporal: se adopta el entrenamiento que SI exporto
    # predicciones (config ..._preds.yaml, agosto), que es el del .npz y el de
    # la descomposicion de analysis/e1_results.json. El de julio no guardo
    # predicciones y sus pesos no se conservaron, asi que su descomposicion no
    # es reproducible. Asi la tabla, el .npz y la descomposicion salen del
    # mismo entrenamiento. El de julio se registra en 'nota_s7'.
    S7_TEMP = 'results/norm_v3/lambda/resunetpp_v3_sp00_s7_preds/resunetpp_v3_sp00_s7_metrics.json'
    S7_TEMP_ALT = 'results/norm_v3/lambda/resunetpp_v3_sp00_s7/resunetpp_v3_sp00_s7_metrics.json'
    MET = {'temporal_red': [S7_TEMP] + [f'results/norm_v3/lambda/resunetpp_v3_sp00_s{s}/'
                                        f'resunetpp_v3_sp00_s{s}_metrics.json' for s in SEEDS[1:]],
           'loyo2021_red': ['results/norm_v3/loyo/resunetpp_v3_loyo2021/resunetpp_v3_loyo2021_metrics.json'],
           'loyo2022_red': ['results/norm_v3/loyo/resunetpp_v3_loyo2022/resunetpp_v3_loyo2022_metrics.json'],
           'loyo2023_red': ['results/norm_v3/loyo/resunetpp_v3_loyo2023/resunetpp_v3_loyo2023_metrics.json'],
           'loyo2025_red': ['results/norm_v3/loyo/resunetpp_v3_loyo2025/resunetpp_v3_loyo2025_metrics.json']}
    for f in FOLDS:
        MET[f'loyo{f}_red'] += [f'results/norm_v3/loyo_seeds/resunetpp_v3_loyo{f}_s{s}/'
                                f'resunetpp_v3_loyo{f}_s{s}_metrics.json' for s in ('42', '123')]
    dom = {}
    for lab, files in MET.items():
        ms = [json.load(open(p, encoding='utf-8')) for p in files if os.path.exists(p)]
        if not ms:
            continue
        dom[lab] = {'n_semillas': len(ms), 'fuente': files[0].replace('_s7', '_s<semilla>'),
                    'ficheros': files}
        for k in ('R2', 'MAE', 'RMSE', 'Bias', 'R2_full', 'MAE_full', 'RMSE_full', 'Bias_full',
                  'snow_precision', 'snow_recall', 'false_pos_rate'):
            v = [m[k] for m in ms if k in m]
            if v:
                dom[lab][k] = mmm(v, 4)
    # trazabilidad del cambio de run en s7 temporal: se guardan las dos cifras
    if 'temporal_red' in dom and os.path.exists(S7_TEMP_ALT):
        alt = json.load(open(S7_TEMP_ALT, encoding='utf-8'))
        dom['temporal_red']['nota_s7'] = {
            'adoptado': S7_TEMP,
            'timestamp_adoptado': json.load(open(S7_TEMP, encoding='utf-8')).get('timestamp'),
            'descartado': S7_TEMP_ALT,
            'timestamp_descartado': alt.get('timestamp'),
            'valores_descartados': {k: alt[k] for k in
                                    ('R2', 'MAE', 'RMSE', 'Bias', 'R2_full', 'MAE_full')
                                    if k in alt},
            'motivo': ('dos entrenamientos de la misma configuracion y semilla (config '
                       'identica, best_val_loss 0.508555 frente a 0.492361). El adoptado '
                       'es el que exporto predicciones y por tanto el del .npz y de la '
                       'descomposicion de analysis/e1_results.json; del descartado no se '
                       'conservaron los pesos, asi que su descomposicion no es reproducible.')}
    out['1a_dominio_completo_metrics_json'] = dom

    # ---- 3: dispersion por fecha en C
    if 'espacial_red' in res and 'espacial_rf' in res:
        fechas = sorted(res['espacial_red'][0].get('por_fecha_mascara_C', {}))
        tabla = []
        for d in fechas:
            rn = [x['por_fecha_mascara_C'][d]['R2_modelo'] for x in res['espacial_red']]
            rr = [x['por_fecha_mascara_C'][d]['R2_modelo'] for x in res['espacial_rf']]
            sn = [x['por_fecha_mascara_C'][d]['SPAEF_modelo'] for x in res['espacial_red']]
            sr = [x['por_fecha_mascara_C'][d]['SPAEF_modelo'] for x in res['espacial_rf']]
            cl = res['espacial_red'][0]['por_fecha_mascara_C'][d]
            cq = [x['por_fecha_mascara_C'][d]['cociente_sd_modelo'] for x in res['espacial_red']]
            cqr = [x['por_fecha_mascara_C'][d]['cociente_sd_modelo'] for x in res['espacial_rf']]
            ro = [x['por_fecha_mascara_C'][d]['rho_modelo'] for x in res['espacial_red']]
            tabla.append({'fecha': d, 'n_teselas': cl['n_teselas'], 'n_pixeles': cl['n_pixeles'],
                          'R2_red': mmm(rn), 'R2_rf': mmm(rr),
                          'R2_climatologia': cl['R2_climatologia'],
                          'SPAEF_red': mmm(sn), 'SPAEF_rf': mmm(sr),
                          'SPAEF_climatologia': cl['SPAEF_climatologia'],
                          # suavizado frente a acierto
                          'HS_media_observada_m': cl['HS_media_observada_m'],
                          'sd_observada_m': cl['sd_observada_m'],
                          'sd_climatologia_m': cl['sd_climatologia_m'],
                          'cociente_sd_red': mmm(cq), 'cociente_sd_rf': mmm(cqr),
                          'cociente_sd_climatologia': cl['cociente_sd_climatologia'],
                          'rho_red': mmm(ro), 'rho_climatologia': cl['rho_climatologia'],
                          'sesgo_climatologia_m': cl['sesgo_climatologia_m']})
        def col(key, sub='media'):
            return [t[key][sub] if isinstance(t[key], dict) else t[key] for t in tabla]
        resumen = {}
        for k in ('R2_red', 'R2_rf', 'R2_climatologia', 'SPAEF_red', 'SPAEF_rf', 'SPAEF_climatologia'):
            v = col(k)
            resumen[k] = {'n_fechas': len(v), 'p10': pct(v, 10), 'mediana': pct(v, 50),
                          'p90': pct(v, 90), 'min': r6(min(v)), 'max': r6(max(v))}
        resumen['fechas_red_gana_a_rf_R2'] = int(sum(a > b for a, b in zip(col('R2_red'), col('R2_rf'))))
        resumen['fechas_red_gana_a_rf_SPAEF'] = int(sum(
            a > b for a, b in zip(col('SPAEF_red'), col('SPAEF_rf'))))
        resumen['fechas_red_gana_a_climatologia_R2'] = int(sum(
            a > b for a, b in zip(col('R2_red'), col('R2_climatologia'))))
        # fechas en las que la climatologia tiene R2 < 0: suavizado o acierto
        mal = [t for t in tabla if t['R2_climatologia'] is not None and t['R2_climatologia'] < 0]
        out['3_dispersion_por_fecha_C'] = {
            'aviso': 'las fechas son unidades mas independientes que las semillas, pero comparten '
                     'el mismo split espacial: no sustituyen a repetir el split con otras bandas',
            'resumen_entre_fechas': resumen,
            'fechas_climatologia_R2_negativo': {
                'pregunta': 'si el R2 de la red se salva porque su prediccion es casi plana, el '
                            'cociente de desviaciones tipicas cae muy por debajo de 1 y rho es '
                            'bajo; si acierta, el cociente se mantiene cerca de 1 con rho alto',
                'n_fechas': len(mal),
                'fechas': [{k: t[k] for k in
                            ('fecha', 'n_pixeles', 'HS_media_observada_m', 'sd_observada_m',
                             'R2_red', 'R2_climatologia', 'cociente_sd_red',
                             'cociente_sd_climatologia', 'rho_red', 'rho_climatologia',
                             'sesgo_climatologia_m')} for t in mal]},
            'por_fecha': tabla}

    # ---------------------------------------------------------- impresion
    print('=' * 72)
    print('1a. Dominio completo frente a solo nieve (metrics.json)')
    print(f"  {'ejecucion':<16}{'R2 nieve':>10}{'R2 completo':>13}{'MAE nieve':>11}{'MAE completo':>14}")
    for lab, v in dom.items():
        if 'R2_full' in v:
            print(f"  {lab:<16}{v['R2']['media']:>10.3f}{v['R2_full']['media']:>13.3f}"
                  f"{v['MAE']['media']:>11.3f}{v['MAE_full']['media']:>14.3f}")
    print('\n1b. Extension de nieve (umbral del pipeline y sensibilidad)')
    for lab, xs in res.items():
        for t in THRESHOLDS:
            e = [x['extension'][THRESHOLDS.index(t)] for x in xs]
            print(f"  {lab:<16}{t:>6.2f} m  POD {np.mean([x['exhaustividad_POD'] for x in e]):.3f}"
                  f"  especificidad {np.mean([x['especificidad'] for x in e]):.3f}"
                  f"  CSI {np.mean([x['CSI'] for x in e]):.3f}"
                  f"  sesgo frec. {np.mean([x['sesgo_frecuencia'] for x in e]):.3f}"
                  f"  (desnudo {100 * xs[0]['dominio_completo']['fraccion_desnudo']:.1f} %)")
    print('\n2. Componentes del SPAEF (media por tesela)')
    print(f"  {'ejecucion':<16}{'alpha':>8}{'beta':>8}{'gamma':>8}{'SPAEF':>8}")
    for lab, xs in res.items():
        c = [x['componentes_spaef'] for x in xs if x['componentes_spaef']]
        if c:
            print(f"  {lab:<16}" + ''.join(f"{np.mean([y[k]['media'] for y in c]):>8.3f}"
                                           for k in ('alpha', 'beta', 'gamma', 'spaef')))
    print('\n2b. Descomposicion de (1-SPAEF)^2 (media por tesela de cada termino)')
    print(f"  {'ejecucion':<16}{'(1-a)^2':>10}{'(1-b)^2':>10}{'(1-g)^2':>10}{'suma':>10}"
          f"{'% alpha':>9}{'% beta':>8}{'% gamma':>9}")
    for lab, xs in res.items():
        t = [x['componentes_spaef']['descomposicion_terminos']
             for x in xs if x['componentes_spaef']]
        if t:
            g = lambda k: np.mean([y[k] for y in t])               # noqa: E731
            print(f"  {lab:<16}{g('term_alpha2'):>10.4f}{g('term_beta2'):>10.4f}"
                  f"{g('term_gamma2'):>10.4f}{g('suma_terminos'):>10.4f}"
                  f"{g('pct_alpha2'):>9.1f}{g('pct_beta2'):>8.1f}{g('pct_gamma2'):>9.1f}")
    if '3_dispersion_por_fecha_C' in out:
        m = out['3_dispersion_por_fecha_C']['fechas_climatologia_R2_negativo']
        print(f"\n3b. Fechas con R2 de la climatologia < 0 ({m['n_fechas']} de "
              f"{len(out['3_dispersion_por_fecha_C']['por_fecha'])})")
        print(f"  {'fecha':<10}{'sd obs':>8}{'R2 red':>9}{'R2 clim':>10}"
              f"{'sd red/obs':>12}{'sd clim/obs':>13}{'rho red':>9}{'rho clim':>10}")
        for t in m['fechas']:
            print(f"  {t['fecha']:<10}{t['sd_observada_m']:>8.3f}{t['R2_red']['media']:>9.3f}"
                  f"{t['R2_climatologia']:>10.3f}{t['cociente_sd_red']['media']:>12.3f}"
                  f"{t['cociente_sd_climatologia']:>13.3f}{t['rho_red']['media']:>9.3f}"
                  f"{t['rho_climatologia']:>10.3f}")
    if '3_dispersion_por_fecha_C' in out:
        s = out['3_dispersion_por_fecha_C']['resumen_entre_fechas']
        print('\n3. Dispersion entre fechas en C (mediana [p10-p90] de las 27 fechas)')
        for k in ('R2_red', 'R2_rf', 'R2_climatologia', 'SPAEF_red', 'SPAEF_rf', 'SPAEF_climatologia'):
            print(f"  {k:<20}{s[k]['mediana']:>8.3f} [{s[k]['p10']:.3f} - {s[k]['p90']:.3f}]"
                  f"   min {s[k]['min']:.3f}  max {s[k]['max']:.3f}")
        print(f"  la red gana al RF en R2 en {s['fechas_red_gana_a_rf_R2']} de {s['R2_red']['n_fechas']} "
              f"fechas; en SPAEF en {s['fechas_red_gana_a_rf_SPAEF']}; "
              f"a la climatologia en R2 en {s['fechas_red_gana_a_climatologia_R2']}")

    print('\n0. Control frente a los metrics.json')
    for lab, xs in res.items():
        for x in xs:
            c = x.get('control')
            if not c:
                print(f"  {lab:<16}{x['npz']:<48} sin metrics.json localizable")
                continue
            m = c.get('max_diferencia_frente_a_metrics_json')
            print(f"  {lab:<16}{x['npz']:<48} {c['n_metrics_json']} metrics.json  "
                  f"max dif {m if m is None else f'{m:.6f}'}")
            if 'aviso' in c:
                print(f"      AVISO: varios metrics.json discrepantes; el .npz coincide con "
                      f"{c.get('coincide_con')}")
                for k, vs in c['metrics_json_multiples'].items():
                    print(f"        {k}: " + ', '.join(f'{v} ({p})' for p, v in vs.items()))
    if 'temporal_red' in dom and 'nota_s7' in dom['temporal_red']:
        n = dom['temporal_red']['nota_s7']
        print(f"  s7 temporal: adoptado {n['timestamp_adoptado']} (R2 "
              f"{dom['temporal_red']['R2']['por_semilla'][0]}), descartado "
              f"{n['timestamp_descartado']} (R2 {n['valores_descartados'].get('R2')})")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nGuardado en: {args.out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
