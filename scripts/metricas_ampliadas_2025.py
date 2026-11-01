r"""
Métricas ampliadas en el test de 2025 (enmienda 1 del protocolo; sin GPU)
=========================================================================

Calcula sobre las predicciones ya guardadas las métricas que añadió la
enmienda 1 de analysis/protocolo_test_2026.md (apartados 8.1-8.3), para los
cinco modelos del protocolo, con la mejor época y con la última:

    combinacion             climatología desplazada al nivel de la red B
                            (misma función que scripts/clim_v1_hybrid.py)
    Bclim_con_climatologia  red con climatología
    Bclim_sin_climatologia  la misma red con la climatología apagada
    B_paso1                 red sin climatología (paso 1, brazo B)
    climatologia_sola       sin red (una sola ejecución, sin semillas)

Métricas
--------
A 1 m, sobre los píxeles de la evaluación (HS observada > 0,01 m y dato
válido); la predicción se recorta a 0, como en training/evaluate.py:
    R2 (acumulado), SPAEF (por tesela, media sin ponderar), RMSE y MAE (m),
    correlación de patrón por tesela (Pearson, en las mismas teselas que el
    SPAEF; media y mediana).
A 5 m: dentro de cada tesela, bloques de 5 x 5 píxeles (se descartan la
última fila y la última columna, 256 = 51*5 + 1). Un bloque cuenta si tiene
al menos 13 de 25 píxeles válidos; su valor es la media de esos píxeles,
observada y predicha. Sobre los bloques con nieve (media observada > 0,01 m)
se calculan las mismas cinco métricas, igual que a 1 m.
Error de nivel: por fecha, media predicha - media observada sobre los
píxeles de la evaluación de esa fecha (m); y la media de su valor absoluto.
Dos lecturas (apartado 8.3): tal cual (opción aplicable) y con el nivel
corregido por fecha (diagnóstico; misma corrección que
scripts/clim_v1_level_corrected.py, sin recorte a 0).

Controles (si alguno falla, se detiene sin escribir nada)
---------------------------------------------------------
1. Tal cual a 1 m: R2, SPAEF, RMSE y MAE iguales a los de cada metrics.json
   (tolerancia 1e-3).
2. Combinación: R2 y SPAEF iguales a analysis/clim_v1_hibrido.json.
3. Nivel corregido: R2 y SPAEF iguales a analysis/clim_v1_nivel_corregido.json.
4. Ninguna tesela posterior al 30-09-2025 (la temporada 2026 está reservada).

Uso (desde la raíz de repo-moises, con geo_env)
-----------------------------------------------
    python scripts/metricas_ampliadas_2025.py
Opciones: --clim-dir (results/clim_v1), --b-dir (results/sat_v2),
          --out (analysis/metricas_ampliadas_2025.json)
"""

import argparse
import json
import os
import numpy as np

# Funciones copiadas TAL CUAL de utils/metrics.py (compute_spaef),
# scripts/clim_v1_level_corrected.py (date_of, load, mmm) y
# scripts/clim_v1_hybrid.py (hybrid). Se copian para no importar
# utils/metrics.py, que carga scikit-learn y scipy (en el portátil, una
# directiva de Control de aplicaciones de Windows bloquea una DLL de scipy).
# Los controles comprueban que el resultado es idéntico al de la evaluación.


def compute_spaef(obs, sim, n_bins=100):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.maximum(sim, 0.0)
    if len(obs) < 10:
        return float('nan')
    rho = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(rho):
        return float('nan')
    mean_obs = float(np.mean(obs))
    mean_sim = float(np.mean(sim))
    if mean_obs == 0.0 or mean_sim == 0.0:
        return float('nan')
    cv_obs = float(np.std(obs)) / mean_obs
    cv_sim = float(np.std(sim)) / mean_sim
    if cv_obs == 0.0:
        return float('nan')
    alpha = cv_sim / cv_obs
    std_obs = float(np.std(obs))
    std_sim = float(np.std(sim))
    if std_obs == 0.0 or std_sim == 0.0:
        return float('nan')
    oz = (obs - mean_obs) / std_obs
    sz = (sim - mean_sim) / std_sim
    lo = min(float(oz.min()), float(sz.min()))
    hi = max(float(oz.max()), float(sz.max()))
    if hi <= lo:
        return float('nan')
    bins = np.linspace(lo, hi, n_bins + 1)
    h_obs, _ = np.histogram(oz, bins=bins)
    h_sim, _ = np.histogram(sz, bins=bins)
    h_obs = h_obs / (h_obs.sum() + 1e-10)
    h_sim = h_sim / (h_sim.sum() + 1e-10)
    beta = float(np.sum(np.minimum(h_obs, h_sim)))
    return float(1.0 - np.sqrt((rho - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))


def date_of(t):
    return str(t).split('_')[0]


def load(npz):
    z = np.load(npz)
    p = np.squeeze(z['preds']).astype(np.float64)
    o = np.squeeze(z['targets']).astype(np.float64)
    v = np.squeeze(z['valids']) > 0.5 if 'valids' in z.files else np.ones_like(o, bool)
    return [str(t) for t in z['tile_ids']], p, o, v


def mmm(v):
    v = np.asarray(v, float)
    return {'media': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4), 'por_semilla': [round(float(x), 4) for x in v]}


def hybrid(clim_ids, clim_p, net_ids, net_p, valid):
    idx = {t: i for i, t in enumerate(net_ids)}
    order = np.array([idx[t] for t in clim_ids])
    net_p = net_p[order]
    dates = np.array([date_of(t) for t in clim_ids])
    out = np.empty_like(clim_p)
    for d in np.unique(dates):
        k = dates == d
        v = valid[k]
        shift = float(net_p[k][v].mean() - clim_p[k][v].mean())
        out[k] = np.maximum(clim_p[k] + shift, 0.0)
    return out, net_p


SNOW = 0.01
TOL = 1e-3
SEEDS = (7, 42, 123)
BLOCK = 5
MIN_VALID = 13
LIMITE = '20250930'
EPOCAS = (('mejor_epoca', ''), ('ultima_epoca', '_last'))
MODELOS = ('combinacion', 'Bclim_con_climatologia', 'Bclim_sin_climatologia', 'B_paso1')


# --------------------------------------------------------------------------
# métricas sobre una lista de (obs, pred) por tesela
# --------------------------------------------------------------------------
def pearson(o, p):
    if o.size < 2 or o.std() == 0 or p.std() == 0:
        return np.nan
    return float(np.corrcoef(o, p)[0, 1])


def conjunto(pares):
    """R2 acumulado, RMSE, MAE, SPAEF y correlación por tesela."""
    so = so2 = sr = sa = 0.0
    n = 0
    sp, cr = [], []
    for o, p in pares:
        if o.size == 0:
            continue
        so += o.sum(); so2 += (o ** 2).sum()
        sr += ((o - p) ** 2).sum(); sa += np.abs(o - p).sum(); n += o.size
        s = compute_spaef(o, p) if o.size >= 10 else np.nan
        if np.isfinite(s):
            sp.append(float(s))
            cr.append(pearson(o, np.maximum(p, 0.0)))
    ss = so2 - so ** 2 / n
    cr = np.array(cr, float)
    return {'R2': 1.0 - sr / ss, 'RMSE': float(np.sqrt(sr / n)), 'MAE': sa / n,
            'SPAEF': float(np.mean(sp)), 'corr_media': float(np.nanmean(cr)),
            'corr_mediana': float(np.nanmedian(cr)), 'n_teselas': len(sp), 'n': n}


def bloques(o, p, v):
    """Medias por bloque de 5 x 5 con al menos MIN_VALID píxeles válidos."""
    k = (o.shape[0] // BLOCK) * BLOCK
    o, p, v = o[:k, :k], p[:k, :k], v[:k, :k]
    sh = (k // BLOCK, BLOCK, k // BLOCK, BLOCK)
    vf = v.reshape(sh).astype(float)
    cnt = vf.sum(axis=(1, 3))
    so = np.where(v, o, 0.0).reshape(sh).sum(axis=(1, 3))
    sp = np.where(v, p, 0.0).reshape(sh).sum(axis=(1, 3))
    ok = cnt >= MIN_VALID
    with np.errstate(invalid='ignore', divide='ignore'):
        bo, bp = so / cnt, sp / cnt
    sel = ok & (bo > SNOW)
    return bo[sel], bp[sel]


def evaluar(ids, p, o, v):
    """Todas las métricas de un modelo (una semilla)."""
    if any(date_of(t) > LIMITE for t in ids):
        raise SystemExit('STOP: hay teselas posteriores al 30-09-2025 (temporada reservada).')
    dates = np.array([date_of(t) for t in ids])
    valid = v & np.isfinite(o)
    m = valid & (o > SNOW)
    pc = np.maximum(p, 0.0)

    err, corr = {}, {}
    for d in np.unique(dates):
        k = dates == d
        e = float(pc[k][m[k]].mean() - o[k][m[k]].mean())
        err[d] = e
        corr[d] = float(o[k][m[k]].mean() - p[k][m[k]].mean())   # como clim_v1_level_corrected

    tal = conjunto([(o[i][m[i]], pc[i][m[i]]) for i in range(len(ids))])
    niv = conjunto([(o[i][m[i]], p[i][m[i]] + corr[dates[i]]) for i in range(len(ids))])
    b5 = []
    for i in range(len(ids)):
        oi = np.where(valid[i], o[i], 0.0)
        b5.append(bloques(oi, pc[i], valid[i]))
    a5 = conjunto(b5)
    return {'tal_cual': tal, 'nivel_corregido': niv, 'a_5m': a5,
            'error_nivel_por_fecha': {d: round(e, 4) for d, e in sorted(err.items())},
            'error_nivel_media_abs': float(np.mean(np.abs(list(err.values()))))}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Métricas ampliadas en el test de 2025')
    ap.add_argument('--clim-dir', default='results/clim_v1')
    ap.add_argument('--b-dir', default='results/sat_v2')
    ap.add_argument('--hibrido', default='analysis/clim_v1_hibrido.json')
    ap.add_argument('--nivel', default='analysis/clim_v1_nivel_corregido.json')
    ap.add_argument('--out', default='analysis/metricas_ampliadas_2025.json')
    a = ap.parse_args()

    ref_h = json.load(open(a.hibrido))
    ref_n = json.load(open(a.nivel))
    fallos = []

    def check(nombre, valor, ref):
        if abs(valor - ref) > TOL:
            fallos.append(f'{nombre}: {valor:.4f} frente a {ref:.4f}')

    def path(modelo, s, e):
        if modelo == 'B_paso1':
            n = f'resunetpp_sat2_B_s{s}'
            return os.path.join(a.b_dir, n, f'{n}{e}')
        n = f'resunetpp_clim1_Bclim_s{s}'
        suf = e + ('_sinclim' if modelo == 'Bclim_sin_climatologia' else '')
        return os.path.join(a.clim_dir, n, f'{n}{suf}')

    cs = os.path.join(a.clim_dir, 'climatologia_sola', 'climatologia_sola')
    c_ids, c_p, c_o, c_v = load(cs + '_predictions.npz')
    valid_c = c_v & np.isfinite(c_o)

    res = {'descripcion': ('Métricas de la enmienda 1 del protocolo (apartados 8.1-8.3) en el test '
                           '2025. Media [mín-máx] y valor por semilla (7, 42, 123).'),
           'definiciones': {'umbral_nieve_m': SNOW, 'bloque_px': BLOCK, 'minimo_validos_bloque': MIN_VALID,
                            'recorte_a_0': 'tal cual y 5 m; no en nivel corregido'}}

    # climatología sola
    r = evaluar(c_ids, c_p, c_o, c_v)
    ref = json.load(open(cs + '_metrics.json'))
    for k in ('R2', 'SPAEF', 'RMSE', 'MAE'):
        check(f'climatologia_sola {k}', r['tal_cual'][k], ref[k])
    for k in ('R2', 'SPAEF'):
        check(f'climatologia_sola nivel {k}', r['nivel_corregido'][k], ref_n['climatologia_sola']['nivel_corregido'][k])
    res['climatologia_sola'] = r
    print(f"climatologia_sola  R2 {r['tal_cual']['R2']:.3f}", flush=True)

    claves = ('R2', 'SPAEF', 'RMSE', 'MAE', 'corr_media', 'corr_mediana')
    for epoca, e in EPOCAS:
        res[epoca] = {}
        for modelo in MODELOS:
            filas = {}
            for s in SEEDS:
                fuente = 'B_paso1' if modelo == 'combinacion' else modelo
                base = path(fuente, s, e)
                if not os.path.exists(base + '_predictions.npz'):
                    fallos.append(f'falta {base}_predictions.npz')
                    continue
                ids, p, o, v = load(base + '_predictions.npz')
                if modelo == 'combinacion':
                    h, _ = hybrid(c_ids, c_p, ids, p, valid_c)
                    r = evaluar(c_ids, h, c_o, c_v)
                    i = ref_h[epoca]['B_paso1']['semillas'].index(s)
                    for k in ('R2', 'SPAEF'):
                        check(f'{epoca} combinacion s{s} {k}', r['tal_cual'][k],
                              ref_h[epoca]['B_paso1']['clim_mas_nivel_de_la_red'][k]['por_semilla'][i])
                else:
                    r = evaluar(ids, p, o, v)
                    ref = json.load(open(base + '_metrics.json'))
                    for k in ('R2', 'SPAEF', 'RMSE', 'MAE'):
                        check(f'{epoca} {modelo} s{s} {k}', r['tal_cual'][k], ref[k])
                    i = ref_n[epoca][modelo]['semillas'].index(s)
                    for k in ('R2', 'SPAEF'):
                        check(f'{epoca} {modelo} s{s} nivel {k}', r['nivel_corregido'][k],
                              ref_n[epoca][modelo]['nivel_corregido'][k]['por_semilla'][i])
                filas[s] = r
                print(f"{epoca:12s} {modelo:24s} s{s:<3d} R2 {r['tal_cual']['R2']:.3f} "
                      f"corr {r['tal_cual']['corr_media']:.3f} | 5 m R2 {r['a_5m']['R2']:.3f}", flush=True)
            if not filas:
                continue
            out = {'semillas': sorted(filas)}
            for lect in ('tal_cual', 'nivel_corregido', 'a_5m'):
                out[lect] = {k: mmm([filas[s][lect][k] for s in filas]) for k in claves}
            fechas = sorted(next(iter(filas.values()))['error_nivel_por_fecha'])
            out['error_nivel_por_fecha'] = {d: mmm([filas[s]['error_nivel_por_fecha'][d] for s in filas])
                                            for d in fechas}
            out['error_nivel_media_abs'] = mmm([filas[s]['error_nivel_media_abs'] for s in filas])
            res[epoca][modelo] = out

    res['controles'] = {'tolerancia': TOL, 'fallos': fallos, 'superados': not fallos}
    if fallos:
        print('\nControles NO superados:')
        for f in fallos:
            print('  ' + f)
        raise SystemExit('STOP: no se escribe nada.')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)

    def fila(nombre, d, lect, extra=''):
        g = lambda k: d[lect][k]['media'] if isinstance(d[lect][k], dict) else d[lect][k]
        print(f"  {nombre:26s} {g('R2'):7.3f} {g('SPAEF'):7.3f} {g('corr_media'):7.3f} "
              f"{g('RMSE'):7.3f} {g('MAE'):7.3f}{extra}")

    cab = f"  {'':26s} {'R2':>7s} {'SPAEF':>7s} {'corr':>7s} {'RMSE':>7s} {'MAE':>7s}"
    nombres = {'combinacion': 'Combinación', 'Bclim_con_climatologia': 'Red con climatología',
               'Bclim_sin_climatologia': 'Red con clim. apagada', 'B_paso1': 'Red sin climatología (B)'}
    cl = res['climatologia_sola']
    for epoca, _ in EPOCAS:
        print(f'\n===== {epoca} (media de 3 semillas) =====')
        for lect, titulo in (('tal_cual', 'Paso 1. Opciones aplicables, tal cual, 1 m'),
                             ('a_5m', 'Tal cual, a 5 m'),
                             ('nivel_corregido', 'Paso 2. Diagnóstico: nivel corregido, 1 m')):
            print(f'\n{titulo}\n{cab}')
            for modelo in MODELOS:
                if modelo in res[epoca]:
                    d = res[epoca][modelo]
                    extra = f"   error de nivel {d['error_nivel_media_abs']['media']:.3f} m" if lect == 'tal_cual' else ''
                    fila(nombres[modelo], d, lect, extra)
            extra = f"   error de nivel {cl['error_nivel_media_abs']:.3f} m" if lect == 'tal_cual' else ''
            fila('Climatología sola', cl, lect, extra)
    print('\nControles: SUPERADOS')
    print(f'Resultados: {a.out}')


if __name__ == '__main__':
    main()
