r"""
Paso 2: lectura tal cual y con el nivel corregido (sin GPU)
===========================================================

Segunda lectura fijada ANTES de ver los resultados del paso 2: ademas de las
metricas tal cual, se mide solo el DIBUJO quitando el error de nivel de cada
fecha. Es la misma correccion del articulo (nivel perfecto aditivo): a cada
prediccion de una fecha se le suma (media observada - media predicha) de esa
fecha, sobre los pixeles con nieve de todas sus teselas. Asi se responde a la
pregunta de Hernan: ¿la red con climatologia dibuja mejor que la
climatologia sola?

Modelos (mismas 220 teselas de test, mismos pixeles):
    Bclim con climatologia     results/clim_v1/resunetpp_clim1_Bclim_s*/..._predictions.npz
    Bclim sin climatologia     ..._sinclim_predictions.npz
    B (paso 1)                 <b-dir>/resunetpp_sat2_B_s*/..._predictions.npz
    climatologia sola          results/clim_v1/climatologia_sola/climatologia_sola_predictions.npz
para la mejor epoca y la ultima (_last).

Pixeles: con nieve (HS > 0,01 m) y dato valido, los del pipeline. R2
acumulado sobre todos; SPAEF por tesela con utils.metrics.compute_spaef y
media sin ponderar, como la evaluacion.

Control: la lectura tal cual debe reproducir el R2 y el SPAEF de cada
metrics.json (tolerancia 1e-3). Si no, el script se detiene.

Uso (desde la raiz del repo)
----------------------------
    python scripts/clim_v1_level_corrected.py --b-dir results/sat_v2_instancia
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.metrics import compute_spaef  # noqa: E402

SNOW = 0.01
TOL = 1e-3
SEEDS = (7, 42, 123)


def date_of(t):
    return str(t).split('_')[0]


def load(npz):
    z = np.load(npz)
    p = np.squeeze(z['preds']).astype(np.float64)
    o = np.squeeze(z['targets']).astype(np.float64)
    v = np.squeeze(z['valids']) > 0.5 if 'valids' in z.files else np.ones_like(o, bool)
    return [str(t) for t in z['tile_ids']], p, o, v


def metrics(ids, p, o, v):
    dates = np.array([date_of(t) for t in ids])
    m = v & np.isfinite(o) & (o > SNOW)
    corr = {}
    for d in np.unique(dates):
        k = dates == d
        mk = m[k]
        corr[d] = float(o[k][mk].mean() - p[k][mk].mean())
    out = {}
    for lect in ('tal_cual', 'nivel_corregido'):
        so = so2 = sr = 0.0
        n = 0
        sp = []
        for i, t in enumerate(ids):
            mi = m[i]
            if not mi.any():
                continue
            oi = o[i][mi]
            pi = p[i][mi] + (corr[dates[i]] if lect == 'nivel_corregido' else 0.0)
            so += oi.sum(); so2 += (oi ** 2).sum(); sr += ((oi - pi) ** 2).sum(); n += oi.size
            s = compute_spaef(oi, pi)
            if s is not None and np.isfinite(s):
                sp.append(float(s))
        ss = so2 - so ** 2 / n
        out[lect] = {'R2': 1.0 - sr / ss, 'SPAEF': float(np.mean(sp)), 'n_teselas_spaef': len(sp)}
    out['sesgo_medio_por_fecha'] = {d: round(-c, 4) for d, c in sorted(corr.items())}
    return out


def mmm(v):
    v = np.asarray(v, float)
    return {'media': round(float(v.mean()), 4), 'min': round(float(v.min()), 4),
            'max': round(float(v.max()), 4), 'por_semilla': [round(float(x), 4) for x in v]}


def main():
    ap = argparse.ArgumentParser(description='Paso 2: tal cual y con el nivel corregido')
    ap.add_argument('--clim-dir', default='results/clim_v1')
    ap.add_argument('--b-dir', default='results/sat_v2_instancia')
    ap.add_argument('--out', default='analysis/clim_v1_nivel_corregido.json')
    a = ap.parse_args()

    def bclim(s, suf):
        n = f'resunetpp_clim1_Bclim_s{s}'
        return os.path.join(a.clim_dir, n, f'{n}{suf}')

    def b(s, suf):
        n = f'resunetpp_sat2_B_s{s}'
        return os.path.join(a.b_dir, n, f'{n}{suf}')

    res, fallos = {}, []
    for epoca, e in (('mejor_epoca', ''), ('ultima_epoca', '_last')):
        res[epoca] = {}
        for modelo, base in (('Bclim_con_climatologia', lambda s: bclim(s, e)),
                             ('Bclim_sin_climatologia', lambda s: bclim(s, e + '_sinclim')),
                             ('B_paso1', lambda s: b(s, e))):
            filas = {}
            for s in SEEDS:
                npz, mj = base(s) + '_predictions.npz', base(s) + '_metrics.json'
                if not os.path.exists(npz):
                    print(f'  [aviso] falta {npz}')
                    continue
                r = metrics(*load(npz))
                ref = json.load(open(mj))
                for k in ('R2', 'SPAEF'):
                    if abs(r['tal_cual'][k] - ref[k]) > TOL:
                        fallos.append(f'{mj}: {k} {r["tal_cual"][k]:.4f} frente a {ref[k]:.4f}')
                filas[s] = r
                print(f"{epoca:12s} {modelo:24s} s{s:<3d} tal cual R2 {r['tal_cual']['R2']:.3f} "
                      f"SPAEF {r['tal_cual']['SPAEF']:.3f} | nivel corregido R2 "
                      f"{r['nivel_corregido']['R2']:.3f} SPAEF {r['nivel_corregido']['SPAEF']:.3f}", flush=True)
            if filas:
                res[epoca][modelo] = {
                    lect: {k: mmm([filas[s][lect][k] for s in filas]) for k in ('R2', 'SPAEF')}
                    for lect in ('tal_cual', 'nivel_corregido')}
                res[epoca][modelo]['semillas'] = sorted(filas)

    cs = os.path.join(a.clim_dir, 'climatologia_sola', 'climatologia_sola')
    r = metrics(*load(cs + '_predictions.npz'))
    ref = json.load(open(cs + '_metrics.json'))
    for k in ('R2', 'SPAEF'):
        if abs(r['tal_cual'][k] - ref[k]) > TOL:
            fallos.append(f'climatologia_sola: {k} {r["tal_cual"][k]:.4f} frente a {ref[k]:.4f}')
    res['climatologia_sola'] = {lect: {k: round(r[lect][k], 4) for k in ('R2', 'SPAEF')}
                                for lect in ('tal_cual', 'nivel_corregido')}
    res['climatologia_sola']['sesgo_medio_por_fecha'] = r['sesgo_medio_por_fecha']
    res['control_tal_cual_frente_a_metrics_json'] = {'tolerancia': TOL, 'fallos': fallos,
                                                     'superado': not fallos}
    res['descripcion'] = ('Nivel corregido = nivel perfecto aditivo por fecha (media observada - '
                          'media predicha sobre los pixeles con nieve de la fecha). Mide solo el '
                          'dibujo; no es desplegable.')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)

    print('\nResumen (media [min-max] de 3 semillas)')
    for epoca in ('mejor_epoca', 'ultima_epoca'):
        print(f'  [{epoca}]')
        for modelo, v in res[epoca].items():
            t, c = v['tal_cual'], v['nivel_corregido']
            print(f"    {modelo:24s} tal cual R2 {t['R2']['media']:.3f} SPAEF {t['SPAEF']['media']:.3f} | "
                  f"nivel corregido R2 {c['R2']['media']:.3f} [{c['R2']['min']:.3f}-{c['R2']['max']:.3f}] "
                  f"SPAEF {c['SPAEF']['media']:.3f} [{c['SPAEF']['min']:.3f}-{c['SPAEF']['max']:.3f}]")
    c = res['climatologia_sola']
    print(f"    {'climatologia_sola':24s} tal cual R2 {c['tal_cual']['R2']:.3f} SPAEF {c['tal_cual']['SPAEF']:.3f} | "
          f"nivel corregido R2 {c['nivel_corregido']['R2']:.3f} SPAEF {c['nivel_corregido']['SPAEF']:.3f}")
    print(f"\nControl tal cual frente a metrics.json: {'SUPERADO' if not fallos else 'NO SUPERADO'}")
    for f in fallos:
        print('  ' + f)
    print(f'Resultados: {a.out}')
    if fallos:
        raise SystemExit('STOP: la lectura tal cual no reproduce las metricas de la evaluacion.')


if __name__ == '__main__':
    main()
