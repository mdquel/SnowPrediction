r"""
Experimento A - ¿Cuantas medidas puntuales hacen falta para corregir el nivel?
==============================================================================

Pregunta
--------
Si un anyo no se vuela, pero se mide la profundidad de nieve en unos pocos
puntos, ¿cuantos hacen falta para corregir el nivel del mapa casi tan bien
como si se conociera la media real (el oraculo)?

Escenario: anyo nuevo sin vuelo, los cuatro folds de E2. Dos patrones:
    CLIMATOLOGIA  media de los anyos de entrenamiento (zona ya sobrevolada)
    RED           ResUNet++ de E2, semillas 7, 42 y 123 (sin vuelos previos)

Simulacion de las medidas
-------------------------
Para cada fecha de test se eligen al azar N pixeles del mapa real, como si
se hubiera sondeado ahi, y se les suma un error de medida gaussiano. 200
repeticiones con puntos distintos. N = 1, 3, 5, 10, 20, 50, 100.
Los puntos se extraen de una muestra de hasta 50 pixeles por tile.

Tres metodos de correccion (todos aditivos: E5 mostro que el error es de
sesgo, no de dispersion)
------------------------------------------------------------------------
    ingenuo        llevar la media del mapa a la media de las medidas
    patron         desplazar el mapa la media de (medida - patron) en cada
                   punto: el patron explica la variacion entre sitios y la
                   diferencia aisla el error de nivel
    patron+sat     lo anterior combinado con el regresor de E5 (solo
                   cobertura) por varianza inversa. La varianza del
                   regresor se estima por validacion cruzada dejando una
                   fecha fuera en las fechas de train. CONFIGURACION
                   PRINCIPAL, fijada antes de ejecutar.

Medidas
-------
R2 tras la correccion (pixeles con nieve y dato valido, mascara comun a los
dos patrones), frente a sin corregir y al oraculo. Resumen: N90, el numero
de puntos necesario para capturar el 90% de la mejora del oraculo.

Cautelas (declaradas)
---------------------
* Las medidas salen del propio mapa del dron: escenario optimista. Por eso
  el ruido (10 cm principal; tambien 0 y 20 cm).
* La varianza de los residuos (medida - patron) de cada fecha, que pondera
  las medidas en patron+sat, se toma de la propia fecha. En campo habria
  que estimarla; es una ligera ventaja informativa, no sobre la media.
* Se usa la red temporal. El caso "zona nueva y anyo nuevo" a la vez no se
  puede evaluar con los modelos disponibles.

Uso
---
    python scripts/point_measurements.py ^
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
from recalibration_test import (date_of, mean_from_masks, ridge_fit,   # noqa: E402
                                ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX)
from climatology_baseline import build_climatology                   # noqa: E402

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
N_LIST = [1, 3, 5, 10, 20, 50, 100]
NOISES = [0.10, 0.0, 0.20]          # la primera es la principal
REPS = 200
POOL_PER_TILE = 50
METHODS = ['ingenuo', 'patron', 'patron+sat']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'


# ----------------------------------------------------------------------
class DateStats:
    """Estadisticos suficientes por fecha para evaluar cualquier desplazamiento.

    Con S1 = sum(o - p), S2 = sum((o - p)^2), el error tras desplazar el
    patron una cantidad delta es S2 - 2*delta*S1 + n*delta^2, asi que el R2
    de cualquier correccion aditiva se calcula sin guardar los pixeles.
    """
    def __init__(self):
        self.n = 0; self.s1 = 0.0; self.s2 = 0.0; self.sp = 0.0
        self.pool_o = []; self.pool_p = []

    def add(self, o, p, rng):
        d = o - p
        self.n += o.size; self.s1 += d.sum(); self.s2 += (d ** 2).sum(); self.sp += p.sum()
        k = min(POOL_PER_TILE, o.size)
        idx = rng.choice(o.size, k, replace=False)
        self.pool_o.append(o[idx]); self.pool_p.append(p[idx])

    def finish(self):
        self.pool_o = np.concatenate(self.pool_o); self.pool_p = np.concatenate(self.pool_p)
        self.mean_p = self.sp / self.n
        self.res_var = float(np.var(self.pool_o - self.pool_p))


def r2_from(stats, deltas, so, so2, n_tot):
    """R2 para desplazamientos por fecha. deltas: dict fecha -> array (reps,)."""
    ss_tot = so2 - so ** 2 / n_tot
    sse = 0.0
    for d, st in stats.items():
        dl = deltas[d]
        sse = sse + (st.s2 - 2 * dl * st.s1 + st.n * dl ** 2)
    return 1.0 - sse / ss_tot


def run_pattern(stats, so, so2, n_tot, prior, var_prior, noise, rng):
    """R2 medio por N y metodo, mas sin corregir, oraculo y solo satelite."""
    zero = {d: np.zeros(1) for d in stats}
    orac = {d: np.array([st.s1 / st.n]) for d, st in stats.items()}
    sat = {d: np.array([prior[d] - st.mean_p]) for d, st in stats.items()}
    out = {'sin': float(r2_from(stats, zero, so, so2, n_tot)[0]),
           'oraculo': float(r2_from(stats, orac, so, so2, n_tot)[0]),
           'satelite': float(r2_from(stats, sat, so, so2, n_tot)[0]),
           'curvas': {m: [] for m in METHODS}}
    for N in N_LIST:
        dl = {m: {} for m in METHODS}
        for d, st in stats.items():
            idx = rng.integers(0, st.pool_o.size, size=(REPS, N))
            meas = st.pool_o[idx] + rng.normal(0, noise, size=(REPS, N)) if noise > 0 else st.pool_o[idx]
            pat = st.pool_p[idx]
            dl['ingenuo'][d] = meas.mean(axis=1) - st.mean_p
            d_pts = (meas - pat).mean(axis=1)
            dl['patron'][d] = d_pts
            v_pts = (st.res_var + noise ** 2) / N
            w_p, w_s = 1.0 / v_pts, 1.0 / var_prior
            dl['patron+sat'][d] = (d_pts * w_p + (prior[d] - st.mean_p) * w_s) / (w_p + w_s)
        for m in METHODS:
            out['curvas'][m].append(float(np.mean(r2_from(stats, dl[m], so, so2, n_tot))))
    return out


def n90(res):
    gain = res['oraculo'] - res['sin']
    out = {}
    for m in METHODS:
        out[m] = next((N for N, v in zip(N_LIST, res['curvas'][m])
                       if v - res['sin'] >= 0.9 * gain), None)
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Experimento A: medidas puntuales')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/eA_point_measurements.json')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')

    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)

    todo = {}
    for fold in args.folds:
        print('=' * 72)
        print(f'Fold {fold}', flush=True)
        clim, _ = build_climatology(geo, fold, args.masks, shape)

        # regresor del satelite y su varianza por validacion cruzada
        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        trm = mean_from_masks(tr_ids, args.masks)
        trd = [d for d in sorted(trm) if d in sig.index]
        X = sig.loc[trd, ['snow_pct']].to_numpy(float)
        y = np.array([trm[d] for d in trd])
        res_cv = [ridge_predict(ridge_fit(np.delete(X, j, 0), np.delete(y, j)), X[j:j + 1])[0] - y[j]
                  for j in range(len(trd))]
        var_prior = float(np.mean(np.square(res_cv)))
        model = ridge_fit(X, y)
        print(f'  regresor: {len(trd)} fechas de train, error CV {np.sqrt(var_prior):.3f} m')

        paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                   f'resunetpp_v3_loyo{fold}_predictions.npz')}
        for s in ('42', '123'):
            paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                    f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')

        fold_res = {'clim': None, 'red': []}
        for s in SEEDS:
            if not os.path.exists(paths[s]):
                print(f'  [aviso] falta la semilla {s}')
                continue
            rng = np.random.default_rng(int(s))
            z = np.load(paths[s])
            ids = z['tile_ids']; preds = z['preds']; tgts = z['targets']
            vals = z['valids'] if 'valids' in z.files else None
            net, cli = {}, {}
            so = so2 = 0.0; n_tot = 0
            for i, t in enumerate(ids):
                t = str(t); d = date_of(t)
                if d not in sig.index or t not in pos:
                    continue
                r, c = pos[t]
                cp = clim[r:r + TILE, c:c + TILE]
                tg = tgts[i]
                m = (tg > SNOW_THRESHOLD) & np.isfinite(tg) & np.isfinite(cp)
                if vals is not None:
                    m &= vals[i] > 0.5
                if m.sum() < MIN_VALID_PX:
                    continue
                o = tg[m].astype(np.float64)
                so += o.sum(); so2 += (o ** 2).sum(); n_tot += o.size
                net.setdefault(d, DateStats()).add(o, preds[i][m].astype(np.float64), rng)
                if fold_res['clim'] is None:
                    cli.setdefault(d, DateStats()).add(o, cp[m].astype(np.float64), rng)
            for st in list(net.values()) + list(cli.values()):
                st.finish()
            dates = sorted(net)
            prior = {d: float(v) for d, v in zip(
                dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}

            por_ruido = {}
            for noise in NOISES:
                por_ruido[str(noise)] = run_pattern(net, so, so2, n_tot, prior, var_prior, noise,
                                                    np.random.default_rng(1000 + int(s)))
            fold_res['red'].append(por_ruido)
            if fold_res['clim'] is None:
                fold_res['clim'] = {str(nz): run_pattern(cli, so, so2, n_tot, prior, var_prior, nz,
                                                         np.random.default_rng(99))
                                    for nz in NOISES}
            print(f'  semilla {s} evaluada ({len(dates)} fechas de test)', flush=True)
        todo[fold] = fold_res

    # ---------------- resumen (ruido principal) ----------------
    def media_red(fold_res, noise):
        runs = [r[noise] for r in fold_res['red']]
        out = {k: float(np.mean([x[k] for x in runs])) for k in ('sin', 'oraculo', 'satelite')}
        out['curvas'] = {m: list(np.mean([x['curvas'][m] for x in runs], axis=0)) for m in METHODS}
        return out

    for noise in (str(NOISES[0]),):
        print('=' * 72)
        print(f'RESUMEN (ruido de medida {float(noise) * 100:.0f} cm, media de los folds)')
        for nombre in ('climatologia', 'red'):
            per = [todo[f]['clim'][noise] if nombre == 'climatologia' else media_red(todo[f], noise)
                   for f in todo]
            sin = np.mean([p['sin'] for p in per]); orac = np.mean([p['oraculo'] for p in per])
            sat = np.mean([p['satelite'] for p in per])
            print(f'\n  PATRON: {nombre.upper()}')
            print(f'    sin corregir {sin:.3f} | solo satelite {sat:.3f} | oraculo {orac:.3f}')
            print(f"    {'N':>5}" + ''.join(f'{m:>13}' for m in METHODS))
            for k, N in enumerate(N_LIST):
                print(f'    {N:>5}' + ''.join(f"{np.mean([p['curvas'][m][k] for p in per]):>13.3f}"
                                              for m in METHODS))
            agg = {'sin': sin, 'oraculo': orac,
                   'curvas': {m: [np.mean([p['curvas'][m][k] for p in per]) for k in range(len(N_LIST))]
                              for m in METHODS}}
            nn = n90(agg)
            print('    N90 (puntos para el 90% de la mejora del oraculo): ' +
                  ', '.join(f"{m} {nn[m] if nn[m] else '>100'}" for m in METHODS))

    print('\nSensibilidad al ruido (N90 del metodo principal, patron+sat):')
    for noise in NOISES:
        linea = []
        for nombre in ('climatologia', 'red'):
            per = [todo[f]['clim'][str(noise)] if nombre == 'climatologia' else media_red(todo[f], str(noise))
                   for f in todo]
            agg = {'sin': np.mean([p['sin'] for p in per]), 'oraculo': np.mean([p['oraculo'] for p in per]),
                   'curvas': {m: [np.mean([p['curvas'][m][k] for p in per]) for k in range(len(N_LIST))]
                              for m in METHODS}}
            v = n90(agg)['patron+sat']
            linea.append(f"{nombre} {v if v else '>100'}")
        print(f'  ruido {noise * 100:>3.0f} cm: ' + ' | '.join(linea))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'N': N_LIST, 'metodos': METHODS, 'ruidos': NOISES, 'folds': todo}, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
