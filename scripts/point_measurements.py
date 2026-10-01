r"""
Experimento A (v2) - Medidas puntuales: zonas accesibles y peor caso
====================================================================

Que cambia respecto a la v1 (revision de Hernan)
-----------------------------------------------
1. MUESTREO EN ZONAS ACCESIBLES. Puntos al azar sobre toda la cuenca son
   optimistas: nadie sondea en una pared. Se anade un muestreo restringido a
   pixeles con pendiente < 30 grados (umbral habitual de terreno de aludes),
   con sensibilidad a 25 y 35 grados. La pendiente sale del DEM de 1 m en la
   posicion real de cada tile (data/tile_georef.csv). Los puntos son
   accesibles; el mapa que se corrige y se evalua es el COMPLETO.

2. DISPERSION ENTRE CAMPANAS. Una campana es una salida de campo: una fecha
   con N puntos. Ademas de la media, se reportan los percentiles 10, 50 y 90
   del R2 por campana (R2 de esa fecha tras corregir), sobre las 200
   repeticiones y todas las fechas de test. El P10 es el peor caso razonable.

Muestreo de puntos: cada tile aporta al pool un 2% de sus pixeles de nieve,
en proporcion a su tamano, para que el pool represente la zona nevada con el
mismo peso que la evaluacion.

Lo demas, igual que la v1: anyo nuevo sin vuelo, cuatro folds de E2,
patrones climatologia y red (semillas 7, 42, 123), ruido de 10 cm
(sensibilidad 0 y 20 cm), metodos ingenuo / patron / patron+sat, este
ultimo como principal.

Limitaciones declaradas
-----------------------
* Medidas sacadas del propio mapa del dron (optimista; por eso el ruido).
* La varianza de residuos que pondera las medidas se toma de la fecha.
* La pendiente es un indicador de accesibilidad, no un mapa de accesos.
* El escenario combinado, terreno nuevo en un anyo nuevo, no esta probado:
  ningun split excluye ambas cosas a la vez.

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
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_test import (date_of, mean_from_masks, ridge_fit,   # noqa: E402
                                ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX)
from climatology_baseline import build_climatology                   # noqa: E402

TILE = 256
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
N_LIST = [1, 3, 5, 10, 20, 50, 100]
NOISES = [0.10, 0.0, 0.20]           # la primera es la principal
POOLS = ['todos', 'pend<30', 'pend<25', 'pend<35']
REPS = 200
POOL_RATE = 0.02     # fraccion de los pixeles de cada tile que entra en el pool
METHODS = ['ingenuo', 'patron', 'patron+sat']
PRINCIPAL = 'patron+sat'
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
DEM_PATH = os.path.join('Articulo 1', 'Data', 'izas', 'LiDAR', 'Topografia', 'DEMbigIzas_1m.tif')


def slope_map(dem_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        res = src.res[0]
        nod = src.nodata
    if nod is not None:
        dem[dem == nod] = np.nan
    gy, gx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


class DateStats:
    """Estadisticos suficientes por fecha y pools de puntos por accesibilidad."""
    def __init__(self):
        self.n = 0; self.s1 = 0.0; self.s2 = 0.0; self.sp = 0.0
        self.so = 0.0; self.so2 = 0.0
        self.pool = {k: ([], []) for k in POOLS}

    def add(self, o, p, slope, rng):
        d = o - p
        self.n += o.size; self.s1 += d.sum(); self.s2 += (d ** 2).sum()
        self.sp += p.sum(); self.so += o.sum(); self.so2 += (o ** 2).sum()
        sel = {'todos': np.ones(o.size, bool), 'pend<30': slope < 30,
               'pend<25': slope < 25, 'pend<35': slope < 35}
        for k in POOLS:
            idx = np.flatnonzero(sel[k])
            if idx.size == 0:
                continue
            # Proporcional a los pixeles del tile: el pool representa la zona
            # nevada con el mismo peso que la evaluacion. (La v2 inicial cogia
            # hasta 50 pixeles por tile, lo que sobrerrepresentaba los tiles con
            # poca nieve y sesgaba la estimacion del nivel; se vio en el P10.)
            n_take = max(1, int(round(POOL_RATE * idx.size)))
            idx = rng.choice(idx, n_take, replace=False)
            self.pool[k][0].append(o[idx]); self.pool[k][1].append(p[idx])

    def finish(self):
        self.mean_p = self.sp / self.n
        self.pools = {}
        for k, (lo, lp) in self.pool.items():
            if lo:
                po, pp = np.concatenate(lo), np.concatenate(lp)
                self.pools[k] = (po, pp, float(np.var(po - pp)))
        del self.pool

    def r2(self, dl):
        sse = self.s2 - 2 * dl * self.s1 + self.n * dl ** 2
        sst = self.so2 - self.so ** 2 / self.n
        return 1.0 - sse / sst


def r2_global(stats, deltas, so, so2, n_tot):
    sst = so2 - so ** 2 / n_tot
    sse = 0.0
    for d, st in stats.items():
        dl = deltas[d]
        sse = sse + (st.s2 - 2 * dl * st.s1 + st.n * dl ** 2)
    return 1.0 - sse / sst


def run_pattern(stats, so, so2, n_tot, prior, var_prior, noise, pool_key, rng):
    zero = {d: np.zeros(1) for d in stats}
    orac = {d: np.array([st.s1 / st.n]) for d, st in stats.items()}
    sat = {d: np.array([prior[d] - st.mean_p]) for d, st in stats.items()}
    out = {'sin': float(r2_global(stats, zero, so, so2, n_tot)[0]),
           'oraculo': float(r2_global(stats, orac, so, so2, n_tot)[0]),
           'satelite': float(r2_global(stats, sat, so, so2, n_tot)[0]),
           # Mismo peso que las campanas corregidas (REPS por fecha). Con una
           # sola entrada por fecha, el percentil se interpolaba entre fechas y
           # no era comparable con el de la correccion.
           'camp_sin': [float(st.r2(0.0)) for st in stats.values() for _ in range(REPS)],
           'camp_oraculo': [float(st.r2(st.s1 / st.n)) for st in stats.values() for _ in range(REPS)],
           'oraculo_por_fecha': sorted(float(st.r2(st.s1 / st.n)) for st in stats.values()),
           'curvas': {m: [] for m in METHODS},
           'campanas': {m: [] for m in METHODS},
           'fechas_sin_pool': sum(1 for st in stats.values() if pool_key not in st.pools)}
    for N in N_LIST:
        dl = {m: {} for m in METHODS}
        camp = {m: [] for m in METHODS}
        for d, st in stats.items():
            po, pp, rv = st.pools.get(pool_key, st.pools['todos'])
            idx = rng.integers(0, po.size, size=(REPS, N))
            meas = po[idx] + (rng.normal(0, noise, size=(REPS, N)) if noise > 0 else 0.0)
            pat = pp[idx]
            dpts = (meas - pat).mean(axis=1)
            vpts = (rv + noise ** 2) / N
            wp, ws = 1.0 / vpts, 1.0 / var_prior
            dl['ingenuo'][d] = meas.mean(axis=1) - st.mean_p
            dl['patron'][d] = dpts
            dl['patron+sat'][d] = (dpts * wp + (prior[d] - st.mean_p) * ws) / (wp + ws)
            for m in METHODS:
                camp[m].append(st.r2(dl[m][d]))
        for m in METHODS:
            out['curvas'][m].append(float(np.mean(r2_global(stats, dl[m], so, so2, n_tot))))
            out['campanas'][m].append(np.concatenate(camp[m]).tolist())
    return out


def n90(sin, orac, curva):
    return next((N for N, v in zip(N_LIST, curva) if v - sin >= 0.9 * (orac - sin)), None)


def main():
    ap = argparse.ArgumentParser(description='Experimento A v2: medidas puntuales')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--signals', default='data/scale_signals.csv')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--dem', default=DEM_PATH)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/eA_point_measurements_v2.json')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['k'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('k')
    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)

    print('Calculando pendientes desde el DEM...', flush=True)
    slope = slope_map(args.dem)

    # resultados: res[patron][pool][ruido] = lista de runs (uno por fold, o por fold x semilla)
    res = {p: {k: {str(nz): [] for nz in NOISES} for k in POOLS} for p in ('climatologia', 'red')}
    for fold in args.folds:
        print('=' * 72); print(f'Fold {fold}', flush=True)
        clim, _ = build_climatology(geo, fold, args.masks, shape)
        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        tr_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        trm = mean_from_masks(tr_ids, args.masks)
        trd = [d for d in sorted(trm) if d in sig.index]
        X = sig.loc[trd, ['snow_pct']].to_numpy(float); y = np.array([trm[d] for d in trd])
        cv = [ridge_predict(ridge_fit(np.delete(X, j, 0), np.delete(y, j)), X[j:j + 1])[0] - y[j]
              for j in range(len(trd))]
        var_prior = float(np.mean(np.square(cv)))
        model = ridge_fit(X, y)

        paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                   f'resunetpp_v3_loyo{fold}_predictions.npz')}
        for s in ('42', '123'):
            paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                    f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')
        clim_hecha = False
        for s in SEEDS:
            if not os.path.exists(paths[s]):
                print(f'  [aviso] falta la semilla {s}'); continue
            rng = np.random.default_rng(int(s))
            z = np.load(paths[s])
            ids, preds, tgts = z['tile_ids'], z['preds'], z['targets']
            vals = z['valids'] if 'valids' in z.files else None
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
            prior = {d: float(v) for d, v in zip(
                dates, ridge_predict(model, sig.loc[dates, ['snow_pct']].to_numpy(float)))}
            for k in POOLS:
                for nz in NOISES:
                    if k != 'todos' and nz != NOISES[0]:
                        continue            # sensibilidad al ruido solo con el muestreo libre
                    res['red'][k][str(nz)].append(run_pattern(
                        net, so, so2, n_tot, prior, var_prior, nz, k, np.random.default_rng(1000 + int(s))))
                    if not clim_hecha:
                        res['climatologia'][k][str(nz)].append(run_pattern(
                            cli, so, so2, n_tot, prior, var_prior, nz, k, np.random.default_rng(99)))
            clim_hecha = True
            print(f'  semilla {s} evaluada ({len(dates)} fechas de test)', flush=True)

    principal = str(NOISES[0])

    def resumen(runs):
        sin = np.mean([r['sin'] for r in runs]); orac = np.mean([r['oraculo'] for r in runs])
        sat = np.mean([r['satelite'] for r in runs])
        curvas = {m: [float(np.mean([r['curvas'][m][k] for r in runs])) for k in range(len(N_LIST))]
                  for m in METHODS}
        camp = {m: [np.concatenate([np.asarray(r['campanas'][m][k]) for r in runs]) for k in range(len(N_LIST))]
                for m in METHODS}
        camp_sin = np.concatenate([r['camp_sin'] for r in runs])
        camp_orac = np.concatenate([r['camp_oraculo'] for r in runs])
        sin_pool = sum(r['fechas_sin_pool'] for r in runs)
        return sin, orac, sat, curvas, camp, camp_sin, camp_orac, sin_pool

    for pool in ('todos', 'pend<30'):
        print('=' * 72)
        print(f"MUESTREO: {'toda la cuenca' if pool == 'todos' else 'solo pendiente < 30 grados'}"
              f'  (ruido {float(principal) * 100:.0f} cm)')
        for pat in ('climatologia', 'red'):
            sin, orac, sat, curvas, camp, csin, corac, sp = resumen(res[pat][pool][principal])
            print(f'\n  PATRON: {pat.upper()}')
            print(f'    media: sin corregir {sin:.3f} | solo satelite {sat:.3f} | oraculo {orac:.3f}')
            if sp:
                print(f'    [aviso] {sp} fecha-modelo sin puntos accesibles: se usaron todos los pixeles')
            print(f"    {'N':>5}" + ''.join(f'{m:>12}' for m in METHODS) +
                  f"   | campanas {PRINCIPAL}: P10    P50    P90")
            for k, N in enumerate(N_LIST):
                p10, p50, p90 = np.percentile(camp[PRINCIPAL][k], [10, 50, 90])
                print(f'    {N:>5}' + ''.join(f'{curvas[m][k]:>12.3f}' for m in METHODS) +
                      f'   |                {p10:>6.3f} {p50:>6.3f} {p90:>6.3f}')
            p10s, p50s, p90s = np.percentile(csin, [10, 50, 90])
            p10o, p50o, p90o = np.percentile(corac, [10, 50, 90])
            print(f"    {'sin corregir':>41}   |                {p10s:>6.3f} {p50s:>6.3f} {p90s:>6.3f}")
            print(f"    {'oraculo':>41}   |                {p10o:>6.3f} {p50o:>6.3f} {p90o:>6.3f}")
            print('    N90: ' + ', '.join(f"{m} {n90(sin, orac, curvas[m]) or '>100'}" for m in METHODS))
            peores = sorted(v for r in res[pat][pool][principal] for v in r['oraculo_por_fecha'])[:3]
            print('    tres peores campanas incluso con el nivel perfecto: ' +
                  ', '.join(f'{v:.3f}' for v in peores))

    print('=' * 72)
    print(f'SENSIBILIDAD (metodo principal {PRINCIPAL})')
    k10 = N_LIST.index(10)
    print(f"  {'muestreo / ruido':<28}{'clim: R2@10':>13}{'P10@10':>9}{'N90':>6}"
          f"{'red: R2@10':>13}{'P10@10':>9}{'N90':>6}")
    casos = [(p, principal) for p in POOLS] + [('todos', str(nz)) for nz in NOISES[1:]]
    for pool, nz in casos:
        fila = f"  {pool + ', ' + str(int(float(nz) * 100)) + ' cm':<28}"
        for pat in ('climatologia', 'red'):
            sin, orac, sat, curvas, camp, *_ = resumen(res[pat][pool][nz])
            p10 = np.percentile(camp[PRINCIPAL][k10], 10)
            nn = n90(sin, orac, curvas[PRINCIPAL])
            fila += f"{curvas[PRINCIPAL][k10]:>13.3f}{p10:>9.3f}{(nn or '>100'):>6}"
        print(fila)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    resumen_json = {}
    for pat in res:
        for pool in POOLS:
            for nz, runs in res[pat][pool].items():
                if not runs:
                    continue
                sin, orac, sat, curvas, camp, csin, corac, sp = resumen(runs)
                resumen_json[f'{pat}|{pool}|{nz}'] = {
                    'sin': sin, 'oraculo': orac, 'satelite': sat, 'curvas': curvas,
                    'campanas_p10_p50_p90': {m: [np.percentile(c, [10, 50, 90]).tolist() for c in camp[m]]
                                             for m in METHODS},
                    'fechas_sin_pool': sp}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'N': N_LIST, 'metodos': METHODS, 'principal': PRINCIPAL,
                   'resumen': resumen_json}, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
