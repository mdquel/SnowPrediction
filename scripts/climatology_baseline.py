r"""
Experimento B - ¿El patron de la red supera a la climatologia?  (v2)
====================================================================

Pregunta
--------
La propuesta es obtener el PATRON de la red y la ESCALA de fuera. Lo primero
que preguntaria un revisor de la literatura de repetibilidad de patrones de
nieve: ¿para que una red, si el patron se puede sacar promediando los mapas
de anyos anteriores?

    RED            prediccion del ResUNet++ (E2, semillas 7, 42 y 123)
    CLIMATOLOGIA   para cada pixel del terreno, la media de lo observado en
                   ese mismo sitio en los anyos de entrenamiento del fold

Por que v2
----------
La primera version emparejaba tiles por NOMBRE, y el nombre no es un sitio
fijo: cada vuelo tiene su propio origen (hasta 915 m de diferencia E-O).
Aquella climatologia promediaba sitios distintos y no valia. Esta version
usa las coordenadas reales de cada tile (data/tile_georef.csv, validadas
contra el DEM con residuo 0.000 m).

Construccion de la climatologia
-------------------------------
Para cada fecha de entrenamiento se reconstruye su mapa de nieve completo
sobre la rejilla del DEM, colocando cada tile en su posicion real, y despues
se promedian esos mapas FECHA A FECHA. Promediar tile a tile daria mas peso
a los sitios cubiertos por varios tiles solapados de un mismo dia.

Diseno (fijado antes de la primera ejecucion, sin cambios)
----------------------------------------------------------
* Solo anyos de entrenamiento del fold: ni validacion (2024) ni test.
* Tres versiones para ambos patrones:
      1. tal cual
      2. nivel perfecto (media real de cada fecha, oraculo): mide SOLO la
         calidad del dibujo
      3. nivel del regresor de E5 (solo cobertura, aditiva): desplegable
* R2 sobre pixeles con nieve y dato valido; correlacion media por tile.
* Pixeles sin climatologia: excluidos para AMBOS; se reporta la cobertura.
* Test restringido a fechas con senales, como en E5.
* Metricas acumuladas tile a tile (sin juntar todos los pixeles en memoria).

Criterio (fijado antes, sin cambios)
------------------------------------
Comparacion principal: version 2, red = media de las tres semillas.
    red gana en >= 3 de 4 folds          ->  el patron de la red aporta
    climatologia gana en >= 3 de 4 folds ->  donde hay vuelos previos basta la
                                             climatologia; la red aporta en
                                             zonas no sobrevoladas (exp. C)
    2 y 2                                ->  no concluyente

Cautela: queda un desfase sub-pixel entre fechas (0-1 m); se midio que su
efecto es despreciable (check_subpixel_shift.py, techo R2 0.992).

Uso
---
    python scripts/climatology_baseline.py ^
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
from recalibration_test import (date_of, mean_from_masks, ridge_fit,  # noqa: E402
                                ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX)

TILE = 256
VAL_YEAR = 2024
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
SIGNALS_CSV = 'data/scale_signals.csv'
GEOREF_CSV = 'data/tile_georef.csv'


# ----------------------------------------------------------------------
def build_climatology(geo, fold, masks_dir, shape):
    """Media fecha a fecha de los mapas de nieve de los anyos de train."""
    test_year = int(fold)
    years = geo['date'].str[:4].astype(int)
    sub = geo[~years.isin([test_year, VAL_YEAR])]
    H, W = shape
    suma = np.zeros((H, W), dtype=np.float64)
    cuenta = np.zeros((H, W), dtype=np.int16)
    fechas = sorted(sub['date'].unique())
    for d in fechas:
        mosaico = np.full((H, W), np.nan, dtype=np.float32)
        for t, r, c in sub.loc[sub['date'] == d, ['tile_id', 'row_real', 'col_real']].itertuples(index=False):
            path = os.path.join(masks_dir, t)
            if not os.path.exists(path):
                continue
            m = np.load(path).astype(np.float32)
            ok = np.isfinite(m) & (m > -100)
            dst = mosaico[r:r + TILE, c:c + TILE]
            dst[ok] = m[ok]                  # solapes del mismo dia: mismo valor
        v = np.isfinite(mosaico)
        suma[v] += mosaico[v]
        cuenta[v] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        clim = np.where(cuenta > 0, suma / np.maximum(cuenta, 1), np.nan)
    return clim, len(fechas)


def new_acc():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0}


def add(acc, o, p):
    acc['n'] += o.size
    acc['so'] += o.sum()
    acc['so2'] += (o ** 2).sum()
    acc['sr'] += ((o - p) ** 2).sum()


def r2(acc):
    ss = acc['so2'] - acc['so'] ** 2 / acc['n']
    return 1.0 - acc['sr'] / ss if ss > 0 else float('nan')


def evaluate_seed(npz_path, pos, clim, sig_index, model):
    """Metricas de red y climatologia para un modelo, en streaming."""
    z = np.load(npz_path)
    ids = z['tile_ids']
    dates = np.array([date_of(t) for t in ids])
    keep = np.where(np.isin(dates, list(sig_index)))[0]
    preds = z['preds']
    tgts = z['targets']
    vals = z['valids'] if 'valids' in z.files else None

    # tiles utilizables y su mascara comun (se guarda la mascara, no los datos)
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

    # pasada 1: medias por fecha (real, red, climatologia) sobre la mascara comun
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
        tdates, ridge_predict(model, sig_index_frame.loc[tdates, ['snow_pct']].to_numpy(float)))}

    # pasada 2: R2 de las tres versiones y correlacion por tile
    accs = {(p, v): new_acc() for p in ('red', 'clim') for v in ('raw', 'orac', 'regr')}
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
            add(accs[(p, 'regr')], o, x + (regr[d] - medias[p][d]))

    out = {}
    for p in ('red', 'clim'):
        out[p] = dict(r2_raw=r2(accs[(p, 'raw')]), r2_oraculo=r2(accs[(p, 'orac')]),
                      r2_regresor=r2(accs[(p, 'regr')]), corr=float(np.mean(corrs[p])))
    out['cobertura'] = comun_px / max(base_px, 1)
    return out


sig_index_frame = None     # se fija en main


# ----------------------------------------------------------------------
def main():
    global sig_index_frame
    ap = argparse.ArgumentParser(description='Experimento B v2: red frente a climatologia alineada')
    ap.add_argument('--seed7-dir', required=True)
    ap.add_argument('--seeds-dir', required=True)
    ap.add_argument('--masks', default=MASKS_DIR)
    ap.add_argument('--signals', default=SIGNALS_CSV)
    ap.add_argument('--georef', default=GEOREF_CSV)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/eB_climatology_v2.json')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['fecha_key'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig_index_frame = sig.set_index('fecha_key')

    geo = pd.read_csv(args.georef, dtype={'date': str})
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)

    resultados, veredictos = {}, []
    for fold in args.folds:
        print('=' * 76)
        print(f'Fold {fold}')
        print('  construyendo climatologia en coordenadas reales...', flush=True)
        clim, n_f = build_climatology(geo, fold, args.masks, shape)
        print(f'  climatologia: media de {n_f} fechas de train, '
              f'{100 * np.isfinite(clim).mean():.0f}% de la rejilla con dato')

        df = pd.read_csv(FOLD_CSV.format(fold=fold))
        train_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        tr_means = mean_from_masks(train_ids, args.masks)
        tr_dates = [d for d in sorted(tr_means) if d in sig_index_frame.index]
        model = ridge_fit(sig_index_frame.loc[tr_dates, ['snow_pct']].to_numpy(float),
                          np.array([tr_means[d] for d in tr_dates]))

        paths = {'7': os.path.join(args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}',
                                   f'resunetpp_v3_loyo{fold}_predictions.npz')}
        for s in ('42', '123'):
            paths[s] = os.path.join(args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                                    f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz')

        por_semilla, clim_res, cob = {}, None, None
        for s in SEEDS:
            if not os.path.exists(paths[s]):
                print(f'  [aviso] sin predicciones para la semilla {s}: {paths[s]}')
                continue
            print(f'  evaluando semilla {s}...', flush=True)
            res = evaluate_seed(paths[s], pos, clim, sig_index_frame.index, model)
            por_semilla[s] = res['red']
            if clim_res is None:
                clim_res, cob = res['clim'], res['cobertura']
        if not por_semilla:
            continue

        red = {k: float(np.mean([v[k] for v in por_semilla.values()]))
               for k in ('r2_raw', 'r2_oraculo', 'r2_regresor', 'corr')}
        c = clim_res
        print(f'  cobertura de la climatologia: {100 * cob:.1f}% de los pixeles de test')
        print(f"  {'':<24}{'tal cual':>10}{'nivel perfecto':>16}{'nivel regresor':>16}{'corr':>8}")
        print(f"  {'red (media ' + str(len(por_semilla)) + ' sem.)':<24}{red['r2_raw']:>10.3f}"
              f"{red['r2_oraculo']:>16.3f}{red['r2_regresor']:>16.3f}{red['corr']:>8.3f}")
        print(f"  {'climatologia':<24}{c['r2_raw']:>10.3f}{c['r2_oraculo']:>16.3f}"
              f"{c['r2_regresor']:>16.3f}{c['corr']:>8.3f}")
        gana = 'RED' if red['r2_oraculo'] > c['r2_oraculo'] else 'CLIMATOLOGIA'
        print(f'  comparacion principal (nivel perfecto): gana {gana}')
        veredictos.append((fold, red['r2_oraculo'], c['r2_oraculo'], gana))
        resultados[fold] = {'red_por_semilla': por_semilla, 'red_media': red,
                            'climatologia': c, 'cobertura': cob}

    print('=' * 76)
    print('RESUMEN - comparacion principal (nivel perfecto)')
    print(f"  {'fold':<8}{'red':>9}{'climatologia':>15}   gana")
    for f_, r_, c_, g_ in veredictos:
        print(f'  {f_:<8}{r_:>9.3f}{c_:>15.3f}   {g_}')
    n_red = sum(1 for v in veredictos if v[3] == 'RED')
    n_cli = len(veredictos) - n_red
    print()
    if n_red >= 3:
        print(f'  VEREDICTO: la red supera a la climatologia en {n_red} de {len(veredictos)} folds.')
        print('  El patron de la red aporta; la propuesta se sostiene tal cual.')
    elif n_cli >= 3:
        print(f'  VEREDICTO: la climatologia supera a la red en {n_cli} de {len(veredictos)} folds.')
        print('  Donde hay vuelos previos basta la climatologia; la contribucion')
        print('  de la red se desplaza a zonas no sobrevoladas (experimento C).')
    else:
        print(f'  VEREDICTO: no concluyente ({n_red} a {n_cli}). El experimento C pasa a ser decisivo.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
