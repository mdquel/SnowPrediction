r"""
Experimento B - ¿El patron de la red supera a la climatologia?
===============================================================

Pregunta
--------
La propuesta es obtener el PATRON de la red y la ESCALA de fuera. Antes
de invertir en ella hay que responder lo primero que preguntaria un
revisor de la literatura de repetibilidad de patrones de nieve: ¿para que
una red, si el patron se puede sacar promediando los mapas de anyos
anteriores?

Este script compara los dos "dibujos" en igualdad de condiciones:

    RED            prediccion del ResUNet++ (E2, semillas 7, 42 y 123)
    CLIMATOLOGIA   para cada pixel, la media de lo observado en ese mismo
                   sitio en los anyos de entrenamiento del fold

Diseno (fijado antes de mirar resultados)
-----------------------------------------
* La climatologia usa SOLO los anyos de entrenamiento del fold (ni el de
  validacion, 2024, ni el de test), igual que la red. Usa TODOS los tiles
  de esos anyos, no solo los submuestreados: tiene derecho a todos los
  mapas pasados.

* Ambos patrones reciben el mismo tratamiento en tres versiones:
      1. tal cual, sin corregir el nivel
      2. nivel perfecto: se desplaza cada fecha para que su media coincida
         con la media REAL (oraculo). Elimina el error de cantidad en los
         dos y mide SOLO la calidad del dibujo.
      3. nivel del regresor de E5 (solo cobertura, aditiva): desplegable.

* Se mide R2 (pixeles con nieve y dato valido, como siempre) y la
  correlacion media por tile. La correlacion no cambia al desplazar el
  nivel, asi que se reporta una sola vez.

* Donde la climatologia no existe (posiciones no sobrevoladas en los anyos
  de entrenamiento) se excluyen los pixeles para AMBOS, y se reporta la
  cobertura.

* Test restringido a fechas con senales, igual que en E5, para que las
  tres versiones se calculen sobre el mismo conjunto.

Criterio de decision (fijado antes)
-----------------------------------
Comparacion principal: version 2 (nivel perfecto), red = media de las tres
semillas.
    red > climatologia en >= 3 de 4 folds  ->  el patron de la red aporta
    climatologia > red en >= 3 de 4 folds  ->  donde hay vuelos previos
                                               basta la climatologia; la
                                               contribucion de la red se
                                               desplaza a zonas no
                                               sobrevoladas (experimento C)
    2 y 2                                  ->  no concluyente

Uso
---
    python scripts/climatology_baseline.py ^
        --seed7-dir E:\uniovi\mapunet\experimentos\E2_loyo_baseline ^
        --seeds-dir E:\uniovi\mapunet\experimentos\E2b_semillas
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Reutilizar las piezas ya probadas de E5
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recalibration_test import (date_of, mean_from_masks, ridge_fit,  # noqa: E402
                                ridge_predict, SNOW_THRESHOLD, MIN_VALID_PX)

VAL_YEAR = 2024
FOLDS = ['2021', '2022', '2023', '2025']
SEEDS = ['7', '42', '123']
FOLD_CSV = 'dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
SIGNALS_CSV = 'data/scale_signals.csv'


# ----------------------------------------------------------------------
def position_of(tile_id):
    """Clave de posicion: el tile_id sin la fecha."""
    return '_'.join(str(tile_id).split('_')[1:])


def build_climatology(df, fold, masks_dir):
    """Media por pixel de lo observado en los anyos de entrenamiento.

    Anyos de entrenamiento = todos salvo el de test y el de validacion.
    Se usan todos sus tiles (train y unused), no solo el submuestreo.
    """
    test_year = int(fold)
    sub = df[~df['year'].isin([test_year, VAL_YEAR])]
    sums, counts = {}, {}
    for t in sub['tile_id']:
        path = os.path.join(masks_dir, str(t))
        if not os.path.exists(path):
            continue
        m = np.load(path).astype(np.float64)
        valid = np.isfinite(m) & (m > -100)
        m = np.where(valid, m, 0.0)
        k = position_of(t)
        if k not in sums:
            sums[k] = np.zeros_like(m)
            counts[k] = np.zeros(m.shape, dtype=np.int32)
        sums[k] += m
        counts[k] += valid
    clim = {}
    for k in sums:
        c = counts[k]
        with np.errstate(invalid='ignore', divide='ignore'):
            clim[k] = np.where(c > 0, sums[k] / np.maximum(c, 1), np.nan)
    n_dates = sub['date'].nunique()
    return clim, n_dates


def load_pred(path, keep_dates):
    z = np.load(path)
    ids = z['tile_ids']
    dates = np.array([date_of(t) for t in ids])
    keep = np.isin(dates, list(keep_dates))
    return (ids[keep], np.asarray(z['preds'])[keep].astype(np.float64),
            np.asarray(z['targets'])[keep].astype(np.float64),
            (np.asarray(z['valids'])[keep] > 0.5) if 'valids' in z.files else None,
            dates[keep])


def evaluate_pattern(pred, tgt, mask, dates, target_means):
    """R2 tras desplazar cada fecha a una media objetivo (o sin desplazar).

    target_means: None -> tal cual; dict fecha -> media objetivo.
    """
    # medias del patron por fecha, sobre la mascara comun
    pm = {}
    for i, d in enumerate(dates):
        mi = mask[i]
        if mi.sum() < MIN_VALID_PX:
            continue
        s, n = pm.get(d, (0.0, 0))
        pm[d] = (s + pred[i][mi].sum(), n + int(mi.sum()))
    pm = {d: s / n for d, (s, n) in pm.items()}

    obs, sim, corrs = [], [], []
    for i, d in enumerate(dates):
        mi = mask[i]
        if mi.sum() < MIN_VALID_PX:
            continue
        p = pred[i][mi]
        o = tgt[i][mi]
        if o.std() > 1e-9 and p.std() > 1e-9:
            corrs.append(float(np.corrcoef(o, p)[0, 1]))
        if target_means is not None and d in target_means and d in pm:
            p = p + (target_means[d] - pm[d])
        obs.append(o)
        sim.append(p)
    o = np.concatenate(obs)
    s_ = np.concatenate(sim)
    r2 = 1.0 - np.sum((o - s_) ** 2) / np.sum((o - o.mean()) ** 2)
    return float(r2), float(np.mean(corrs)) if corrs else float('nan')


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Experimento B: red frente a climatologia')
    ap.add_argument('--seed7-dir', required=True,
                    help='carpeta de E2 (contiene loyo/resunetpp_v3_loyoYYYY/)')
    ap.add_argument('--seeds-dir', required=True,
                    help='carpeta de E2b (contiene loyo_seeds/)')
    ap.add_argument('--masks', default=MASKS_DIR)
    ap.add_argument('--signals', default=SIGNALS_CSV)
    ap.add_argument('--folds', nargs='+', default=FOLDS)
    ap.add_argument('--out', default='analysis/eB_climatology.json')
    args = ap.parse_args()

    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['fecha_key'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')
    sig = sig.set_index('fecha_key')

    out, veredictos = {}, []
    for fold in args.folds:
        print('=' * 72)
        print(f'Fold {fold}')
        df = pd.read_csv(FOLD_CSV.format(fold=fold))

        print('  construyendo climatologia (lee todos los mapas de train)...', flush=True)
        clim, n_dates = build_climatology(df, fold, args.masks)
        print(f'  climatologia: {len(clim)} posiciones, {n_dates} fechas de train')

        # regresor de E5 (solo cobertura), ajustado con las fechas de train
        train_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
        tr_means = mean_from_masks(train_ids, args.masks)
        tr_dates = [d for d in sorted(tr_means) if d in sig.index]
        model = ridge_fit(sig.loc[tr_dates, ['snow_pct']].to_numpy(float),
                          np.array([tr_means[d] for d in tr_dates]))

        paths = {'7': glob.glob(os.path.join(
            args.seed7_dir, 'loyo', f'resunetpp_v3_loyo{fold}', f'resunetpp_v3_loyo{fold}_predictions.npz'))}
        for s in ['42', '123']:
            paths[s] = glob.glob(os.path.join(
                args.seeds_dir, 'loyo_seeds', f'resunetpp_v3_loyo{fold}_s{s}',
                f'resunetpp_v3_loyo{fold}_s{s}_predictions.npz'))

        res = {'red': {}, 'clim': None}
        for s in SEEDS:
            if not paths[s]:
                print(f'  [aviso] sin predicciones para la semilla {s}')
                continue
            ids, pred, tgt, val, dates = load_pred(paths[s][0], sig.index)

            # patron climatologico para los mismos tiles
            cpred = np.full_like(pred, np.nan)
            for i, t in enumerate(ids):
                k = position_of(t)
                if k in clim:
                    cpred[i] = clim[k]

            # mascara comun: nieve + dato valido + climatologia existente
            mask = tgt > SNOW_THRESHOLD
            if val is not None:
                mask &= val
            base_mask = mask.copy()
            mask &= np.isfinite(cpred)
            cobertura = mask.sum() / max(base_mask.sum(), 1)

            # medias reales por fecha (oraculo) sobre la mascara comun
            real = {}
            for i, d in enumerate(dates):
                mi = mask[i]
                if mi.sum() < MIN_VALID_PX:
                    continue
                a, n = real.get(d, (0.0, 0))
                real[d] = (a + tgt[i][mi].sum(), n + int(mi.sum()))
            real = {d: a / n for d, (a, n) in real.items()}

            test_dates = sorted(real)
            regr = {d: float(v) for d, v in zip(
                test_dates, ridge_predict(model, sig.loc[test_dates, ['snow_pct']].to_numpy(float)))}

            fila = {}
            for nombre, p in [('red', pred), ('clim', np.nan_to_num(cpred))]:
                r_raw, corr = evaluate_pattern(p, tgt, mask, dates, None)
                r_orac, _ = evaluate_pattern(p, tgt, mask, dates, real)
                r_regr, _ = evaluate_pattern(p, tgt, mask, dates, regr)
                fila[nombre] = dict(r2_raw=r_raw, r2_oraculo=r_orac,
                                    r2_regresor=r_regr, corr=corr)

            res['red'][s] = fila['red']
            if res['clim'] is None:           # la climatologia no depende de la semilla
                res['clim'] = fila['clim']
                res['cobertura'] = float(cobertura)

        if not res['red']:
            continue
        red_m = {k: float(np.mean([v[k] for v in res['red'].values()]))
                 for k in ['r2_raw', 'r2_oraculo', 'r2_regresor', 'corr']}
        c = res['clim']
        print(f"  cobertura de la climatologia: {100*res['cobertura']:.1f}% de los pixeles de test")
        print(f"  {'':<22}{'tal cual':>10}{'nivel perfecto':>16}{'nivel regresor':>16}{'corr':>8}")
        print(f"  {'red (media 3 sem.)':<22}{red_m['r2_raw']:>10.3f}{red_m['r2_oraculo']:>16.3f}"
              f"{red_m['r2_regresor']:>16.3f}{red_m['corr']:>8.3f}")
        print(f"  {'climatologia':<22}{c['r2_raw']:>10.3f}{c['r2_oraculo']:>16.3f}"
              f"{c['r2_regresor']:>16.3f}{c['corr']:>8.3f}")
        gana = 'RED' if red_m['r2_oraculo'] > c['r2_oraculo'] else 'CLIMATOLOGIA'
        print(f"  comparacion principal (nivel perfecto): gana {gana}")
        veredictos.append((fold, red_m['r2_oraculo'], c['r2_oraculo'], gana))
        res['red_media'] = red_m
        out[fold] = res

    print('=' * 72)
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
        json.dump(out, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
