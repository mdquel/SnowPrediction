r"""
E5 - Recalibracion en dos etapas.
=================================

Que se prueba
-------------
E4 y E4b metieron las senales de escala DENTRO de la red, en una cabeza
entrenada end-to-end, y no funciono: con datos suficientes las senales
llegan a perjudicar (-0.082 de R2, rangos sin solapar).

Pero el analisis de senales ya habia medido que la informacion de
magnitud SI existe fuera del terreno: la cobertura de nieve correlaciona
+0.678 con la profundidad media de la fecha, y los grados-dia -0.69.

La hipotesis de este experimento es que el problema no era la informacion
sino la via de entrada: pedirle a la red que aprenda a la vez a usar las
senales y a no estropear el patron, con 2284 tiles y 23 fechas, es mucho
con tan pocos datos.

La alternativa desacopla las dos tareas en etapas independientes:

    ResUNet++ (ya entrenado, no se toca)  ->  el PATRON
    regresor pequeno (nuevo)              ->  la MEDIA de la fecha
    mapa_corregido = f(mapa, media_predicha)

El techo ya esta medido: la recalibracion oraculo, que usa la media REAL,
lleva el R2 medio de 0.303 a 0.437. La pregunta es cuanto de esos 0.134
se captura con una media PREDICHA a partir de senales disponibles sin
volar.

Diseno
------
Tres filas por comparar:

    1. sin recalibrar       el modelo tal cual
    2. media predicha       DESPLEGABLE: solo senales externas
    3. media real           ORACULO: cota superior, no desplegable

Y dos formas de corregir, porque el sesgo y la dispersion no responden
igual:

    multiplicativa:  mapa * (media_objetivo / media_del_mapa)
    aditiva:         mapa + (media_objetivo - media_del_mapa)

Control de fuga
---------------
El regresor se ajusta UNICAMENTE con las fechas de train del fold y se
evalua en las de test. Ajustarlo con todas las fechas seria fuga y el
resultado no valdria: es el mismo error que se corrigio en SCALE_NORM.

Se prueban varios subconjuntos de senales porque con ~18 fechas de
entrenamiento un modelo con cuatro variables puede sobreajustar. Se
reportan todos en lugar de elegir uno, para no seleccionar a posteriori.

Se reporta ademas el MAE del propio regresor al predecir la media en las
fechas de test: es el eslabon que limita todo lo demas, porque si la
media predicha es mala la recalibracion no puede funcionar por bien
montada que este.

Uso
---
    python scripts/recalibration_test.py
    python scripts/recalibration_test.py --fold 2025 --arm base
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SNOW_THRESHOLD = 0.01     # m, mismo criterio snow-only que la evaluacion
MIN_VALID_PX = 10

SIGNALS_CSV = 'data/scale_signals.csv'
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
FOLD_CSV_TPL = 'dataset_v4_ms_sx200/loyo_e4b/dataset_e4b_loyo{fold}.csv'

# Subconjuntos de senales a probar. Se reportan todos.
SIGNAL_SETS = {
    'cobertura':        ['snow_pct'],
    'cobertura+pdd30':  ['snow_pct', 'pdd_30d'],
    'cobertura+pdd+pp': ['snow_pct', 'pdd_30d', 'ppAcc_mm'],
    'las_cuatro':       ['snow_pct', 'pdd_15d', 'pdd_30d', 'ppAcc_mm'],
}


# ----------------------------------------------------------------------
def date_of(tile_id):
    return str(tile_id).split('_')[0]


def mean_from_masks(tile_ids, masks_dir, max_per_date=60):
    """Media de HS por fecha, calculada desde los .npy originales.

    Mismo criterio que la evaluacion: solo pixeles con nieve y dato
    valido. Se usa esto en lugar del Excel de referencia porque garantiza
    que la media esta tomada sobre exactamente los mismos pixeles que
    entran en las metricas, y porque el Excel no cubre 2025.
    """
    # Muestrear como mucho max_per_date tiles por fecha: la media por
    # fecha se estabiliza mucho antes de leerlos todos, y leer 2284
    # ficheros sueltos domina el tiempo de ejecucion.
    por_fecha = {}
    for t in tile_ids:
        por_fecha.setdefault(date_of(t), []).append(t)
    sel = []
    for d, ts in por_fecha.items():
        sel.extend(ts[:max_per_date])

    acc = {}
    for t in sel:
        path = os.path.join(masks_dir, str(t))
        if not os.path.exists(path):
            continue
        m = np.load(path).ravel()
        m = m[np.isfinite(m) & (m > SNOW_THRESHOLD)]
        if m.size < MIN_VALID_PX:
            continue
        d = date_of(t)
        s, n = acc.get(d, (0.0, 0))
        acc[d] = (s + float(m.sum()), n + int(m.size))
    return {d: s / n for d, (s, n) in acc.items() if n > 0}


def mean_from_npz(npz, key='targets'):
    """Media observada o predicha por fecha, desde un .npz de predicciones.

    Los arrays se materializan UNA sola vez. Indexar npz['targets'][i]
    dentro de un bucle descomprime el array completo en cada iteracion,
    lo que con 220 tiles hace el calculo inviable.
    """
    ids = npz['tile_ids']
    arr = np.asarray(npz[key])
    tgt = arr if key == 'targets' else np.asarray(npz['targets'])
    val = np.asarray(npz['valids']) if 'valids' in npz.files else None

    acc = {}
    for i, t in enumerate(ids):
        m = tgt[i] > SNOW_THRESHOLD
        if val is not None:
            m &= (val[i] > 0.5)
        if m.sum() < MIN_VALID_PX:
            continue
        d = date_of(t)
        s_, n_ = acc.get(d, (0.0, 0))
        acc[d] = (s_ + float(arr[i][m].sum()), n_ + int(m.sum()))
    return {d: s_ / n_ for d, (s_, n_) in acc.items() if n_ > 0}


def ridge_fit(X, y, alpha=1.0):
    """Ridge con estandarizacion, implementado a mano.

    Con 18 muestras y hasta 4 variables no hace falta mas, y evita una
    dependencia de sklearn para algo tan simple.
    """
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    n, p = Z.shape
    A = Z.T @ Z + alpha * np.eye(p)
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return dict(w=w, b=y.mean(), mu=mu, sd=sd)


def ridge_predict(model, X):
    Z = (X - model['mu']) / model['sd']
    return Z @ model['w'] + model['b']


def r2(obs, sim):
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def load_npz(path):
    """Materializa un .npz de predicciones una sola vez.

    Descomprimir dentro de los bucles de evaluacion multiplicaba el
    trabajo por el numero de configuraciones probadas.
    """
    z = np.load(path)
    d = {'tile_ids': z['tile_ids'],
         'preds': np.asarray(z['preds'], dtype=np.float64),
         'targets': np.asarray(z['targets'], dtype=np.float64),
         'valids': np.asarray(z['valids']) if 'valids' in z.files else None}
    # mascara de pixeles utilizables por tile, calculada una vez
    m = d['targets'] > SNOW_THRESHOLD
    if d['valids'] is not None:
        m &= (d['valids'] > 0.5)
    d['mask'] = m
    d['dates'] = np.array([date_of(t) for t in d['tile_ids']])
    return d


def means_by_date(data, key):
    """Media por fecha de 'preds' o 'targets', sobre pixeles utilizables."""
    acc = {}
    arr, m, dates = data[key], data['mask'], data['dates']
    for i in range(len(dates)):
        mi = m[i]
        if mi.sum() < MIN_VALID_PX:
            continue
        s_, n_ = acc.get(dates[i], (0.0, 0))
        acc[dates[i]] = (s_ + float(arr[i][mi].sum()), n_ + int(mi.sum()))
    return {d: s_ / n_ for d, (s_, n_) in acc.items() if n_ > 0}


def evaluate(data, pred_means, target_means, mode):
    """R2 tras recalibrar cada tile segun la media objetivo de su fecha.

    mode: 'none' | 'mult' | 'add'
    """
    obs_all, sim_all = [], []
    for i in range(len(data['dates'])):
        mi = data['mask'][i]
        if mi.sum() < MIN_VALID_PX:
            continue
        p = data['preds'][i][mi]
        if mode != 'none':
            d = data['dates'][i]
            tm, pm = target_means.get(d), pred_means.get(d)
            if tm is not None and pm is not None:
                if mode == 'mult' and abs(pm) > 1e-9:
                    p = p * (tm / pm)
                elif mode == 'add':
                    p = p + (tm - pm)
        obs_all.append(data['targets'][i][mi])
        sim_all.append(p)
    return r2(np.concatenate(obs_all), np.concatenate(sim_all))


# ----------------------------------------------------------------------
def run_fold(fold, arm, npz_glob, signals, masks_dir, alpha,
             fold_csv_tpl=FOLD_CSV_TPL):
    """Procesa un fold: ajusta el regresor con train y evalua en test."""
    paths = [p for p in sorted(glob.glob(npz_glob)) if '_last' not in os.path.basename(p)]
    if not paths:
        print(f'  [aviso] sin .npz para {npz_glob}')
        return None

    # --- fechas de train del fold, desde su CSV ---
    fold_csv = fold_csv_tpl.format(fold=fold)
    if not os.path.exists(fold_csv):
        raise SystemExit(f'No se encuentra {fold_csv}')
    df = pd.read_csv(fold_csv)
    train_ids = df.loc[df['exp_temporal_split'] == 'train', 'tile_id'].tolist()
    train_dates = sorted({date_of(t) for t in train_ids})

    # --- medias reales de train, desde los .npy ---
    train_means = mean_from_masks(train_ids, masks_dir)
    train_dates = [d for d in train_dates if d in train_means]

    sig = signals.set_index('fecha_key')
    train_dates = [d for d in train_dates if d in sig.index]
    if len(train_dates) < 5:
        print(f'  [aviso] solo {len(train_dates)} fechas de train utilizables')
        return None

    y_tr = np.array([train_means[d] for d in train_dates])

    results = []
    for seed_path in paths:
        name = os.path.basename(seed_path).replace('_predictions.npz', '')
        print(f'  procesando {name} ...', flush=True)
        data = load_npz(seed_path)

        # Restringir el test a fechas con senales. En E2 el dataset tiene
        # 27 fechas pero solo 23 tienen las cuatro senales; las filas
        # sin recalibrar, recalibrada y oraculo se calculan todas sobre
        # el MISMO subconjunto para que sean comparables.
        keep = np.isin(data['dates'], list(sig.index))
        n_excl = int((~keep).sum())
        for k in ('tile_ids', 'preds', 'targets', 'mask', 'dates'):
            data[k] = data[k][keep]
        if data['valids'] is not None:
            data['valids'] = data['valids'][keep]
        if n_excl:
            print(f'    ({n_excl} tiles de test excluidos por fecha sin senales)')

        test_means_real = means_by_date(data, 'targets')
        pred_means_raw = means_by_date(data, 'preds')
        test_dates = [d for d in sorted(test_means_real) if d in sig.index]

        # Error de la RED al estimar la media de cada fecha: misma cantidad
        # que estima el regresor, para que el criterio compare igual con igual.
        mae_red = float(np.mean([abs(pred_means_raw[d] - test_means_real[d])
                                 for d in test_dates if d in pred_means_raw]))
        row = {'experimento': name,
               'n_train_fechas': len(train_dates),
               'n_test_fechas': len(test_dates),
               'mae_media_red': mae_red,
               'r2_sin_recal': evaluate(data, pred_means_raw, {}, 'none'),
               'r2_oraculo_mult': evaluate(data, pred_means_raw, test_means_real, 'mult'),
               'r2_oraculo_add': evaluate(data, pred_means_raw, test_means_real, 'add')}

        for set_name, cols in SIGNAL_SETS.items():
            X_tr = sig.loc[train_dates, cols].to_numpy(dtype=np.float64)
            X_te = sig.loc[test_dates, cols].to_numpy(dtype=np.float64)

            model = ridge_fit(X_tr, y_tr, alpha=alpha)
            y_hat = ridge_predict(model, X_te)
            pred_means = {d: float(v) for d, v in zip(test_dates, y_hat)}

            y_true = np.array([test_means_real[d] for d in test_dates])
            row[f'mae_media_{set_name}'] = float(np.mean(np.abs(y_hat - y_true)))
            row[f'r2_pred_mult_{set_name}'] = evaluate(data, pred_means_raw, pred_means, 'mult')
            row[f'r2_pred_add_{set_name}'] = evaluate(data, pred_means_raw, pred_means, 'add')

        results.append(row)
    return results


PRESETS = {
    # E4b: tres semillas, fold 2025, subconjunto de 23 fechas
    'e4b': dict(
        npz=os.path.join('loyo_e4b_{arm}', 'e4b_{arm}_loyo{fold}_s*', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo_e4b/dataset_e4b_loyo{fold}.csv'),
    # E2: ResUNet++ estandar lambda=0, una semilla, CUATRO folds
    'e2': dict(
        npz=os.path.join('loyo', 'resunetpp_v3_loyo{fold}', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'),
    # E2 con semillas 42 y 123: misma particion, resultados en loyo_seeds
    'e2seeds': dict(
        npz=os.path.join('loyo_seeds', 'resunetpp_v3_loyo{fold}_s*', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'),
}


def main():
    ap = argparse.ArgumentParser(description='Recalibracion en dos etapas')
    ap.add_argument('--preset', default='e4b', choices=list(PRESETS))
    ap.add_argument('--folds', nargs='+', default=['2025'])
    ap.add_argument('--arm', default='base',
                    choices=['base', 'dual', 'dualscale'])
    ap.add_argument('--results-dir', default='results/norm_v3')
    ap.add_argument('--masks', default=MASKS_DIR)
    ap.add_argument('--signals', default=SIGNALS_CSV)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--out', default='analysis/e5_recalibration.json')
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    sig = pd.read_csv(args.signals)
    sig = sig[sig['admisible'] == True].copy()
    sig['fecha_key'] = pd.to_datetime(sig['fecha']).dt.strftime('%Y%m%d')

    all_out = {}
    criterio = []   # (experimento, prediccion_correcta)
    resumen = []
    for fold in args.folds:
        pattern = os.path.join(args.results_dir,
                               preset['npz'].format(arm=args.arm, fold=fold))
        paths = [q for q in sorted(glob.glob(pattern))
                 if '_last' not in os.path.basename(q)]
        print('=' * 68)
        print(f'Fold {fold} ({args.preset}, brazo {args.arm}): {len(paths)} modelo(s)')
        if not paths:
            print(f'  sin resultados en {pattern}')
            continue

        rows = run_fold(fold, args.arm, pattern, sig, args.masks,
                        args.alpha, fold_csv_tpl=preset['fold_csv'])
        if not rows:
            continue
        all_out[fold] = rows

        print(f"Fechas de train con media: {rows[0]['n_train_fechas']}")
        print(f"Fechas de test con senal : {rows[0]['n_test_fechas']}")
        print()
        print('MAE del regresor al predecir la media de test (m):')
        for s_ in SIGNAL_SETS:
            v = np.mean([r[f'mae_media_{s_}'] for r in rows])
            print(f'  {s_:<22}{v:>10.4f}')
        print()

        base = np.mean([r['r2_sin_recal'] for r in rows])
        orac_m = np.mean([r['r2_oraculo_mult'] for r in rows])
        orac_a = np.mean([r['r2_oraculo_add'] for r in rows])
        techo = max(orac_m, orac_a) - base
        print(f"  {'configuracion':<34}{'mult':>9}{'add':>9}{'% techo':>10}")
        print(f"  {'sin recalibrar':<34}{base:>9.4f}{base:>9.4f}{'-':>10}")
        for s_ in SIGNAL_SETS:
            m = np.mean([r[f'r2_pred_mult_{s_}'] for r in rows])
            a = np.mean([r[f'r2_pred_add_{s_}'] for r in rows])
            pct = 100 * (max(m, a) - base) / techo if techo > 1e-9 else float('nan')
            print(f'  {"media predicha: " + s_:<34}{m:>9.4f}{a:>9.4f}{pct:>9.0f}%')
        print(f"  {'media REAL (oraculo)':<34}{orac_m:>9.4f}{orac_a:>9.4f}{100:>9.0f}%")
        print()

        # Criterio de E5 aplicado modelo a modelo
        print('  Criterio: recalibrar ayuda si MAE regresor < MAE red')
        print(f"  {'modelo':<32}{'MAE regr':>9}{'MAE red':>9}"
              f"{'predice':>10}{'ganancia':>10}{'acierta':>9}")
        for r in rows:
            mr, mn = r['mae_media_cobertura'], r['mae_media_red']
            g = r['r2_pred_add_cobertura'] - r['r2_sin_recal']
            predice_ayuda = mr < mn
            ok = (predice_ayuda and g > 0) or (not predice_ayuda and g <= 0)
            criterio.append((r['experimento'], ok))
            print(f"  {r['experimento']:<32}{mr:>9.3f}{mn:>9.3f}"
                  f"{'ayuda' if predice_ayuda else 'empeora':>10}"
                  f"{g:>+10.3f}{'SI' if ok else 'NO':>9}")
        print()

        # configuracion PRINCIPAL fijada a priori: cobertura, aditiva
        pa = np.mean([r['r2_pred_add_cobertura'] for r in rows])
        resumen.append((fold, base, pa, orac_a))

    if len(resumen) > 1:
        print('=' * 68)
        print('RESUMEN - configuracion principal fijada a priori')
        print('(solo cobertura, correccion aditiva)')
        print(f"  {'fold':<8}{'sin recal':>11}{'recalibrado':>13}{'oraculo':>10}{'ganancia':>10}")
        for f_, b_, p_, o_ in resumen:
            print(f'  {f_:<8}{b_:>11.3f}{p_:>13.3f}{o_:>10.3f}{p_-b_:>+10.3f}')
        mb = np.mean([x[1] for x in resumen])
        mp = np.mean([x[2] for x in resumen])
        mo = np.mean([x[3] for x in resumen])
        print(f"  {'MEDIA':<8}{mb:>11.3f}{mp:>13.3f}{mo:>10.3f}{mp-mb:>+10.3f}")
        n_up = sum(1 for x in resumen if x[2] > x[1])
        print(f'  folds que mejoran: {n_up} de {len(resumen)}')

    if criterio:
        n_ok = sum(1 for _, ok in criterio if ok)
        print()
        print(f'CRITERIO DE E5: acierta en {n_ok} de {len(criterio)} modelos')
        fallos = [e for e, ok in criterio if not ok]
        if fallos:
            print('  falla en: ' + ', '.join(fallos))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'preset': args.preset, 'arm': args.arm, 'alpha': args.alpha,
                   'principal': 'cobertura, aditiva (fijada a priori)',
                   'folds': all_out}, f, indent=2)
    print()
    print(f'Guardado en: {args.out}')


if __name__ == '__main__':
    main()
