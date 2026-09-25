r"""
Parche para scripts/recalibration_test.py: E5 sobre las semillas de E2
======================================================================

Que anade
---------
1. Un preset 'e2seeds' que busca los resultados de las semillas 42 y 123
   en results/norm_v3/loyo_seeds/, con los mismos CSV de folds que E2.

2. El error de la RED al estimar la media de cada fecha (mae_media_red),
   medido igual que el del regresor: media de |media_predicha -
   media_real| sobre las fechas de test.

   Hasta ahora el criterio de E5 se comparo contra el Bias global de
   metrics.json, que es un sesgo con signo sobre pixeles. La comparacion
   correcta es contra el error de la red en la MISMA cantidad que estima
   el regresor: la media por fecha. Asi las dos cifras son directamente
   comparables.

3. Una tabla por modelo que aplica el criterio:
       el regresor deberia ayudar  <=>  MAE regresor < MAE red
   y comprueba si lo que predice coincide con lo que ocurre. Al final,
   un recuento global de aciertos.

Uso
---
    python scripts/patch_recalibration_seeds.py --check
    python scripts/patch_recalibration_seeds.py
"""

import argparse
import sys

TARGET = 'scripts/recalibration_test.py'

REPLACEMENTS = [
    # 1) preset nuevo
    ("""    'e2': dict(
        npz=os.path.join('loyo', 'resunetpp_v3_loyo{fold}', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'),
}""",
     """    'e2': dict(
        npz=os.path.join('loyo', 'resunetpp_v3_loyo{fold}', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'),
    # E2 con semillas 42 y 123: misma particion, resultados en loyo_seeds
    'e2seeds': dict(
        npz=os.path.join('loyo_seeds', 'resunetpp_v3_loyo{fold}_s*', '*_predictions.npz'),
        fold_csv='dataset_v4_ms_sx200/loyo/dataset_loyo{fold}.csv'),
}"""),

    # 2) error de la red al estimar la media por fecha
    ("""        row = {'experimento': name,
               'n_train_fechas': len(train_dates),
               'n_test_fechas': len(test_dates),""",
     """        # Error de la RED al estimar la media de cada fecha: misma cantidad
        # que estima el regresor, para que el criterio compare igual con igual.
        mae_red = float(np.mean([abs(pred_means_raw[d] - test_means_real[d])
                                 for d in test_dates if d in pred_means_raw]))
        row = {'experimento': name,
               'n_train_fechas': len(train_dates),
               'n_test_fechas': len(test_dates),
               'mae_media_red': mae_red,"""),

    # 3a) acumulador global
    ("""    all_out = {}""",
     """    all_out = {}
    criterio = []   # (experimento, prediccion_correcta)"""),

    # 3b) tabla por modelo, antes del resumen del fold
    ("""        # configuracion PRINCIPAL fijada a priori: cobertura, aditiva""",
     """        # Criterio de E5 aplicado modelo a modelo
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

        # configuracion PRINCIPAL fijada a priori: cobertura, aditiva"""),

    # 3c) recuento global
    ("""    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)""",
     """    if criterio:
        n_ok = sum(1 for _, ok in criterio if ok)
        print()
        print(f'CRITERIO DE E5: acierta en {n_ok} de {len(criterio)} modelos')
        fallos = [e for e, ok in criterio if not ok]
        if fallos:
            print('  falla en: ' + ', '.join(fallos))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--file', default=TARGET)
    args = ap.parse_args()

    src = open(args.file, encoding='utf-8').read()
    if "'e2seeds'" in src:
        print('El parche ya parece aplicado (existe e2seeds). No se toca nada.')
        return

    problems = []
    for old, _ in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            problems.append(f'  patron que empieza por "{old.strip()[:50]}..." aparece {n} veces')
    if problems:
        print('NO se aplica el parche:')
        print('\n'.join(problems))
        sys.exit(1)

    print('Todos los patrones encontrados exactamente una vez.')
    if args.check:
        return
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)
    open(args.file, 'w', encoding='utf-8').write(src)
    print(f'Parche aplicado a {args.file}.')


if __name__ == '__main__':
    main()
