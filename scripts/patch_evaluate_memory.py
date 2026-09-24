r"""
Parche de memoria para training/evaluate.py
===========================================

Problema
--------
La evaluacion acumula los pixeles de todos los tiles con
    lista.extend(array.tolist())
.tolist() convierte cada pixel en un objeto float de Python (~32 bytes)
en lugar de 4 bytes de float32. En el fold 2022 (1802 tiles de test, del
orden de 1e8 pixeles validos, cuatro listas) eso supera por si solo los
13 GB, y con las copias posteriores rebasa el limite de 21 GB del
contenedor: el proceso muere con "Killed" al terminar la inferencia.

Arreglo
-------
Guardar los trozos como arrays de numpy y unirlos una sola vez al final.
Memoria: de >13 GB a <2 GB.

Por que los resultados NO cambian
---------------------------------
array_float32.tolist() produce floats de Python cuyo valor es exactamente
el float32 promovido a float64. np.concatenate(...).astype(np.float64)
produce exactamente esos mismos valores, en el mismo orden. Las metricas
deben salir identicas a las de antes; el script de verificacion lo
comprueba sobre un fold ya evaluado.

Uso
---
    python scripts/patch_evaluate_memory.py            # aplica
    python scripts/patch_evaluate_memory.py --check    # solo comprueba

El script exige que cada patron aparezca EXACTAMENTE una vez. Si alguno no
aparece o aparece repetido, no toca nada y lo avisa.
"""

import argparse
import sys

TARGET = 'training/evaluate.py'

HELPER = '''

def _cat64(chunks):
    """Une trozos de arrays y promueve a float64.

    Equivale exactamente a np.array(lista_de_floats_de_python) construida
    con .tolist(), pero sin crear un objeto Python por pixel.
    """
    if not chunks:
        return np.array([])
    return np.concatenate(chunks).astype(np.float64)


def _cat_like_tolist(chunks):
    """Como _cat64, pero respetando el tipo que habria dado np.array(tolist()).

    tolist() de floats da float64; de enteros, int64; de booleanos, bool.
    """
    if not chunks:
        return np.array([])
    arr = np.concatenate(chunks)
    if arr.dtype.kind == 'f':
        return arr.astype(np.float64)
    if arr.dtype.kind in 'iu':
        return arr.astype(np.int64)
    return arr
'''

REPLACEMENTS = [
    # acumulacion snow-only
    ('all_preds.extend(out_flat[snow].tolist())',
     'all_preds.append(out_flat[snow])'),
    ('all_targets.extend(tgt_flat[snow].tolist())',
     'all_targets.append(tgt_flat[snow])'),
    # acumulacion dominio completo
    ('all_preds_full.extend(out_flat[vflat].tolist())',
     'all_preds_full.append(out_flat[vflat])'),
    ('all_targets_full.extend(tgt_flat[vflat].tolist())',
     'all_targets_full.append(tgt_flat[vflat])'),
    # conversion final
    ('y_pred = np.array(all_preds)',
     'y_pred = _cat64(all_preds)'),
    ('y_true = np.array(all_targets)',
     'y_true = _cat64(all_targets)'),
    ('yf_true = np.array(all_targets_full)',
     'yf_true = _cat64(all_targets_full)'),
    ('yf_pred = np.array(all_preds_full)',
     'yf_pred = _cat64(all_preds_full)'),
    # benchmark ingenuo
    ('values.extend(valid.tolist())',
     'values.append(valid)'),
    ('return np.array(values)',
     'return _cat_like_tolist(values)'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='solo comprobar, no escribir')
    ap.add_argument('--file', default=TARGET)
    args = ap.parse_args()

    src = open(args.file, encoding='utf-8').read()

    if '_cat64' in src:
        print('El parche ya parece aplicado (existe _cat64). No se toca nada.')
        return

    problems = []
    for old, _ in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            problems.append(f'  "{old}"  aparece {n} veces (se esperaba 1)')
    if 'import numpy as np' not in src:
        problems.append('  no se encuentra "import numpy as np"')

    if problems:
        print('NO se aplica el parche. Patrones problematicos:')
        print('\n'.join(problems))
        sys.exit(1)

    print('Todos los patrones encontrados exactamente una vez.')
    if args.check:
        return

    for old, new in REPLACEMENTS:
        src = src.replace(old, new)

    # insertar los helpers justo despues del import de numpy
    src = src.replace('import numpy as np', 'import numpy as np' + HELPER, 1)

    open(args.file, 'w', encoding='utf-8').write(src)
    print(f'Parche aplicado a {args.file}: {len(REPLACEMENTS)} sustituciones.')


if __name__ == '__main__':
    main()
