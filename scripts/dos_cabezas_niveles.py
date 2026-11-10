r"""
Red de dos cabezas: tabla de niveles por fecha y semilla (sin GPU)
==================================================================

Protocolo 2026, apartado 8.13. La red de dos cabezas recibe dos escalares por
fecha: el nivel L(d) y la media de la climatologia Cm(d). Este script escribe
una tabla por semilla con esos valores para todas las fechas:

    entrenamiento y validacion
        nivel_m = nivel_last_m = nivel observado de la fecha
        clim_media_m           = media de la climatologia (clim_v2)
        (los dos salen de analysis/niveles_por_fecha_clim_v2.csv, que escribe
        data/patch_clim_v2.py)

    test
        nivel_m       media de la prediccion de la red B, mejor epoca
        nivel_last_m  media de la prediccion de la red B, ultima epoca
        clim_media_m  media de la climatologia
        calculados EXACTAMENTE como en la combinacion (hybrid, de
        scripts/clim_v1_hybrid.py): sobre todos los pixeles volados de la
        fecha, con la prediccion de B tal cual sale de la red y la
        climatologia a 0 donde no la hay. La red B es la de la misma semilla.

La red de dos cabezas se evalua con la mejor epoca usando nivel_m y con la
ultima usando nivel_last_m, igual que la combinacion de cada lectura.

El nivel observado de las fechas de test no se lee ni se escribe.

Controles (si alguno falla, se detiene sin escribir nada)
---------------------------------------------------------
1. Con los niveles de la tabla, max(0, C + L - Cm) es identico a la salida de
   hybrid() para cada semilla y epoca: la red arrancara en la combinacion.
2. La media de la climatologia de las fechas de test coincide con la de la
   tabla de clim_v2 (dos calculos independientes: predicciones guardadas y
   mosaico), con tolerancia 1e-4 m.
3. Las fechas de test de la tabla de clim_v2 son las de las predicciones.

Salida
------
    analysis/dos_cabezas/niveles_s<semilla>.csv
    analysis/dos_cabezas/niveles_resumen.json

Uso (desde la raiz del repo)
----------------------------
    python scripts/dos_cabezas_niveles.py
    python scripts/dos_cabezas_niveles.py --b-dir results/regen2024/sat_v2 \
        --clim-npz results/regen2024/clim_v1/climatologia_sola/climatologia_sola_predictions.npz
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clim_v1_hybrid import hybrid  # noqa: E402
from clim_v1_level_corrected import date_of, load  # noqa: E402

TOL_CLIM = 1e-4      # m, control 2
EPOCAS = (('nivel_m', ''), ('nivel_last_m', '_last'))
CAMPOS = ['date', 'split', 'nivel_m', 'nivel_last_m', 'clim_media_m', 'origen_nivel']


def niveles_de_la_red(c_ids, c_p, n_ids, n_p, valid):
    """L(d) y Cm(d) por fecha, con las mismas operaciones que hybrid()."""
    idx = {t: i for i, t in enumerate(n_ids)}
    n_p = n_p[np.array([idx[t] for t in c_ids])]
    dates = np.array([date_of(t) for t in c_ids])
    out = {}
    for d in np.unique(dates):
        k = dates == d
        v = valid[k]
        out[str(d)] = (float(n_p[k][v].mean()), float(c_p[k][v].mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description='Niveles por fecha para la red de dos cabezas')
    ap.add_argument('--niveles-clim', default='analysis/niveles_por_fecha_clim_v2.csv')
    ap.add_argument('--b-dir', default='results/regen2024/sat_v2')
    ap.add_argument('--clim-npz',
                    default='results/regen2024/clim_v1/climatologia_sola/climatologia_sola_predictions.npz')
    ap.add_argument('--out-dir', default='analysis/dos_cabezas')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 42, 123])
    a = ap.parse_args()

    with open(a.niveles_clim, encoding='utf-8') as fh:
        tabla = list(csv.DictReader(fh))
    if not tabla:
        raise SystemExit(f'STOP: {a.niveles_clim} esta vacio')
    test_tabla = {r['date']: float(r['clim_media_m']) for r in tabla if r['split'] == 'test'}

    c_ids, c_p, c_o, c_v = load(a.clim_npz)
    valid = c_v & np.isfinite(c_o)
    dates = np.array([date_of(t) for t in c_ids])
    fechas_test = sorted(str(d) for d in np.unique(dates))

    fallos = []
    if sorted(test_tabla) != fechas_test:
        fallos.append(f'control 3: fechas de test de {a.niveles_clim} {sorted(test_tabla)} '
                      f'frente a las de las predicciones {fechas_test}')

    resumen = {'descripcion': ('Niveles por fecha para la red de dos cabezas (protocolo 2026, 8.13). '
                               'Test: media de la prediccion de la red B y de la climatologia sobre '
                               'todos los pixeles volados, como en la combinacion.'),
               'fuentes': {'niveles_clim_v2': a.niveles_clim, 'b_dir': a.b_dir, 'clim_npz': a.clim_npz},
               'test_por_semilla': {}, 'controles': {}}
    tablas, dif_clim_max, dif_hib_max = {}, 0.0, 0.0

    for s in a.seeds:
        por_epoca = {}
        for col, e in EPOCAS:
            name = f'resunetpp_sat2_B_s{s}'
            npz = os.path.join(a.b_dir, name, f'{name}{e}_predictions.npz')
            if not os.path.exists(npz):
                fallos.append(f'falta {npz}')
                continue
            n_ids, n_p, _, _ = load(npz)
            if sorted(n_ids) != sorted(c_ids):
                fallos.append(f'{npz}: sus teselas no son las de la climatologia sola')
                continue
            niv = niveles_de_la_red(c_ids, c_p, n_ids, n_p, valid)
            por_epoca[col] = niv

            # control 1: con estos niveles se reproduce hybrid() pixel a pixel
            hib, _ = hybrid(c_ids, c_p, n_ids, n_p, valid)
            rec = np.empty_like(c_p)
            for d, (nivel, clim_media) in niv.items():
                k = dates == d
                rec[k] = np.maximum(c_p[k] + (nivel - clim_media), 0.0)
            dif = float(np.abs(rec - hib).max())
            dif_hib_max = max(dif_hib_max, dif)
            if dif > 1e-9:
                fallos.append(f'control 1: semilla {s}{e}: diferencia {dif:.3e} m con hybrid()')

            # control 2: media de la climatologia, frente a la tabla de clim_v2
            for d, (_, clim_media) in niv.items():
                if d in test_tabla:
                    dc = abs(clim_media - test_tabla[d])
                    dif_clim_max = max(dif_clim_max, dc)
                    if dc > TOL_CLIM:
                        fallos.append(f'control 2: {d}: Cm {clim_media:.6f} (predicciones) frente a '
                                      f'{test_tabla[d]:.6f} (clim_v2)')
        if len(por_epoca) != len(EPOCAS):
            continue

        filas = []
        for r in tabla:
            if r['split'] == 'test':
                continue
            if r['nivel_obs_m'] in ('', None):
                fallos.append(f"{r['date']} ({r['split']}): sin nivel observado en {a.niveles_clim}")
                continue
            filas.append({'date': r['date'], 'split': r['split'],
                          'nivel_m': float(r['nivel_obs_m']), 'nivel_last_m': float(r['nivel_obs_m']),
                          'clim_media_m': float(r['clim_media_m']), 'origen_nivel': 'observado'})
        for d in fechas_test:
            filas.append({'date': d, 'split': 'test',
                          'nivel_m': por_epoca['nivel_m'][d][0],
                          'nivel_last_m': por_epoca['nivel_last_m'][d][0],
                          'clim_media_m': por_epoca['nivel_m'][d][1],
                          'origen_nivel': f'red B, semilla {s}'})
        tablas[s] = filas
        resumen['test_por_semilla'][str(s)] = {
            d: {'nivel_m': round(por_epoca['nivel_m'][d][0], 6),
                'nivel_last_m': round(por_epoca['nivel_last_m'][d][0], 6),
                'clim_media_m': round(por_epoca['nivel_m'][d][1], 6)} for d in fechas_test}
        print(f'semilla {s}:')
        for d in fechas_test:
            v = resumen['test_por_semilla'][str(s)][d]
            print(f"  {d}  L(d) mejor epoca {v['nivel_m']:.3f} m, ultima {v['nivel_last_m']:.3f} m; "
                  f"Cm(d) {v['clim_media_m']:.3f} m")

    resumen['controles'] = {'1_reproduce_hybrid_max_dif_m': dif_hib_max,
                            '2_clim_media_max_dif_m': dif_clim_max, 'tolerancia_2_m': TOL_CLIM,
                            'fallos': fallos, 'superados': not fallos}
    print(f'\nControl 1 (reproduce la combinacion): diferencia maxima {dif_hib_max:.2e} m')
    print(f'Control 2 (media de la climatologia frente a clim_v2): diferencia maxima {dif_clim_max:.2e} m')
    if fallos:
        print('Controles NO superados:')
        for f in fallos:
            print('  ' + f)
        raise SystemExit('STOP: no se escribe nada.')

    os.makedirs(a.out_dir, exist_ok=True)
    for s, filas in tablas.items():
        path = os.path.join(a.out_dir, f'niveles_s{s}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            for f in filas:
                w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in f.items()})
        print(f'  {path}: {len(filas)} fechas')
    with open(os.path.join(a.out_dir, 'niveles_resumen.json'), 'w', encoding='utf-8') as fh:
        json.dump(resumen, fh, indent=2, ensure_ascii=False)
    print(f'Controles: SUPERADOS. Resumen: {os.path.join(a.out_dir, "niveles_resumen.json")}')


if __name__ == '__main__':
    main()
