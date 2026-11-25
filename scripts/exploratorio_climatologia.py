r"""
Pruebas exploratorias de climatología (sin GPU; fuera del protocolo de 2026)
============================================================================

Plan registrado antes de calcular: analysis/exploratorio_climatologia.md.
Compara cuatro formas de construir y aplicar la climatología en la
combinación, medidas SOLO en la validación de 2024 y con el nivel observado
de cada fecha (sin la red B):

    A  media de los vuelos (= clim_v1)                    combinación sumando
    B  mediana de los vuelos                              combinación sumando
    C  media de los vuelos                                combinación multiplicando
    D  media de los vuelos, cada uno dividido antes por   combinación multiplicando
       su nieve media ("dibujo")

    sumando:       HS = max(0, C - Cm(d) + L(d))
    multiplicando: HS = max(0, C * L(d) / Cm(d))

Cm(d) y L(d) se calculan como en data/patch_clim_v2.py (apartado 8.13 del
protocolo): sobre el mosaico de la fecha (sin contar dos veces los solapes),
en todos los píxeles volados con dato válido; la climatología vale 0 donde no
la hay y la HS observada entra con los negativos a 0.

Las climatologías salen de los 15 vuelos de entrenamiento (temporadas
2021-2023), mosaico a mosaico sobre la rejilla del DEM, cada vuelo con el
mismo peso, como data/patch_clim_channels.py. No se lee ninguna tesela de
test (2025) ni de 2026.

Controles (si alguno falla, se detiene sin escribir nada)
--------------------------------------------------------
1. Solo se leen teselas de entrenamiento y validación, todas anteriores al
   30-09-2024, y la climatología usa solo las temporadas 2021-2023.
2. La variante A reproduce el Cm(d) de analysis/niveles_por_fecha_clim_v2.csv
   en las 7 fechas de validación (tolerancia 1e-4 m), y el L(d) observado.

Medidas, en las teselas de validación (como la evaluación de los modelos)
--------------------------------------------------------------------------
* R2 y RMSE acumulados en los píxeles con nieve (HS > 0,01 m) y dato válido,
  y sesgo (predicho - observado) en esos píxeles.
* Correlación media por tesela entre la HS observada y la climatología, en los
  píxeles con nieve, dato válido y climatología (teselas con al menos 1000).
  Es la medida del dibujo; no depende de la combinación.
* R2 por fecha.
* Criterio de mejora (registrado): B frente a A y D frente a C mejoran si el R2
  sube al menos 0,01, la correlación no baja y el R2 por fecha es mejor en al
  menos 5 de las 7 fechas.

Salida: analysis/exploratorio_climatologia.json

Uso (desde la raíz del repo)
----------------------------
    python scripts/exploratorio_climatologia.py --dataset RUTA/dataset_v4_ms_sx200
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_AQUI, '..', 'data'))
sys.path.insert(0, _AQUI)
from patch_clim_channels import mosaic, season_of  # noqa: E402

TILE = 256
SNOW = 0.01
LIMITE = '20240930'          # nada posterior (test 2025 y 2026 quedan fuera)
TOL_CTRL = 1e-4
MIN_PIX_CORR = 1000
UMBRAL_R2 = 0.01
MIN_FECHAS = 5
VARIANTES = {'A': ('media', 'suma'), 'B': ('mediana', 'suma'),
             'C': ('media', 'producto'), 'D': ('dibujo', 'producto')}


def acc_new():
    return {'n': 0, 'so': 0.0, 'so2': 0.0, 'sr': 0.0, 'se': 0.0}


def acc_add(acc, o, p):
    acc['n'] += o.size
    acc['so'] += float(o.sum())
    acc['so2'] += float((o ** 2).sum())
    acc['sr'] += float(((o - p) ** 2).sum())
    acc['se'] += float((p - o).sum())


def acc_metrics(acc):
    ss = acc['so2'] - acc['so'] ** 2 / acc['n']
    return {'R2': 1.0 - acc['sr'] / ss, 'RMSE': float(np.sqrt(acc['sr'] / acc['n'])),
            'sesgo': acc['se'] / acc['n'], 'n_pixeles': acc['n']}


def r6(x):
    return None if x is None else round(float(x), 6)


def main():
    # píxeles sin ningún vuelo: nanmean/nanmedian devuelven NaN, que es lo buscado
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    ap = argparse.ArgumentParser(description='Pruebas exploratorias de climatología (validación 2024)')
    ap.add_argument('--dataset', default='dataset_v4_ms_sx200')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--niveles', default='analysis/niveles_por_fecha_clim_v2.csv')
    ap.add_argument('--out', default='analysis/exploratorio_climatologia.json')
    a = ap.parse_args()
    t0 = time.time()
    masks_dir = os.path.join(a.dataset, 'masks')

    split = pd.read_csv(a.csv or os.path.join(a.dataset, 'dataset_v4_ms_sx200.csv'), dtype={'date': str})
    geo = pd.read_csv(a.georef, dtype={'date': str})
    geo = geo.merge(split[['tile_id', 'exp_temporal_split']], on='tile_id', how='inner')
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)
    # --- control 1: solo entrenamiento y validación, nada posterior al 30-09-2024
    geo = geo[geo['exp_temporal_split'].isin(['train', 'val'])].copy()
    if (geo['date'] > LIMITE).any():
        raise SystemExit('STOP: hay teselas de entrenamiento o validación posteriores al 30-09-2024')
    geo['season'] = geo['date'].map(season_of)
    train, val = geo[geo['exp_temporal_split'] == 'train'], geo[geo['exp_temporal_split'] == 'val']
    temporadas = sorted(int(s) for s in train['season'].unique())
    if temporadas != [2021, 2022, 2023]:
        raise SystemExit(f'STOP: temporadas de entrenamiento {temporadas}, se esperaban 2021-2023')
    fechas_train = sorted(train['date'].unique())
    fechas_val = sorted(val['date'].unique())
    print(f'Rejilla {shape}; {len(fechas_train)} vuelos de entrenamiento, {len(fechas_val)} fechas de validación')

    # --- mosaicos de entrenamiento y las tres climatologías
    pila = np.full((len(fechas_train),) + shape, np.nan, np.float32)
    for k, d in enumerate(fechas_train):
        pila[k] = mosaic(train[train['date'] == d], masks_dir, shape)
        print(f'  mosaico {d} ({(time.time() - t0) / 60:.1f} min)')
    with np.errstate(invalid='ignore', divide='ignore'):
        clim = {'media': np.nanmean(pila, axis=0)}
        print(f'  media ({(time.time() - t0) / 60:.1f} min)')
        clim['mediana'] = np.nanmedian(pila, axis=0)
        print(f'  mediana ({(time.time() - t0) / 60:.1f} min)')
        medias_vuelo = np.array([float(np.maximum(p[np.isfinite(p)], 0.0).mean()) for p in pila])
        for k in range(len(pila)):
            pila[k] /= medias_vuelo[k]
        clim['dibujo'] = np.nanmean(pila, axis=0)
    del pila
    print(f'  dibujo ({(time.time() - t0) / 60:.1f} min). Nieve media de cada vuelo: '
          + ', '.join(f'{d} {m:.3f}' for d, m in zip(fechas_train, medias_vuelo)))
    cw = {k: np.where(np.isfinite(v), v, 0.0).astype(np.float32) for k, v in clim.items()}

    # --- Cm(d) y L(d) por fecha de validación, sobre el mosaico de la fecha
    tabla = {r['date']: r for r in csv.DictReader(open(a.niveles, encoding='utf-8'))}
    niveles, fallos = {}, []
    for d in fechas_val:
        mz = mosaic(val[val['date'] == d], masks_dir, shape)
        v = np.isfinite(mz)
        L = float(np.maximum(mz[v], 0.0).mean())
        niveles[d] = {'L': L, **{f'Cm_{k}': float(cw[k][v].mean()) for k in cw}}
        ref = tabla.get(d)
        if ref is None:                                                   # control 2
            fallos.append(f'{d}: no está en {a.niveles}')
            continue
        dc, dl = abs(niveles[d]['Cm_media'] - float(ref['clim_media_m'])), abs(L - float(ref['nivel_obs_m']))
        print(f"  {d}: L {L:.4f} m (tabla {float(ref['nivel_obs_m']):.4f}), Cm media "
              f"{niveles[d]['Cm_media']:.4f} m (tabla {float(ref['clim_media_m']):.4f})")
        if dc > TOL_CTRL or dl > TOL_CTRL:
            fallos.append(f'{d}: Cm dif {dc:.2e} m, L dif {dl:.2e} m')
    if fallos:
        for f in fallos:
            print('  ' + f)
        raise SystemExit('STOP: la variante A no reproduce la tabla de niveles. No se escribe nada.')

    # --- evaluación en las teselas de validación
    acc = {k: acc_new() for k in VARIANTES}
    acc_fecha = {k: {d: acc_new() for d in fechas_val} for k in VARIANTES}
    corr = {k: [] for k in clim}
    for t, d, r, c in val[['tile_id', 'date', 'row_real', 'col_real']].itertuples(index=False):
        m = np.load(os.path.join(masks_dir, t)).astype(np.float32)
        ok = np.isfinite(m) & (m > -100)
        o = np.maximum(m, 0.0)
        snow = ok & (o > SNOW)
        if not snow.any():
            continue
        L = niveles[d]['L']
        for k, (base, modo) in VARIANTES.items():
            w = cw[base][r:r + TILE, c:c + TILE]
            cm = niveles[d][f'Cm_{base}']
            p = w - cm + L if modo == 'suma' else w * (L / cm)
            p = np.maximum(p, 0.0)
            acc_add(acc[k], o[snow].astype(np.float64), p[snow].astype(np.float64))
            acc_add(acc_fecha[k][d], o[snow].astype(np.float64), p[snow].astype(np.float64))
        for base in clim:
            w = clim[base][r:r + TILE, c:c + TILE]
            sel = snow & np.isfinite(w)
            if sel.sum() >= MIN_PIX_CORR and o[sel].std() > 1e-9 and w[sel].std() > 1e-9:
                corr[base].append(float(np.corrcoef(o[sel], w[sel])[0, 1]))

    res = {'descripcion': ('Pruebas exploratorias de climatología en la validación 2024 con el nivel '
                           'observado. Plan: analysis/exploratorio_climatologia.md. Fuera del protocolo 2026.'),
           'vuelos_climatologia': fechas_train, 'fechas_validacion': fechas_val,
           'nieve_media_por_vuelo_m': {d: r6(m) for d, m in zip(fechas_train, medias_vuelo)},
           'niveles_validacion': {d: {k: r6(v) for k, v in x.items()} for d, x in niveles.items()},
           'control_tabla_niveles': 'superado', 'variantes': {}}
    print(f"\n{'':4}{'climatología':14}{'combinación':14}{'R2':>8}{'RMSE':>8}{'sesgo':>8}{'corr':>8}")
    for k, (base, modo) in VARIANTES.items():
        mt = acc_metrics(acc[k])
        cr = float(np.mean(corr[base])) if corr[base] else None
        res['variantes'][k] = {'climatologia': base, 'combinacion': modo,
                               **{x: r6(y) if x != 'n_pixeles' else y for x, y in mt.items()},
                               'corr_por_tesela': r6(cr), 'n_teselas_corr': len(corr[base]),
                               'R2_por_fecha': {d: r6(acc_metrics(acc_fecha[k][d])['R2']) for d in fechas_val}}
        print(f"{k:4}{base:14}{modo:14}{mt['R2']:8.4f}{mt['RMSE']:8.4f}{mt['sesgo']:+8.4f}"
              f"{cr if cr is not None else float('nan'):8.4f}")

    res['criterio'] = {}
    for nueva, ref in (('B', 'A'), ('D', 'C')):
        vn, vr = res['variantes'][nueva], res['variantes'][ref]
        dr2 = vn['R2'] - vr['R2']
        dcorr = vn['corr_por_tesela'] - vr['corr_por_tesela']
        mejores = sum(vn['R2_por_fecha'][d] > vr['R2_por_fecha'][d] for d in fechas_val)
        cumple = dr2 >= UMBRAL_R2 and dcorr >= 0 and mejores >= MIN_FECHAS
        res['criterio'][f'{nueva}_frente_a_{ref}'] = {
            'dif_R2': r6(dr2), 'dif_corr': r6(dcorr), 'fechas_mejores': f'{mejores}/{len(fechas_val)}',
            'mejora_segun_criterio': cumple}
        print(f'{nueva} frente a {ref}: R2 {dr2:+.4f}, corr {dcorr:+.4f}, mejor en {mejores}/{len(fechas_val)} '
              f'fechas -> {"MEJORA" if cumple else "no mejora"} según el criterio registrado')
    dca = res['variantes']['C']['R2'] - res['variantes']['A']['R2']
    res['multiplicar_frente_a_sumar_C_menos_A_R2'] = r6(dca)
    print(f'C frente a A (multiplicar frente a sumar, informativo): R2 {dca:+.4f}')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f'\nResultados: {a.out} ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
