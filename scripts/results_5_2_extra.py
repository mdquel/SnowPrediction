r"""
Anexo a Resultados 5.2: SPAEF sin corregir el nivel (hallazgo 13), sin GPU
=========================================================================

Decision del 12-oct-2026: el SPAEF se usa siempre SIN corregir el nivel (su
termino beta, cociente de coeficientes de variacion, cambia con un
desplazamiento aditivo). Este script anade lo que faltaba con esa metrica y
lo guarda en analysis/cifras_canonicas_5_2.json SIN borrar nada de lo que ya
hay (anade los bloques 5, 6 y 7):

5. Figura de mapas: misma tesela que eligio scripts/results_5_2.py (bloque
   4_tesela_figura; la regla no cambia, usa el R2 con nivel perfecto). En
   cada panel: R2 con nivel perfecto y SPAEF sin corregir. Se anaden a
   analysis/fig05_tesela.npz los campos sin corregir (claves *_sin_corregir),
   conservando los que ya habia.

6. Hallazgo 11 con la metrica nueva: RF antiguo (solo pixeles con nieve,
   results de scripts/rf_spatial.py) frente al RF justo, experimento C:
   fraccion del techo en SPAEF sin corregir y en R2 con nivel perfecto.
   Control: = analysis/eC_net_vs_rf.json.

7. Fila 9 de la verificacion (mapa duplicado): experimento C quitando
   20240314 de la evaluacion y de la climatologia (como
   spatial_models_vs_climatology.py --exclude-dates), red y RF justo, con el
   SPAEF sin corregir, frente al resultado con todas las fechas.
   Control: = analysis/eC_net_vs_rf_allpix_sin_20240314.json.

Mismo calculo que scripts/results_5_2.py (se importa evaluate_C).

Uso (desde la raiz del repo)
----------------------------
    python scripts/results_5_2_extra.py ^
        --spatial-net <npz s7> <npz s42> <npz s123> ^
        --spatial-rf  <npz s7> <npz s42> <npz s123> ^
        --spatial-rf-old <npz s7> <npz s42> <npz s123>
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from results_5_2 import (evaluate_C, build_year_climatology, compute_spaef,   # noqa: E402
                         r2_arr, mmm, TILE)
import results_5_2 as R52                                                    # noqa: E402

DUP_DATE = '20240314'


def make_clim(geo, masks, H, W):
    ysum, ycnt, tsum, tcnt = build_year_climatology(geo, masks, H, W)

    def clim_patch(year, r, c):
        s = tsum[r:r + TILE, c:c + TILE] - ysum[year][r:r + TILE, c:c + TILE]
        n = tcnt[r:r + TILE, c:c + TILE] - ycnt[year][r:r + TILE, c:c + TILE]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return clim_patch


def summary(rs):
    keys = ('r2_raw', 'r2_orac', 'spaef_raw', 'frac_r2', 'frac_spaef_raw')
    out = {k: mmm([r[k] for r in rs]) for k in keys}
    out['techo_r2_orac'] = round(rs[0]['techo_r2'], R52.ND)
    out['techo_spaef_sin_corregir'] = round(rs[0]['clim_spaef_raw'], R52.ND)
    out['n_teselas'] = rs[0]['n_tiles']
    return out


def control(rs, ref_rows):
    d = []
    for a, b in zip(rs, ref_rows):
        d += [abs(a[k] - b[k]) for k in ('r2_raw', 'r2_orac', 'spaef_raw', 'spaef_orac',
                                          'techo_r2', 'techo_spaef')]
    return max(d) if d else None


def main():
    ap = argparse.ArgumentParser(description='Anexo 5.2: SPAEF sin corregir')
    ap.add_argument('--spatial-net', nargs=3, required=True)
    ap.add_argument('--spatial-rf', nargs=3, required=True)
    ap.add_argument('--spatial-rf-old', nargs=3, required=True)
    ap.add_argument('--masks', default='dataset_v4_ms_sx200/masks')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--json', default='analysis/cifras_canonicas_5_2.json')
    ap.add_argument('--fig-npz', default='analysis/fig05_tesela.npz')
    ap.add_argument('--dup-date', default=DUP_DATE, help='fecha del mapa duplicado')
    ap.add_argument('--dup-ref', default='analysis/eC_net_vs_rf_allpix_sin_20240314.json')
    ap.add_argument('--nd', type=int, default=4,
                    help='decimales de salida; usar el mismo que results_5_2.py')
    args = ap.parse_args()
    R52.set_nd(args.nd)
    t0 = time.time()

    canon = json.load(open(args.json, encoding='utf-8'))
    tile_id = canon['4_tesela_figura']['tile_id']

    geo_all = pd.read_csv(args.georef, dtype={'date': str})
    geo_all['year'] = geo_all['date'].str[:4].astype(int)
    H, W = int(geo_all['row_real'].max()) + TILE, int(geo_all['col_real'].max()) + TILE
    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in geo_all.loc[geo_all['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True

    # ================= todas las fechas: figura (5) y RF antiguo (6) =========
    print(f'Climatologia con todas las fechas  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    pos = {t: (int(r), int(c)) for t, r, c in geo_all[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    clim_patch = make_clim(geo_all, args.masks, H, W)

    fig, fig_met = {}, {}
    for label, p in (('red_s42', args.spatial_net[1]), ('rf_s42', args.spatial_rf[1])):
        print(f'  figura: {os.path.basename(p)}', flush=True)
        r, _, (items, preds, tgts, real, med) = evaluate_C(np.load(p), pos, train_cov, clim_patch)
        it = next(x for x in items if x[1] == tile_id)
        i, t, d, cp, m = it
        o = tgts[i][m].astype(np.float64)
        raw = preds[i].astype(np.float64)
        orac = raw + (real[d] - med['mod'][d])
        fig[label + '_sin_corregir'] = np.where(m, raw, np.nan)
        fig_met[label] = {'R2_nivel_perfecto': round(r2_arr(o, orac[m]), R52.ND),
                          'SPAEF_sin_corregir': round(float(compute_spaef(o, raw[m])), R52.ND)}
        if label == 'red_s42':
            craw = cp.astype(np.float64)
            corac = craw + (real[d] - med['clim'][d])
            fig['climatologia_sin_corregir'] = np.where(m, craw, np.nan)
            fig['mask_check'] = m
            fig_met['climatologia'] = {'R2_nivel_perfecto': round(r2_arr(o, corac[m]), R52.ND),
                                       'SPAEF_sin_corregir': round(float(compute_spaef(o, craw[m])), R52.ND)}
        elif not np.array_equal(m, fig['mask_check']):
            sys.exit('STOP: la mascara de la tesela difiere entre red y RF.')
    old = dict(np.load(args.fig_npz))
    if not np.array_equal(old['mask'], fig['mask_check']):
        sys.exit('STOP: la mascara de la tesela no coincide con fig05_tesela.npz.')
    del fig['mask_check']
    old.update(fig)
    np.savez_compressed(args.fig_npz, **old)
    prev = canon['4_tesela_figura']['metricas_tesela_nivel_perfecto']
    chk = max(abs(prev[k]['R2'] - fig_met[k]['R2_nivel_perfecto']) for k in fig_met)
    canon['5_figura_spaef_sin_corregir'] = {
        'tile_id': tile_id, 'fecha': canon['4_tesela_figura']['fecha'],
        'metricas_por_panel': fig_met,
        'control_R2_frente_a_bloque_4': round(chk, 6),
        'campos_nuevos_en': args.fig_npz + ' (claves *_sin_corregir)'}
    print(f'  paneles: {fig_met}  (control R2 {chk:.1e})', flush=True)

    rs_old = []
    for p in args.spatial_rf_old:
        print(f'  RF antiguo: {os.path.basename(p)}', flush=True)
        r, _, _ = evaluate_C(np.load(p), pos, train_cov, clim_patch)
        rs_old.append(r)
    ref = json.load(open('analysis/eC_net_vs_rf.json', encoding='utf-8')) \
        if os.path.exists('analysis/eC_net_vs_rf.json') else {}
    canon['6_hallazgo11_rf_antiguo'] = {
        'fuente': 'scripts/results_5_2_extra.py; RF antiguo = results de scripts/rf_spatial.py '
                  '(solo pixeles con nieve); RF justo = bloque 2_C',
        'rf_antiguo_solo_nieve': summary(rs_old),
        'rf_justo': {k: canon['2_C']['rf_justo'][k] for k in ('frac_r2', 'frac_spaef_raw', 'spaef_raw', 'r2_orac')},
        'control': {'max_diferencia': control(rs_old, ref.get('rf', [])), 'contra': 'analysis/eC_net_vs_rf.json'}}

    # ================= sin el mapa duplicado (7) ============================
    print(f'Climatologia sin {args.dup_date}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
    geo = geo_all[geo_all['date'] != args.dup_date].copy()
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}
    clim_patch = make_clim(geo, args.masks, H, W)
    res = {'red': [], 'rf': []}
    for label, paths in (('red', args.spatial_net), ('rf', args.spatial_rf)):
        for p in paths:
            print(f'  sin duplicado, {label}: {os.path.basename(p)}', flush=True)
            r, _, _ = evaluate_C(np.load(p), pos, train_cov, clim_patch)
            res[label].append(r)
    ref = json.load(open(args.dup_ref, encoding='utf-8')) if os.path.exists(args.dup_ref) else {}
    blk = {}
    for label, key in (('red', 'red'), ('rf', 'rf_justo')):
        s_sin = summary(res[label])
        con = canon['2_C'][key]
        blk[key] = {'sin_duplicado': s_sin,
                    'diferencia_frente_a_todas_las_fechas_puntos_pct': {
                        'frac_r2': round(100 * (s_sin['frac_r2']['media'] - con['frac_r2']['media']), R52.ND_PCT),
                        'frac_spaef_sin_corregir': round(100 * (s_sin['frac_spaef_raw']['media'] - con['frac_spaef_raw']['media']), R52.ND_PCT)},
                    'control_max_diferencia': control(res[label], ref.get(label, []))}
    canon['7_sin_mapa_duplicado'] = {
        'fuente': f'scripts/results_5_2_extra.py; fecha {args.dup_date} fuera de la evaluacion y de la '
                  'climatologia; los modelos no se reentrenan',
        'contra': args.dup_ref, **blk}

    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump(canon, f, indent=2, ensure_ascii=False,
                  default=lambda x: x.item() if hasattr(x, 'item') else str(x))

    print('=' * 72)
    print(f"5. Paneles ({tile_id}): {fig_met}")
    h = canon['6_hallazgo11_rf_antiguo']
    a, j = h['rf_antiguo_solo_nieve'], h['rf_justo']
    print(f"6. RF antiguo: frac R2 {100 * a['frac_r2']['media']:.1f}% [{100 * a['frac_r2']['min']:.1f}-"
          f"{100 * a['frac_r2']['max']:.1f}], frac SPAEF sin corregir {100 * a['frac_spaef_raw']['media']:.1f}% "
          f"[{100 * a['frac_spaef_raw']['min']:.1f}-{100 * a['frac_spaef_raw']['max']:.1f}] | RF justo "
          f"{100 * j['frac_r2']['media']:.1f}% / {100 * j['frac_spaef_raw']['media']:.1f}%  "
          f"(control {h['control']['max_diferencia']})")
    for key in ('red', 'rf_justo'):
        b = canon['7_sin_mapa_duplicado'][key]
        s = b['sin_duplicado']
        print(f"7. {key}: sin duplicado frac R2 {100 * s['frac_r2']['media']:.1f}%, frac SPAEF sin corregir "
              f"{100 * s['frac_spaef_raw']['media']:.1f}%; cambio (puntos) "
              f"{b['diferencia_frente_a_todas_las_fechas_puntos_pct']}  (control {b['control_max_diferencia']})")
    print(f'Guardado en: {args.json} y {args.fig_npz}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
