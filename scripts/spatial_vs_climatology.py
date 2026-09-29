r"""
Experimento C - La red en terreno nunca sobrevolado
===================================================

Contexto
--------
El experimento B mostro que, en una zona ya sobrevolada otros anyos, la
climatologia (media de los mapas anteriores) acierta el patron mejor que
la red. Pero la climatologia solo existe donde se ha volado antes. En
terreno nuevo, la red es la unica fuente de patron. Este experimento mide
cuanto vale.

Pregunta
--------
En la banda de test del split espacial (terreno que la red nunca vio),
¿que fraccion alcanza la red de lo que daria la climatologia si esa zona se
hubiera sobrevolado en anyos anteriores?

    RED            modelo espacial de E1 (lambda=0, semilla 7), entrenado en
                   otras zonas
    CLIMATOLOGIA   para cada tile de test y fecha, la media de ese mismo
                   sitio del terreno en los OTROS anyos. En la practica no
                   existiria (la zona no se volo): se usa como TECHO.

Diseno (fijado antes de ejecutar)
---------------------------------
* Coordenadas reales de data/tile_georef.csv.
* La climatologia de una fecha usa solo fechas de OTROS anyos, promediadas
  fecha a fecha (cada fecha pesa una vez).
* Se excluyen los tiles de test con terreno visto en entrenamiento (fuga
  del split espacial, 28 de 730).
* Dos versiones: tal cual y nivel perfecto (media real de cada fecha), que
  mide solo la calidad del dibujo.

  Cambio respecto al diseno propuesto: se omite la version con el nivel del
  regresor. En el split espacial todas las fechas aparecen en
  entrenamiento, asi que la pregunta de estimar el nivel de un anyo no
  visto no se plantea aqui.

Lectura (fijada antes)
----------------------
    fraccion del techo = R2 red / R2 climatologia, con nivel perfecto
        >= 75%   la red sustituye bien a los vuelos previos
        50-75%   sustitucion parcial: util, con coste claro
        < 50%    la red aporta poco incluso en terreno nuevo

Limitacion: una sola semilla. Si la fraccion queda cerca de un umbral,
habria que repetir con mas semillas.

Uso
---
    python scripts/spatial_vs_climatology.py --npz <predicciones del modelo espacial>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TILE = 256
SNOW_THRESHOLD = 0.01
MIN_PX = 10
MASKS_DIR = 'dataset_v4_ms_sx200/masks'
GEOREF_CSV = 'data/tile_georef.csv'


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


def main():
    ap = argparse.ArgumentParser(description='Experimento C: red frente a techo climatologico en terreno nuevo')
    ap.add_argument('--npz', required=True)
    ap.add_argument('--georef', default=GEOREF_CSV)
    ap.add_argument('--masks', default=MASKS_DIR)
    ap.add_argument('--out', default='analysis/eC_spatial_vs_climatology.json')
    args = ap.parse_args()

    geo = pd.read_csv(args.georef, dtype={'date': str})
    geo['year'] = geo['date'].str[:4].astype(int)
    H = int(geo['row_real'].max()) + TILE
    W = int(geo['col_real'].max()) + TILE
    pos = {t: (int(r), int(c)) for t, r, c in geo[['tile_id', 'row_real', 'col_real']].itertuples(index=False)}

    # ---- terreno visto en entrenamiento (para excluir la fuga) ----
    train_cov = np.zeros((H, W), dtype=bool)
    for r, c in geo.loc[geo['exp_spatial_split'] == 'train', ['row_real', 'col_real']].itertuples(index=False):
        train_cov[r:r + TILE, c:c + TILE] = True

    # ---- acumuladores por anyo: mapa de cada fecha, promediado fecha a fecha ----
    print('Reconstruyendo el mapa de cada fecha en coordenadas reales...', flush=True)
    years = sorted(geo['year'].unique())
    ysum = {y: np.zeros((H, W), dtype=np.float32) for y in years}
    ycnt = {y: np.zeros((H, W), dtype=np.int16) for y in years}
    for d, g in geo.groupby('date'):
        mosaico = np.full((H, W), np.nan, dtype=np.float32)
        for t, r, c in g[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
            path = os.path.join(args.masks, t)
            if not os.path.exists(path):
                continue
            m = np.load(path).astype(np.float32)
            ok = np.isfinite(m) & (m > -100)
            mosaico[r:r + TILE, c:c + TILE][ok] = m[ok]
        v = np.isfinite(mosaico)
        y = int(d[:4])
        ysum[y][v] += mosaico[v]
        ycnt[y][v] += 1
    tsum = sum(ysum.values())
    tcnt = sum(c.astype(np.int32) for c in ycnt.values())

    def clim_patch(year, r, c):
        """Media de las fechas de OTROS anyos en ese sitio."""
        s = tsum[r:r + TILE, c:c + TILE] - ysum[year][r:r + TILE, c:c + TILE]
        n = tcnt[r:r + TILE, c:c + TILE] - ycnt[year][r:r + TILE, c:c + TILE]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    # ---- predicciones del modelo espacial ----
    z = np.load(args.npz)
    ids = z['tile_ids']
    preds, tgts = z['preds'], z['targets']
    vals = z['valids'] if 'valids' in z.files else None

    items, excl_fuga, base_px, comun_px = [], 0, 0, 0
    for i, t in enumerate(ids):
        t = str(t)
        if t not in pos:
            continue
        r, c = pos[t]
        if train_cov[r:r + TILE, c:c + TILE].any():
            excl_fuga += 1
            continue
        y = int(t[:4])
        cp = clim_patch(y, r, c)
        tg = tgts[i]
        m = (tg > SNOW_THRESHOLD) & np.isfinite(tg)
        if vals is not None:
            m &= vals[i] > 0.5
        base_px += int(m.sum())
        m &= np.isfinite(cp)
        comun_px += int(m.sum())
        if m.sum() >= MIN_PX:
            items.append((i, t[:8], cp, m))

    # pasada 1: medias por fecha sobre la mascara comun
    acc_d = {}
    for i, d, cp, m in items:
        s = acc_d.setdefault(d, [0.0, 0.0, 0.0, 0])
        s[0] += float(tgts[i][m].sum()); s[1] += float(preds[i][m].sum())
        s[2] += float(cp[m].sum()); s[3] += int(m.sum())
    real = {d: v[0] / v[3] for d, v in acc_d.items()}
    medias = {'red': {d: v[1] / v[3] for d, v in acc_d.items()},
              'clim': {d: v[2] / v[3] for d, v in acc_d.items()}}

    # pasada 2
    accs = {(p, v): new_acc() for p in ('red', 'clim') for v in ('raw', 'orac')}
    corrs = {'red': [], 'clim': []}
    for i, d, cp, m in items:
        o = tgts[i][m].astype(np.float64)
        pat = {'red': preds[i][m].astype(np.float64), 'clim': cp[m].astype(np.float64)}
        for p, x in pat.items():
            if o.std() > 1e-9 and x.std() > 1e-9:
                corrs[p].append(float(np.corrcoef(o, x)[0, 1]))
            add(accs[(p, 'raw')], o, x)
            add(accs[(p, 'orac')], o, x + (real[d] - medias[p][d]))

    res = {p: dict(r2_raw=r2(accs[(p, 'raw')]), r2_oraculo=r2(accs[(p, 'orac')]),
                   corr=float(np.mean(corrs[p]))) for p in ('red', 'clim')}
    frac = res['red']['r2_oraculo'] / res['clim']['r2_oraculo']

    print(f'\nTiles de test: {len(ids)}  |  excluidos por fuga: {excl_fuga}  |  evaluados: {len(items)}')
    print(f'Cobertura de la climatologia: {100 * comun_px / max(base_px, 1):.1f}% de los pixeles de test')
    print()
    print(f"  {'':<34}{'tal cual':>10}{'nivel perfecto':>16}{'corr':>8}")
    print(f"  {'red (entrenada en otras zonas)':<34}{res['red']['r2_raw']:>10.3f}"
          f"{res['red']['r2_oraculo']:>16.3f}{res['red']['corr']:>8.3f}")
    print(f"  {'climatologia (techo)':<34}{res['clim']['r2_raw']:>10.3f}"
          f"{res['clim']['r2_oraculo']:>16.3f}{res['clim']['corr']:>8.3f}")
    print()
    print(f'  FRACCION DEL TECHO (nivel perfecto): {100 * frac:.0f}%')
    if frac >= 0.75:
        print('  Lectura: la red sustituye bien a los vuelos previos.')
    elif frac >= 0.5:
        print('  Lectura: sustitucion parcial: util, con coste claro.')
    else:
        print('  Lectura: la red aporta poco incluso en terreno nuevo.')
    if 0.70 <= frac <= 0.80 or 0.45 <= frac <= 0.55:
        print('  Aviso: cerca de un umbral; conviene repetir con mas semillas.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'resultados': res, 'fraccion_techo': frac,
                   'excluidos_fuga': excl_fuga, 'evaluados': len(items)}, f, indent=2)
    print(f'\nGuardado en: {args.out}')


if __name__ == '__main__':
    main()
