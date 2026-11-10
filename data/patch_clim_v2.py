r"""
Canales de climatologia v2 para la red de dos cabezas (sin GPU)
================================================================

Protocolo 2026, enmienda 3 (apartado 8.13). Escribe, para cada tesela del
dataset v4_ms_sx200, un .npy aparte con los mismos 2 canales que clim_v1:

    0  clim        climatologia de HS (m) en cada pixel; 0 donde no hay
    1  clim_disp   1 donde hay climatologia, 0 donde no

y una tabla con dos valores por fecha que necesita la red de dos cabezas.

Diferencia con clim_v1 (solo en las fechas de entrenamiento)
------------------------------------------------------------
clim_v1 es causal y deja la temporada 2021 sin climatologia. En la red de
dos cabezas la climatologia es la base de la salida, y con C = 0 la red
tendria que dibujar sola (la degeneracion que se quiere evitar). Por eso, en
las fechas de ENTRENAMIENTO, cada temporada recibe la climatologia de las
OTRAS temporadas de entrenamiento, nunca la suya:

      temporada 2021 (train)  -> 2022 y 2023
      temporada 2022 (train)  -> 2021 y 2023
      temporada 2023 (train)  -> 2021 y 2022
      validacion 2024 y test  -> 2021, 2022 y 2023   (igual que clim_v1)

El resto del calculo es el de clim_v1 (data/patch_clim_channels.py): mosaico
de cada fecha sobre la rejilla del DEM con data/tile_georef.csv y media
FECHA A FECHA de los mosaicos. La validacion (2024) no entra en ninguna
climatologia. clim_v1 no se toca.

Tabla por fecha (analysis/niveles_por_fecha_clim_v2.csv)
--------------------------------------------------------
Sobre el mosaico de la fecha (sin contar dos veces el solape de las teselas
de entrenamiento) y sobre todos los pixeles volados con dato valido, como la
combinacion (scripts/clim_v1_hybrid.py):

    clim_media_m   C(d): media del canal clim (0 donde no hay climatologia)
    nivel_obs_m    L(d) observado: media de la HS observada, con los
                   negativos a 0, que es como entra la HS en la perdida y en
                   la evaluacion (SnowDataset._clean). Solo en entrenamiento
                   y validacion; en el test queda vacio, porque alli el nivel
                   es el de la red B (apartado 8.13).

Controles (si alguno falla, se detiene)
---------------------------------------
1. Validacion y test: cada tesela de clim_v2 identica bit a bit a la de
   clim_v1 (se compara con el fichero existente).
2. Ninguna fecha usa su propia temporada; validacion y test usan todas las
   temporadas de entrenamiento.
3. Ninguna tesela de entrenamiento queda sin climatologia (aviso con el
   numero de teselas y la cobertura por fecha).

Salida
------
    <dataset>/clim_v2/<tile_id>              float32 (2, 256, 256)
    analysis/niveles_por_fecha_clim_v2.csv   C(d) y L(d) por fecha
    analysis/clim_v2_canales.json            temporadas, cobertura, controles

Uso (desde la raiz del repo)
----------------------------
    python data/patch_clim_v2.py --dataset RUTA/dataset_v4_ms_sx200 --limit 5
    python data/patch_clim_v2.py --dataset RUTA/dataset_v4_ms_sx200
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_clim_channels import TILE, mosaic, r6, season_of  # noqa: E402

CHANNELS = ['clim', 'clim_disp']


def main():
    ap = argparse.ArgumentParser(description='Canales de climatologia v2 (red de dos cabezas)')
    ap.add_argument('--dataset', default='dataset_v4_ms_sx200')
    ap.add_argument('--csv', default=None, help='CSV del split (por defecto <dataset>/dataset_v4_ms_sx200.csv)')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--v1-subdir', default='clim_v1')
    ap.add_argument('--out-subdir', default='clim_v2')
    ap.add_argument('--niveles', default='analysis/niveles_por_fecha_clim_v2.csv')
    ap.add_argument('--summary', default='analysis/clim_v2_canales.json')
    ap.add_argument('--limit', type=int, default=0, help='solo las N primeras teselas de cada fecha (prueba)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    t0 = time.time()

    masks_dir = os.path.join(a.dataset, 'masks')
    v1_dir = os.path.join(a.dataset, a.v1_subdir)
    out_dir = os.path.join(a.dataset, a.out_subdir)
    if not os.path.isdir(v1_dir):
        raise SystemExit(f'STOP: no existe {v1_dir} (hace falta para el control 1)')
    os.makedirs(out_dir, exist_ok=True)

    split = pd.read_csv(a.csv or os.path.join(a.dataset, 'dataset_v4_ms_sx200.csv'), dtype={'date': str})
    geo = pd.read_csv(a.georef, dtype={'date': str})
    geo = geo.merge(split[['tile_id', 'exp_temporal_split']], on='tile_id', how='inner')
    if len(geo) != len(split):
        raise SystemExit(f'STOP: {len(split)} teselas en el CSV y {len(geo)} con georreferencia')
    geo['season'] = geo['date'].map(season_of)
    shape = (int(geo['row_real'].max()) + TILE, int(geo['col_real'].max()) + TILE)

    train = geo[geo['exp_temporal_split'] == 'train']
    train_seasons = sorted(int(x) for x in train['season'].unique())
    print(f'Temporadas de entrenamiento: {train_seasons}; rejilla {shape}')

    # --- acumuladores por temporada de entrenamiento (media fecha a fecha, como clim_v1)
    acc = {}
    for s in train_seasons:
        suma = np.zeros(shape, np.float64)
        cuenta = np.zeros(shape, np.int16)
        fechas = sorted(train.loc[train['season'] == s, 'date'].unique())
        for d in fechas:
            mz = mosaic(train[train['date'] == d], masks_dir, shape)
            v = np.isfinite(mz)
            suma[v] += mz[v]
            cuenta[v] += 1
        acc[s] = (suma, cuenta, fechas)
        print(f'  temporada {s}: {len(fechas)} fechas ({(time.time() - t0) / 60:.1f} min)')

    clim_cache = {}

    def climatology(seasons):
        # misma operacion y mismo orden de suma que clim_v1, para que validacion
        # y test salgan identicos bit a bit
        key = tuple(seasons)
        if key not in clim_cache:
            suma = sum(acc[s][0] for s in seasons)
            cuenta = sum(acc[s][1].astype(np.int32) for s in seasons)
            with np.errstate(invalid='ignore', divide='ignore'):
                clim_cache[key] = np.where(cuenta > 0, suma / np.maximum(cuenta, 1), np.nan
                                           ).astype(np.float32)
        return clim_cache[key]

    def seasons_for(row_split, season):
        if row_split == 'train':
            return [s for s in train_seasons if s != season]   # sin la temporada propia
        return list(train_seasons)                            # validacion y test: como clim_v1

    summary = {'descripcion': ('Canales de climatologia v2 (protocolo 2026, 8.13). En entrenamiento, '
                               'cada temporada recibe la climatologia de las otras temporadas de '
                               'entrenamiento; en validacion y test, igual que clim_v1.'),
               'canales': CHANNELS,
               'temporadas_entrenamiento': {int(s): acc[s][2] for s in train_seasons},
               'por_fecha': {}}
    filas_niveles = []
    n_written = n_skipped = n_compared = 0
    distintas_v1, sin_clim_train = [], []
    temporadas_ok = True

    for (date, sp), g in geo.groupby(['date', 'exp_temporal_split'], sort=True):
        season = season_of(date)
        used = seasons_for(sp, season)
        if season in used or (sp != 'train' and used != train_seasons) or not used:
            temporadas_ok = False
        clim = climatology(used)
        rows = g.head(a.limit) if a.limit else g

        cov = []
        for t, r, c in rows[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
            out = np.zeros((2, TILE, TILE), np.float32)
            w = clim[r:r + TILE, c:c + TILE]
            disp = np.isfinite(w)
            out[0][disp] = w[disp]
            out[1] = disp.astype(np.float32)
            cov.append(float(disp.mean()))
            if sp == 'train' and not disp.any():
                sin_clim_train.append(t)
            if sp != 'train':                                  # control 1
                p1 = os.path.join(v1_dir, t)
                if not os.path.exists(p1):
                    raise SystemExit(f'STOP: falta {p1} para el control 1')
                n_compared += 1
                if not np.array_equal(out, np.load(p1)):
                    distintas_v1.append(t)
            p = os.path.join(out_dir, t)
            if os.path.exists(p) and not a.overwrite:
                n_skipped += 1
            else:
                np.save(p, out)
                n_written += 1

        # --- C(d) y L(d) sobre el mosaico de la fecha (todas sus teselas)
        mz = mosaic(g, masks_dir, shape)
        v = np.isfinite(mz)
        cw = np.where(np.isfinite(clim), clim, 0.0)              # 0 donde no hay climatologia
        clim_media = float(cw[v].mean()) if v.any() else None
        nivel_obs = float(np.maximum(mz[v], 0.0).mean()) if (v.any() and sp != 'test') else None
        filas_niveles.append({'date': date, 'split': sp, 'temporada': season,
                              'temporadas_clim': '-'.join(str(s) for s in used),
                              'clim_media_m': r6(clim_media),
                              'nivel_obs_m': '' if nivel_obs is None else r6(nivel_obs),
                              'n_pix_validos': int(v.sum()),
                              'cobertura_clim': r6(float(np.isfinite(clim)[v].mean())) if v.any() else None})

        summary['por_fecha'][date] = {
            'split': sp, 'temporada': int(season), 'temporadas_usadas': [int(s) for s in used],
            'n_teselas': int(len(rows)), 'cobertura_media_teselas': r6(np.mean(cov)) if cov else None,
            'clim_media_m': r6(clim_media), 'nivel_obs_m': r6(nivel_obs)}
        print(f"{date} {sp:5s} temporada {season} <- {used}: {len(rows)} teselas, cobertura "
              f"{summary['por_fecha'][date]['cobertura_media_teselas']}, C(d) {r6(clim_media)} m, "
              f"L(d) {r6(nivel_obs) if nivel_obs is not None else '-'} m  ({(time.time() - t0) / 60:.1f} min)")

    ok1 = not distintas_v1
    summary['control_1_igual_a_clim_v1'] = {'teselas_comparadas': n_compared,
                                            'distintas': distintas_v1[:20], 'superado': ok1}
    summary['control_2_temporadas'] = {'superado': temporadas_ok}
    summary['control_3_entrenamiento_sin_climatologia'] = {'teselas': len(sin_clim_train),
                                                           'ejemplos': sin_clim_train[:20]}
    summary['teselas'] = {'escritas': n_written, 'ya_existian': n_skipped}

    print(f'\nControl 1 (validacion y test identicos a clim_v1): {n_compared} teselas, '
          f'{len(distintas_v1)} distintas -> {"SUPERADO" if ok1 else "NO SUPERADO"}')
    print(f'Control 2 (sin temporada propia): {"SUPERADO" if temporadas_ok else "NO SUPERADO"}')
    print(f'Control 3 (entrenamiento sin climatologia): {len(sin_clim_train)} teselas')
    if not (ok1 and temporadas_ok):
        print('STOP: no se escribe la tabla por fecha ni el resumen. Revisar antes de usar clim_v2.')
        raise SystemExit(1)

    os.makedirs(os.path.dirname(a.niveles) or '.', exist_ok=True)
    with open(a.niveles, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_niveles[0]))
        w.writeheader()
        w.writerows(filas_niveles)
    os.makedirs(os.path.dirname(a.summary) or '.', exist_ok=True)
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f'Teselas escritas {n_written}, ya existian {n_skipped}. Tabla: {a.niveles}. '
          f'Resumen: {a.summary} ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
