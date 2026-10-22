r"""
Canales de climatologia v1 (paso 2; sin GPU)
=============================================

Escribe, para cada tesela del dataset v4_ms_sx200, un .npy aparte con 2
canales nuevos. NO toca las teselas originales ni sat_v2: data/dataset.py
anade estos canales detras cuando la config los pide, como con sat_v2.

    0  clim        climatologia de HS (m) en cada pixel; 0 donde no hay
    1  clim_disp   1 donde hay climatologia, 0 donde no

Diseno (cerrado antes de generar; no se cambia a la vista de resultados)
------------------------------------------------------------------------
* Climatologia = media de TODOS los vuelos de las temporadas anteriores, sin
  mirar la fecha dentro de la temporada (dibujo tipico del terreno).
* Temporada = de octubre a septiembre, con el nombre del anyo en que termina
  (el vuelo del 16-12-2021 es de la temporada 2022).
* Causal: cada fecha usa solo las temporadas de ENTRENAMIENTO anteriores a la
  suya. Nunca su propia temporada (fuga) ni temporadas posteriores (futuro).
      temporada 2021 (train)  -> sin climatologia (clim = 0, clim_disp = 0)
      temporada 2022 (train)  -> 2021
      temporada 2023 (train)  -> 2021 y 2022
      validacion 2024 y test 2025 -> 2021, 2022 y 2023
  La validacion (2024) no entra en ninguna climatologia.
* Mismo calculo que el experimento B (scripts/climatology_baseline.py): cada
  fecha se reconstruye como mosaico sobre la rejilla del DEM colocando cada
  tesela en su posicion real (data/tile_georef.csv), y los mosaicos se
  promedian FECHA A FECHA (no tesela a tesela, para no dar mas peso a los
  sitios con teselas solapadas el mismo dia). Pixeles validos: finitos y
  > -100, como en B.
* El apagado aleatorio del canal (p = 0,5) y la evaluacion con y sin
  climatologia NO estan aqui: van en data/dataset.py y en la evaluacion.

Controles
---------
1. Igualdad con el experimento B: la climatologia de validacion y test
   (temporadas 2021-2023) debe ser identica a la de B en el fold 2025
   (build_climatology de scripts/climatology_baseline.py). Si difiere mas de
   TOL_CONTROL en algun pixel, el script se detiene.
2. Causalidad: para cada fecha se anotan las temporadas usadas y se comprueba
   que todas son anteriores a la suya.
3. Alineacion (diagnostico): en una muestra de teselas con nieve de cada fecha
   que tiene climatologia, correlacion de su HS con la climatologia
   desplazada de -1 a +1 pixel. Se anota el desplazamiento de maxima
   correlacion; no se aplica ninguno (la posicion es la de tile_georef.csv,
   validada contra el DEM, la misma de B).

Salida
------
    <dataset>/clim_v1/<tile_id>        float32 (2, 256, 256)
    analysis/clim_v1_canales.json      temporadas por fecha, cobertura,
                                       controles

Uso (desde la raiz del repo)
----------------------------
    python data/patch_clim_channels.py --dataset RUTA/dataset_v4_ms_sx200 --limit 20
    python data/patch_clim_channels.py --dataset RUTA/dataset_v4_ms_sx200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from climatology_baseline import build_climatology  # noqa: E402

TILE = 256
TOL_CONTROL = 1e-5
SNOW_MIN = 0.01          # m; pixeles con nieve para el diagnostico de alineacion
ALIGN_SAMPLE = 5         # teselas por fecha para el diagnostico
CHANNELS = ['clim', 'clim_disp']


def season_of(date):
    y, m = int(date[:4]), int(date[4:6])
    return y + 1 if m >= 10 else y


def mosaic(geo_date, masks_dir, shape):
    """Mapa de HS de una fecha sobre la rejilla del DEM (NaN sin dato)."""
    H, W = shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    for t, r, c in geo_date[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
        m = np.load(os.path.join(masks_dir, t)).astype(np.float32)
        ok = np.isfinite(m) & (m > -100)
        dst = out[r:r + TILE, c:c + TILE]
        dst[ok] = m[ok]
    return out


def r6(x):
    return None if x is None else round(float(x), 6)


def main():
    ap = argparse.ArgumentParser(description='Canales de climatologia v1')
    ap.add_argument('--dataset', default='dataset_v4_ms_sx200')
    ap.add_argument('--csv', default=None, help='CSV del split (por defecto <dataset>/dataset_v4_ms_sx200.csv)')
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--out-subdir', default='clim_v1')
    ap.add_argument('--summary', default='analysis/clim_v1_canales.json')
    ap.add_argument('--limit', type=int, default=0, help='solo las N primeras teselas de cada fecha (prueba)')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    t0 = time.time()

    masks_dir = os.path.join(a.dataset, 'masks')
    out_dir = os.path.join(a.dataset, a.out_subdir)
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

    # --- acumuladores por temporada de entrenamiento (media fecha a fecha)
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
        key = tuple(seasons)
        if key not in clim_cache:
            if not seasons:
                clim_cache[key] = None
            else:
                suma = sum(acc[s][0] for s in seasons)
                cuenta = sum(acc[s][1].astype(np.int32) for s in seasons)
                with np.errstate(invalid='ignore', divide='ignore'):
                    clim_cache[key] = np.where(cuenta > 0, suma / np.maximum(cuenta, 1), np.nan
                                               ).astype(np.float32)
        return clim_cache[key]

    def seasons_for(row_split, season):
        if row_split == 'train':
            return [s for s in train_seasons if s < season]
        return list(train_seasons)               # validacion y test

    # --- control 1: igualdad con el experimento B (fold 2025)
    clim_all = climatology(train_seasons)
    clim_b, n_b = build_climatology(geo[['tile_id', 'date', 'row_real', 'col_real']], '2025',
                                    masks_dir, shape)
    both = np.isfinite(clim_all) & np.isfinite(clim_b)
    same_cov = bool(np.array_equal(np.isfinite(clim_all), np.isfinite(clim_b)))
    d_b = float(np.abs(clim_all[both] - clim_b[both]).max()) if both.any() else 0.0
    ok_b = same_cov and d_b <= TOL_CONTROL
    print(f'Control con el experimento B: {n_b} fechas en B, misma cobertura {same_cov}, '
          f'max dif {d_b:.2e} -> {"SUPERADO" if ok_b else "NO SUPERADO"}')
    if not ok_b:
        raise SystemExit('STOP: la climatologia de validacion y test no coincide con la de B.')
    del clim_b

    summary = {'descripcion': 'Canales de climatologia v1 (paso 2). Climatologia causal por '
                              'temporada de entrenamiento; la validacion no entra en ninguna.',
               'canales': CHANNELS,
               'temporadas_entrenamiento': {int(s): acc[s][2] for s in train_seasons},
               'control_experimento_B': {'fechas_B': int(n_b), 'misma_cobertura': same_cov,
                                         'max_dif': r6(d_b), 'tolerancia': TOL_CONTROL,
                                         'superado': ok_b},
               'por_fecha': {}}
    n_written = n_skipped = 0
    causal_ok = True

    for (date, sp), g in geo.groupby(['date', 'exp_temporal_split'], sort=True):
        season = season_of(date)
        used = seasons_for(sp, season)
        if any(s >= season for s in used):
            causal_ok = False
        clim = climatology(used)
        rows = g.head(a.limit) if a.limit else g

        cov, corr_by_shift = [], {}
        for t, r, c in rows[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
            out = np.zeros((2, TILE, TILE), np.float32)
            if clim is not None:
                w = clim[r:r + TILE, c:c + TILE]
                disp = np.isfinite(w)
                out[0][disp] = w[disp]
                out[1] = disp.astype(np.float32)
                cov.append(float(disp.mean()))
            else:
                cov.append(0.0)
            p = os.path.join(out_dir, t)
            if os.path.exists(p) and not a.overwrite:
                n_skipped += 1
            else:
                np.save(p, out)
                n_written += 1

        # --- control 3: diagnostico de alineacion en una muestra con nieve
        if clim is not None:
            sample = rows.head(ALIGN_SAMPLE)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    cs = []
                    for t, r, c in sample[['tile_id', 'row_real', 'col_real']].itertuples(index=False):
                        rr, cc = r + dr, c + dc
                        if rr < 0 or cc < 0 or rr + TILE > shape[0] or cc + TILE > shape[1]:
                            continue
                        hs = np.load(os.path.join(masks_dir, t)).astype(np.float32)
                        w = clim[rr:rr + TILE, cc:cc + TILE]
                        ok = np.isfinite(hs) & (hs > SNOW_MIN) & np.isfinite(w)
                        if ok.sum() > 1000:
                            cs.append(np.corrcoef(hs[ok], w[ok])[0, 1])
                    if cs:
                        corr_by_shift[(dr, dc)] = float(np.mean(cs))
        best = max(corr_by_shift, key=corr_by_shift.get) if corr_by_shift else None

        summary['por_fecha'][date] = {
            'split': sp, 'temporada': int(season), 'temporadas_usadas': [int(s) for s in used],
            'n_teselas': int(len(rows)),
            'cobertura_media': r6(np.mean(cov)) if cov else None,
            'teselas_sin_climatologia': int(sum(1 for x in cov if x == 0.0)),
            'alineacion_corr_sin_desplazar': r6(corr_by_shift.get((0, 0))),
            'alineacion_mejor_desplazamiento': list(best) if best else None,
            'alineacion_corr_mejor': r6(corr_by_shift[best]) if best else None}
        f = summary['por_fecha'][date]
        print(f"{date} {sp:5s} temporada {season} <- {used}: {len(rows)} teselas, cobertura "
              f"{f['cobertura_media']}, corr {f['alineacion_corr_sin_desplazar']} "
              f"(mejor {f['alineacion_mejor_desplazamiento']})  ({(time.time() - t0) / 60:.1f} min)")

    summary['control_causalidad'] = {'superado': causal_ok}
    summary['teselas'] = {'escritas': n_written, 'ya_existian': n_skipped}
    moved = {d: v['alineacion_mejor_desplazamiento'] for d, v in summary['por_fecha'].items()
             if v['alineacion_mejor_desplazamiento'] not in (None, [0, 0])}
    summary['alineacion_fechas_con_maximo_desplazado'] = moved
    os.makedirs(os.path.dirname(a.summary) or '.', exist_ok=True)
    with open(a.summary, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f'\nCausalidad: {"SUPERADO" if causal_ok else "NO SUPERADO"}. '
          f'Fechas con la correlacion maxima desplazada: {moved or "ninguna"}')
    print(f'Teselas escritas {n_written}, ya existian {n_skipped}. Resumen: {a.summary} '
          f'({(time.time() - t0) / 60:.1f} min)')
    if not causal_ok:
        raise SystemExit('STOP: alguna fecha usa su propia temporada o una posterior.')


if __name__ == '__main__':
    main()
