r"""
Consolidacion de las senales de escala por fecha de vuelo.
==========================================================

Que produce
-----------
data/scale_signals.csv, una fila por fecha de vuelo:

    fecha, snow_pct, pdd_15d, pdd_30d, ppAcc_mm, admisible

Es la entrada de la cabeza de escala del modelo E4. Tener las cuatro
senales en un unico fichero evita que el pipeline tenga que leer de tres
sitios distintos con formatos distintos.

Por que estas cuatro y no otras
-------------------------------
El analisis de senales (scripts/scale_signal_check.py) midio que
variables disponibles SIN VOLAR correlacionan con la magnitud real de
nieve. Resultado sobre datos reales:

    variable                r parcial   r anomalia
                          (sin estac.)  (intra-anyo)
    snow_pct                  +0.678       +0.151
    ppAcc_mm                  +0.334       -0.085
    pdd_15d                   -0.364       -0.672
    pdd_30d                   -0.294       -0.690
    t2m_7d / 15d / 30d    -0.135..-0.335  -0.117..-0.457

Se incluyen cobertura de nieve, los dos indices de grados-dia y la
precipitacion acumulada. Se EXCLUYE la temperatura media: no aporta
informacion de magnitud una vez descontada la estacionalidad, y eso
explica mecanicisticamente por que el experimento meteo del paper
(26 canales, tres de ellos ventanas de temperatura) salio negativo.

Las dos familias son complementarias, no redundantes:
  * snow_pct discrimina entre ANYOS nevados y secos (r parcial +0.678,
    pero solo +0.151 en anomalia intra-anual)
  * los grados-dia siguen la FUSION dentro de una temporada (r anomalia
    hasta -0.690, pero solo -0.29/-0.36 en parcial)

Fechas sin senal
----------------
Tres fechas (2021-02-02, 2021-05-04, 2024-03-07) no tienen observacion
de cobertura de nieve que cumpla el criterio fijado a priori (nubosidad
<= 20%, dentro de 3 dias). Se marcan con admisible=False y se excluyen
del experimento. Es la opcion mas limpia: imputar cobertura desde una
observacion con 100% de nube seria inventarse el dato.

Normalizacion
-------------
Las cuatro senales tienen rangos muy distintos (snow_pct 0-100,
ppAcc_mm 0-2000, pdd en decenas). Se guardan en unidades fisicas y se
normalizan en el dataset, no aqui, para que el CSV siga siendo legible
y verificable a ojo.

Uso
---
    python scripts/build_scale_signals.py
"""

import argparse
import os

import numpy as np
import pandas as pd

SNOW_CSV = 'data/snowcover_by_flight.csv'
DAILY_CSV = os.path.join('datos jesus', 'correo3', 'meteo_izas_daily.csv')
PPACC_CSV = os.path.join('datos jesus', 'correo2', 'ppAcc.csv')
DATASET_CSV = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'
OUT_CSV = 'data/scale_signals.csv'

TMIN, TMAX = -25.0, 30.0
MAX_GAP = 3


def load_daily_clean(path, tmin=TMIN, tmax=TMAX, max_gap=MAX_GAP):
    """Serie diaria de temperatura, limpia y en rejilla completa.

    Misma limpieza que clean_temp_and_recompute.py: filtro de rango
    fisico (temp_C contiene valores corruptos de hasta 1e34 sin
    centinela), reindexado diario e interpolacion solo de huecos cortos.
    La cabecera del fichero lleva comillas duplicadas, de ahi el skiprows.
    """
    cols = ['idx', 'day', 'temp_C', 'WS_ms', 'WD_deg', 'RH_perc', 'Rad_Wm']
    d = pd.read_csv(path, skiprows=1, names=cols)
    d['day'] = pd.to_datetime(d['day'], errors='coerce')
    d['temp_C'] = pd.to_numeric(d['temp_C'], errors='coerce')
    d = d.dropna(subset=['day']).drop_duplicates('day').set_index('day').sort_index()
    d.loc[(d['temp_C'] < tmin) | (d['temp_C'] > tmax), 'temp_C'] = np.nan
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='D'))
    d['temp_C'] = d['temp_C'].interpolate(limit=max_gap, limit_area='inside')
    return d['temp_C']


def positive_degree_days(temp, date, n):
    """Grados-dia positivos en los n dias anteriores a `date`.

    Suma de max(T, 0) sobre la ventana [D-n, D-1], excluyendo el dia del
    vuelo (misma convencion que las medias de temperatura, verificada
    contra la referencia de Jesus hasta 5.8e-15).

    La media de temperatura no distingue 30 dias a 0 C de 15 dias a -5 y
    15 a +5: ambos promedian 0, pero solo el segundo funde nieve. Los
    grados-dia suman unicamente el exceso sobre el punto de fusion, que
    es lo que gobierna el proceso.

    Devuelve (pdd, n_dias_con_dato).
    """
    w = temp.loc[date - pd.Timedelta(days=n):date - pd.Timedelta(days=1)]
    valid = w.dropna()
    if len(valid) == 0:
        return np.nan, 0
    return float(np.maximum(valid, 0).sum()), len(valid)


def load_ppacc(path):
    """Precipitacion acumulada desde el 1 de octubre. Serie diaria."""
    d = pd.read_csv(path)
    dcol = [c for c in d.columns if 'date' in c.lower()][0]
    acol = [c for c in d.columns if 'acc' in c.lower()][0]
    d[dcol] = pd.to_datetime(d[dcol], errors='coerce')
    d[acol] = pd.to_numeric(d[acol], errors='coerce')
    return (d[[dcol, acol]].rename(columns={dcol: 'fecha', acol: 'ppAcc_mm'})
            .dropna().set_index('fecha')['ppAcc_mm'])


def main():
    ap = argparse.ArgumentParser(description='Consolida las senales de escala')
    ap.add_argument('--snow', default=SNOW_CSV)
    ap.add_argument('--daily', default=DAILY_CSV)
    ap.add_argument('--ppacc', default=PPACC_CSV)
    ap.add_argument('--dataset', default=DATASET_CSV)
    ap.add_argument('--out', default=OUT_CSV)
    args = ap.parse_args()

    for p in (args.snow, args.daily, args.ppacc, args.dataset):
        if not os.path.exists(p):
            raise SystemExit(f'No se encuentra {p}. Ejecuta desde la raiz del repo.\n'
                             f'(scripts/check_snowcover.py genera {SNOW_CSV})')

    # --- fechas de vuelo ---
    ds = pd.read_csv(args.dataset)
    fechas = sorted(pd.to_datetime(ds['date'].astype(str), format='%Y%m%d').unique())
    print(f'Fechas de vuelo: {len(fechas)}')

    # --- cobertura de nieve ---
    snow = pd.read_csv(args.snow)
    snow['fecha_vuelo'] = pd.to_datetime(snow['fecha_vuelo'])
    snow = snow.set_index('fecha_vuelo')

    # --- temperatura y precipitacion ---
    temp = load_daily_clean(args.daily)
    ppacc = load_ppacc(args.ppacc)
    print(f'Serie diaria de temperatura: {temp.notna().sum()} dias con dato')
    print(f'Precipitacion acumulada    : {len(ppacc)} dias\n')

    rows = []
    for f in fechas:
        f = pd.Timestamp(f)
        row = {'fecha': f.strftime('%Y-%m-%d')}

        # cobertura
        if f in snow.index and bool(snow.loc[f, 'admisible']):
            row['snow_pct'] = float(snow.loc[f, 'snow_pct'])
            row['snow_ok'] = True
        else:
            row['snow_pct'] = np.nan
            row['snow_ok'] = False

        # grados-dia
        for n in (15, 30):
            pdd, ndias = positive_degree_days(temp, f, n)
            row[f'pdd_{n}d'] = pdd
            row[f'pdd_{n}d_ndias'] = ndias

        # precipitacion acumulada
        row['ppAcc_mm'] = float(ppacc.loc[f]) if f in ppacc.index else np.nan

        # admisible: exige las cuatro senales
        row['admisible'] = bool(
            row['snow_ok']
            and pd.notna(row['pdd_15d']) and row['pdd_15d_ndias'] >= 10
            and pd.notna(row['pdd_30d']) and row['pdd_30d_ndias'] >= 20
            and pd.notna(row['ppAcc_mm'])
        )
        rows.append(row)

    out = pd.DataFrame(rows)

    # --- informe ---
    print(f"{'fecha':<12}{'snow%':>8}{'pdd15':>8}{'pdd30':>8}{'ppAcc':>9}"
          f"{'d15':>5}{'d30':>5}   estado")
    for _, r in out.iterrows():
        g = lambda v, w, d=1: (f'{v:>{w}.{d}f}' if pd.notna(v) else f'{"-":>{w}}')
        motivos = []
        if not r['snow_ok']:
            motivos.append('sin cobertura')
        if r['pdd_15d_ndias'] < 10:
            motivos.append(f"pdd15 {int(r['pdd_15d_ndias'])}/15 d")
        if r['pdd_30d_ndias'] < 20:
            motivos.append(f"pdd30 {int(r['pdd_30d_ndias'])}/30 d")
        if pd.isna(r['ppAcc_mm']):
            motivos.append('sin ppAcc')
        estado = 'ok' if r['admisible'] else 'EXCLUIDA: ' + ', '.join(motivos)
        print(f"{r['fecha']:<12}{g(r['snow_pct'],8)}{g(r['pdd_15d'],8)}"
              f"{g(r['pdd_30d'],8)}{g(r['ppAcc_mm'],9)}"
              f"{int(r['pdd_15d_ndias']):>5}{int(r['pdd_30d_ndias']):>5}   {estado}")

    n_ok = int(out['admisible'].sum())
    print()
    print(f'  fechas admisibles : {n_ok} de {len(out)}')
    print(f'  excluidas         : {len(out) - n_ok}')
    print()

    out['anyo'] = pd.to_datetime(out['fecha']).dt.year
    print('Reparto por anyo (relevante para los folds LOYO):')
    for y, g in out.groupby('anyo'):
        print(f'  {y}: {int(g["admisible"].sum())} de {len(g)} fechas')
    print()

    # --- rangos, para dimensionar la normalizacion en el dataset ---
    adm = out[out['admisible']]
    if len(adm):
        print('Rangos de las senales admisibles (para normalizar en dataset.py):')
        for c in ('snow_pct', 'pdd_15d', 'pdd_30d', 'ppAcc_mm'):
            print(f'  {c:<12} min {adm[c].min():>9.2f}   max {adm[c].max():>9.2f}   '
                  f'media {adm[c].mean():>9.2f}')
        print()

    cols = ['fecha', 'snow_pct', 'pdd_15d', 'pdd_30d', 'ppAcc_mm', 'admisible',
            'snow_ok', 'pdd_15d_ndias', 'pdd_30d_ndias']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out[cols].to_csv(args.out, index=False)
    print(f'Guardado en: {args.out}')


if __name__ == '__main__':
    main()