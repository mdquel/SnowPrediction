r"""
Cruce de la cobertura de nieve satelital con las fechas de vuelo.
=================================================================

Que se busca
------------
Una senal de ESCALA disponible sin volar el dron.

E2 mostro que el error de extrapolacion interanual es predominantemente
de magnitud, no de patron: la correlacion espacial por tile se mantiene
constante (~0.625) sea cual sea la anomalia del anyo, mientras el sesgo
sigue esa anomalia de forma monotona. Una recalibracion de un solo
parametro por fecha recupera hasta +0.28 de R2.

Pero esa recalibracion usa la media REAL de nieve del dia, que solo se
conoce habiendo volado. Es un oraculo: mide el techo alcanzable, no es
un metodo desplegable. Lo que hace falta es una senal correlacionada con
la magnitud y disponible de antemano. La cobertura de nieve satelital es
candidata.

Criterio de admision (FIJADO ANTES DE VER LOS DATOS)
----------------------------------------------------
    nubosidad <= 20 %
    distancia temporal <= 3 dias respecto al vuelo

Se fija a priori de forma deliberada. Elegir el umbral despues de ver
cuantas fechas sobreviven con cada combinacion es optimizar el criterio
para maximizar la muestra, y no se sostiene ante revision.

Sobre Izas_mean_SD.xlsx
-----------------------
Contiene media, desviacion y asimetria de HS por fecha de vuelo. NO se
puede usar como predictor: es el estadistico del target, y darselo al
modelo seria fuga de informacion (ademas de no estar disponible para una
fecha nueva sin volar). Se carga unicamente como referencia, para poder
contrastar despues cualquier estimacion de escala contra el valor real.

Uso
---
    python scripts/check_snowcover.py
    python scripts/check_snowcover.py --max-cloud 20 --max-days 3
"""

import argparse
import os

import numpy as np
import pandas as pd

SNOWCOVER_XLSX = os.path.join('Articulo 1', 'Data', 'izas', 'csv', 'snowcover_izas.xlsx')
MEANSD_XLSX = os.path.join('Articulo 1', 'Data', 'izas', 'csv', 'Izas_mean_SD.xlsx')
DATASET_CSV = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'

MAX_CLOUD = 20.0    # % de nubosidad admisible
MAX_DAYS = 3        # dias de tolerancia respecto al vuelo


# ----------------------------------------------------------------------
def load_snowcover(path):
    """Serie de cobertura de nieve satelital.

    Columnas esperadas: date, %cloud cover, %snow cover,
    %unclouded snow cover. La ultima es la relevante: corrige el sesgo
    de tratar las nubes como ausencia de nieve, que el propio manuscrito
    reconoce como limitacion de los canales de persistencia actuales.
    """
    d = pd.read_excel(path)
    d.columns = [str(c).strip() for c in d.columns]
    ren = {}
    for c in d.columns:
        lc = c.lower()
        if lc == 'date':
            ren[c] = 'date'
        elif 'cloud cover' in lc and 'snow' not in lc:
            ren[c] = 'cloud_pct'
        elif 'unclouded' in lc:
            ren[c] = 'snow_unclouded_pct'
        elif 'snow cover' in lc:
            ren[c] = 'snow_pct'
    d = d.rename(columns=ren)
    d['date'] = pd.to_datetime(d['date'])
    for c in ('cloud_pct', 'snow_pct', 'snow_unclouded_pct'):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('date').reset_index(drop=True)


def load_mean_sd(path):
    """Estadisticos de HS por fecha de vuelo. SOLO como referencia."""
    d = pd.read_excel(path, header=1)
    d.columns = ['temporada', 'fecha', 'mean_SD', 'std_SD', 'skew_SD', 'SCA_dia'][:len(d.columns)]
    d = d.dropna(subset=['fecha'])
    d['fecha'] = pd.to_datetime(d['fecha'])
    for c in ('mean_SD', 'std_SD', 'skew_SD', 'SCA_dia'):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('fecha').reset_index(drop=True)


def flight_dates(path):
    df = pd.read_csv(path)
    return sorted(pd.to_datetime(df['date'].astype(str), format='%Y%m%d').unique())


def best_observation(sc, date, max_cloud, max_days):
    """Observacion admisible mas cercana a `date`.

    Devuelve (fila, dias_de_diferencia) o (None, None) si ninguna cumple
    el criterio. Ante empate en distancia, se prefiere la de menor
    nubosidad; es una regla determinista, no una eleccion caso a caso.
    """
    lo = date - pd.Timedelta(days=max_days)
    hi = date + pd.Timedelta(days=max_days)
    w = sc[(sc['date'] >= lo) & (sc['date'] <= hi)].copy()
    if w.empty:
        return None, None
    w['dias'] = (w['date'] - date).dt.days.abs()
    ok = w[w['cloud_pct'] <= max_cloud]
    if ok.empty:
        return None, None
    ok = ok.sort_values(['dias', 'cloud_pct'])
    row = ok.iloc[0]
    return row, int(row['dias'])


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Cruce de cobertura de nieve con fechas de vuelo')
    ap.add_argument('--snowcover', default=SNOWCOVER_XLSX)
    ap.add_argument('--meansd', default=MEANSD_XLSX)
    ap.add_argument('--dataset', default=DATASET_CSV)
    ap.add_argument('--max-cloud', type=float, default=MAX_CLOUD)
    ap.add_argument('--max-days', type=int, default=MAX_DAYS)
    ap.add_argument('--out', default='data/snowcover_by_flight.csv')
    args = ap.parse_args()

    for p in (args.snowcover, args.dataset):
        if not os.path.exists(p):
            raise SystemExit(f'No se encuentra {p}. Ejecuta desde la raiz del repo.')

    sc = load_snowcover(args.snowcover)
    print(f'Cobertura de nieve: {len(sc)} observaciones, '
          f'{sc["date"].min().date()} a {sc["date"].max().date()}')
    dif = sc['date'].diff().dt.days.dropna()
    print(f'  cadencia: mediana {dif.median():.0f} d, media {dif.mean():.1f} d, '
          f'maxima {dif.max():.0f} d')
    print(f'  nubosidad media: {sc["cloud_pct"].mean():.1f} %')
    print(f'  observaciones con nube <= {args.max_cloud}%: '
          f'{int((sc["cloud_pct"] <= args.max_cloud).sum())} de {len(sc)}')
    print()

    dates = flight_dates(args.dataset)
    print(f'Fechas de vuelo: {len(dates)}')
    print(f'Criterio (fijado a priori): nube <= {args.max_cloud}%, '
          f'distancia <= {args.max_days} d\n')

    rows = []
    print(f"  {'vuelo':<12}{'obs':>12}{'dias':>6}{'nube%':>8}"
          f"{'nieve%':>9}{'nieve_sin_nube%':>17}   estado")
    n_ok = 0
    for date in dates:
        date = pd.Timestamp(date)
        row, dd = best_observation(sc, date, args.max_cloud, args.max_days)
        if row is None:
            # sin observacion admisible: informar de la mas cercana, aunque no valga
            w = sc.copy()
            w['dias'] = (w['date'] - date).dt.days.abs()
            near = w.sort_values('dias').iloc[0]
            print(f"  {date.date()!s:<12}{'-':>12}{'-':>6}{'-':>8}{'-':>9}{'-':>17}   "
                  f"SIN SENAL (mas cercana: {near['date'].date()}, "
                  f"{int(near['dias'])} d, nube {near['cloud_pct']:.0f}%)")
            rows.append({'fecha_vuelo': date, 'obs_date': pd.NaT, 'dias_dif': np.nan,
                         'cloud_pct': np.nan, 'snow_pct': np.nan,
                         'snow_unclouded_pct': np.nan, 'admisible': False})
            continue

        n_ok += 1
        su = row.get('snow_unclouded_pct', np.nan)
        su_s = f'{su:.1f}' if pd.notna(su) else 'NaN'
        print(f"  {date.date()!s:<12}{row['date'].date()!s:>12}{dd:>6}"
              f"{row['cloud_pct']:>8.1f}{row['snow_pct']:>9.1f}{su_s:>17}   ok")
        rows.append({'fecha_vuelo': date, 'obs_date': row['date'], 'dias_dif': dd,
                     'cloud_pct': float(row['cloud_pct']),
                     'snow_pct': float(row['snow_pct']),
                     'snow_unclouded_pct': float(su) if pd.notna(su) else np.nan,
                     'admisible': True})

    print()
    print(f'  fechas con senal admisible : {n_ok} de {len(dates)}')
    print(f'  fechas sin senal           : {len(dates) - n_ok}')
    print()

    # --- reparto por anyo: importa para el diseno LOYO ---
    out = pd.DataFrame(rows)
    out['anyo'] = out['fecha_vuelo'].dt.year
    print('Reparto por anyo (relevante para los folds LOYO):')
    g = out.groupby('anyo')['admisible'].agg(['sum', 'count'])
    for y, r in g.iterrows():
        print(f'  {y}: {int(r["sum"])} de {int(r["count"])} fechas con senal')
    print()

    # --- referencia: HS real por fecha, solo para contraste posterior ---
    if os.path.exists(args.meansd):
        ms = load_mean_sd(args.meansd)
        merged = out.merge(ms[['fecha', 'mean_SD']], left_on='fecha_vuelo',
                           right_on='fecha', how='left')
        valid = merged[merged['admisible'] & merged['mean_SD'].notna()]
        print(f'Referencia Izas_mean_SD.xlsx: {len(ms)} fechas '
              f'({ms["fecha"].min().date()} a {ms["fecha"].max().date()})')
        print(f'  cruzan con fechas de vuelo con senal: {len(valid)}')
        if len(valid) >= 3:
            for col in ('snow_pct', 'snow_unclouded_pct'):
                sub = valid[valid[col].notna()]
                if len(sub) >= 3:
                    r = np.corrcoef(sub[col], sub['mean_SD'])[0, 1]
                    print(f'  correlacion {col} vs HS media real: r = {r:+.3f} '
                          f'(n={len(sub)})')
            print('  NOTA: esto solo indica si la senal esta relacionada con la')
            print('  magnitud. NO es un resultado del modelo, y mean_SD no puede')
            print('  usarse como entrada (es el estadistico del target).')
    else:
        print(f'[aviso] no se encuentra {args.meansd}: se omite el contraste')
    print()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Guardado en: {args.out}')


if __name__ == '__main__':
    main()