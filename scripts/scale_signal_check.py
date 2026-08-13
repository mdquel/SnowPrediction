r"""
Que senales disponibles predicen la MAGNITUD de nieve?
======================================================

Pregunta
--------
E2 mostro que el error de extrapolacion interanual es de escala, no de
patron, y que un solo factor por fecha recupera hasta +0.28 de R2. Pero
ese factor se calcula con la media REAL de HS del dia, que solo se conoce
habiendo volado: es un oraculo, no un metodo.

Este script responde a la pregunta previa a cualquier experimento caro:
de las variables que SI estan disponibles sin volar, cuales correlacionan
con la magnitud real de nieve?

Si ninguna correlaciona, ninguna arquitectura podra extraer de ellas una
senal que no contienen, y no merece la pena gastar GPU. Si alguna lo
hace, ahi esta la entrada de la cabeza de escala.

Variables evaluadas
-------------------
    t2m_7d, t2m_15d, t2m_30d   temperatura media previa (recalculada)
    ppAcc_mm                    precipitacion acumulada desde el 1 oct
    snow_pct                    cobertura de nieve satelital
    snow_unclouded_pct          idem, corregida por nubes
    doy_sin, doy_cos            dia del anyo (control: estacionalidad pura)

Referencia (variable objetivo del analisis)
-------------------------------------------
    mean_SD de Izas_mean_SD.xlsx: media real de HS por fecha de vuelo.

    Esta variable NO puede usarse como entrada del modelo (es el
    estadistico del target y no existe sin volar). Aqui se usa unicamente
    como criterio para medir si las senales candidatas contienen
    informacion sobre la magnitud.

Nota importante sobre el control estacional
-------------------------------------------
Se incluye el dia del anyo como control deliberado. Si una variable
correlaciona con mean_SD solo porque ambas siguen el ciclo estacional,
esa correlacion no aporta nada al problema real, que es distinguir un
anyo nevado de uno seco EN LA MISMA EPOCA. Por eso se reportan tambien
las correlaciones parciales, descontando la estacionalidad, y las
correlaciones sobre la ANOMALIA respecto a la media de cada anyo.

Uso
---
    python scripts/scale_signal_check.py
"""

import argparse
import os

import numpy as np
import pandas as pd

T2M_CSV = 'data/t2m_recomputed.csv'
SNOW_CSV = 'data/snowcover_by_flight.csv'
PPACC_CSV = os.path.join('datos jesus', 'correo2', 'ppAcc.csv')
MEANSD_XLSX = os.path.join('Articulo 1', 'Data', 'izas', 'csv', 'Izas_mean_SD.xlsx')

CANDIDATES = ['t2m_7d', 't2m_15d', 't2m_30d', 'ppAcc_mm',
              'snow_pct', 'snow_unclouded_pct']


def load_mean_sd(path):
    d = pd.read_excel(path, header=1)
    d.columns = ['temporada', 'fecha', 'mean_SD', 'std_SD', 'skew_SD', 'SCA_dia'][:len(d.columns)]
    d = d.dropna(subset=['fecha'])
    d['fecha'] = pd.to_datetime(d['fecha'])
    for c in ('mean_SD', 'std_SD'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d[['fecha', 'mean_SD', 'std_SD']].dropna(subset=['mean_SD'])


def load_ppacc(path):
    d = pd.read_csv(path)
    dcol = [c for c in d.columns if 'date' in c.lower()][0]
    acc = [c for c in d.columns if 'acc' in c.lower()]
    if not acc:
        raise SystemExit(f'{path}: no encuentro columna de acumulado')
    d[dcol] = pd.to_datetime(d[dcol], errors='coerce')
    d = d.rename(columns={dcol: 'fecha', acc[0]: 'ppAcc_mm'})
    d['ppAcc_mm'] = pd.to_numeric(d['ppAcc_mm'], errors='coerce')
    return d[['fecha', 'ppAcc_mm']].dropna()


def partial_corr(x, y, controls):
    """Correlacion entre x e y descontando el efecto lineal de controls.

    Se regresa x y tambien y sobre los controles, y se correlacionan los
    residuos. Mide la asociacion que NO se explica por la estacionalidad.
    """
    C = np.column_stack([np.ones(len(x))] + [np.asarray(c) for c in controls])
    def resid(v):
        beta, *_ = np.linalg.lstsq(C, np.asarray(v), rcond=None)
        return np.asarray(v) - C @ beta
    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description='Que senales predicen la magnitud de nieve')
    ap.add_argument('--t2m', default=T2M_CSV)
    ap.add_argument('--snow', default=SNOW_CSV)
    ap.add_argument('--ppacc', default=PPACC_CSV)
    ap.add_argument('--meansd', default=MEANSD_XLSX)
    args = ap.parse_args()

    # --- referencia ---
    ms = load_mean_sd(args.meansd)
    df = ms.copy()

    # --- temperatura ---
    if os.path.exists(args.t2m):
        t = pd.read_csv(args.t2m)
        t['fecha'] = pd.to_datetime(t['fecha'])
        df = df.merge(t[['fecha', 't2m_7d', 't2m_15d', 't2m_30d']], on='fecha', how='left')
    else:
        print(f'[aviso] falta {args.t2m}')

    # --- precipitacion ---
    if os.path.exists(args.ppacc):
        df = df.merge(load_ppacc(args.ppacc), on='fecha', how='left')
    else:
        print(f'[aviso] falta {args.ppacc}')

    # --- cobertura de nieve ---
    if os.path.exists(args.snow):
        s = pd.read_csv(args.snow)
        s['fecha_vuelo'] = pd.to_datetime(s['fecha_vuelo'])
        s = s[s['admisible'] == True]
        df = df.merge(s[['fecha_vuelo', 'snow_pct', 'snow_unclouded_pct']],
                      left_on='fecha', right_on='fecha_vuelo', how='left')
    else:
        print(f'[aviso] falta {args.snow}')

    # --- control estacional ---
    doy = df['fecha'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    df['anyo'] = df['fecha'].dt.year
    print(f'Fechas con HS media de referencia: {len(df)} '
          f'({df["fecha"].min().date()} a {df["fecha"].max().date()})')
    print(f'HS media: {df["mean_SD"].mean():.3f} m  '
          f'(rango {df["mean_SD"].min():.3f} - {df["mean_SD"].max():.3f})\n')

    # ------------------------------------------------------------------
    print('CORRELACION CON LA MAGNITUD REAL DE NIEVE (mean_SD)')
    print(f"  {'variable':<22}{'n':>5}{'r bruto':>10}{'r parcial':>12}{'r anomalia':>13}")
    print(f"  {'':<22}{'':>5}{'':>10}{'(sin estac.)':>12}{'(intra-anyo)':>13}")

    results = []
    for col in CANDIDATES:
        if col not in df.columns:
            continue
        sub = df[[col, 'mean_SD', 'doy_sin', 'doy_cos', 'anyo']].dropna()
        if len(sub) < 5:
            print(f'  {col:<22}{len(sub):>5}   (muestra insuficiente)')
            continue

        r_raw = float(np.corrcoef(sub[col], sub['mean_SD'])[0, 1])

        r_par = partial_corr(sub[col], sub['mean_SD'],
                             [sub['doy_sin'], sub['doy_cos']])

        # anomalia intra-anyo: restar la media de cada anyo a ambas variables.
        # Mide si la variable distingue anyos nevados de secos una vez
        # eliminado el nivel medio de cada temporada.
        a = sub.copy()
        a['x_an'] = a[col] - a.groupby('anyo')[col].transform('mean')
        a['y_an'] = a['mean_SD'] - a.groupby('anyo')['mean_SD'].transform('mean')
        r_anom = (float(np.corrcoef(a['x_an'], a['y_an'])[0, 1])
                  if a['x_an'].std() > 1e-12 else np.nan)

        f = lambda v: f'{v:+.3f}' if pd.notna(v) else '   -  '
        print(f'  {col:<22}{len(sub):>5}{f(r_raw):>10}{f(r_par):>12}{f(r_anom):>13}')
        results.append({'variable': col, 'n': len(sub), 'r_raw': r_raw,
                        'r_partial': r_par, 'r_anomaly': r_anom})

    print()

    # ------------------------------------------------------------------
    # Lo que de verdad importa: predecir el NIVEL MEDIO DE CADA ANYO
    # ------------------------------------------------------------------
    print('CORRELACION A NIVEL DE ANYO (media anual de cada variable vs HS media anual)')
    print('  Es la pregunta operativa: distinguir un anyo nevado de uno seco.')
    yearly = df.groupby('anyo').mean(numeric_only=True)
    print(f"  {'variable':<22}{'n anyos':>9}{'r':>10}")
    for col in CANDIDATES:
        if col not in yearly.columns:
            continue
        sub = yearly[[col, 'mean_SD']].dropna()
        if len(sub) < 3:
            continue
        r = float(np.corrcoef(sub[col], sub['mean_SD'])[0, 1])
        print(f'  {col:<22}{len(sub):>9}{r:>+10.3f}')
    print()
    print('  AVISO: con 4-5 anyos, estas correlaciones son orientativas.')
    print('  No son test estadisticos; sirven para decidir si vale la pena')
    print('  gastar GPU, no para afirmar nada en el articulo.')
    print()

    # ------------------------------------------------------------------
    print('LECTURA')
    if results:
        best = max(results,
                   key=lambda r: abs(r['r_partial']) if pd.notna(r['r_partial']) else 0)
        rp = best['r_partial']
        print(f"  Mejor senal descontando estacionalidad: {best['variable']} "
              f"(r parcial = {rp:+.3f})")
        if pd.notna(rp) and abs(rp) >= 0.6:
            print('  >> Senal FUERTE: hay informacion de magnitud disponible.')
            print('     Merece la pena el experimento con entrada de escala.')
        elif pd.notna(rp) and abs(rp) >= 0.35:
            print('  >> Senal MODERADA: podria aportar, sin garantias.')
            print('     Experimento justificable, con expectativas contenidas.')
        else:
            print('  >> Senal DEBIL: las variables disponibles apenas contienen')
            print('     informacion sobre la magnitud. Ninguna arquitectura puede')
            print('     extraer lo que no esta en los datos. Reconsiderar antes')
            print('     de gastar GPU.')


if __name__ == '__main__':
    main()