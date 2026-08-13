r"""
Limpieza de temp_C y recalculo de las medias moviles de temperatura.
====================================================================

Problema
--------
meteo_izas_daily.csv contiene valores corruptos en temp_C, sin centinela
que los marque: pasan como numeros validos. Ejemplos reales:

    2024-02-29     5 535 492 741 935 485 C
    2024-04-24    -2.27e+29 C
    2025-04-02     9.84e+34 C

Una media aritmetica es devastadoramente sensible a esto: un solo dia con
1e34 en la ventana de 30 dias produce 1e32/30, un numero sin sentido
fisico que entraria directo como canal de entrada al modelo.

El pipeline actual (patch_dataset_meteo2.py) filtra WS_ms, RH_perc y
Rad_Wm, pero NO temp_C. Este script cierra ese hueco.

Que hace
--------
1. Filtro de rango fisico sobre temp_C. Fuera de rango -> ausente.
   Antes de aplicar nada, imprime cuantos valores descartaria cada rango
   candidato, para elegir con datos y no a ojo.
2. Reindexado a rejilla diaria completa: los dias que faltan pasan a
   existir explicitamente como huecos, en lugar de simplemente no estar.
3. Interpolacion lineal SOLO de huecos de longitud <= MAX_GAP dias.
   Los huecos largos NO se rellenan: interpolar tres meses seria
   inventarse los datos (que es justo el error que tenia el pipeline
   con el 2022-04-17, al que asignaba +6.1 C arrastrando 87 dias).
4. Medias moviles t2m_7d / t2m_15d / t2m_30d para cada fecha de vuelo,
   con la ventana verificada: los N dias naturales ANTERIORES al vuelo,
   EXCLUYENDO el dia del vuelo (D-N ... D-1).
5. Por cada fecha y ventana, reporta cuantos dias son reales, cuantos
   interpolados y cuantos faltan. Eso es lo que permite decidir despues
   que fechas son utilizables y cuales hay que declarar o excluir.
6. Fusion con las medias de referencia de Jesus (t2m_datesUAV.csv).
   La validacion mostro que las diferencias entre ambas fuentes son
   proporcionales al numero de dias que faltan en la serie diaria: donde
   falta un dia la diferencia es de milesimas, donde faltan diez llega a
   1.6 C. Jesus trabajo sobre una serie mas completa, asi que sus valores
   son mejores donde existan. El recalculo aporta unicamente en las
   fechas que su fichero no cubre.

   Politica: referencia si esta disponible, recalculado en caso
   contrario. Se registra el origen de cada valor (columnas origen_Nd).

Sobre la definicion de ventana
------------------------------
Verificada empiricamente contra cinco fechas de t2m_datesUAV.csv, con
coincidencia hasta el sexto decimal. La alternativa (incluir el dia del
vuelo, D-6...D) no encaja: para 2021-02-02 da 2.1716 frente a 1.9063.

Que NO hace
-----------
No aplica filtro de saltos bruscos. Un dia con -24 C en septiembre es
implausible y un filtro de salto lo cazaria, pero tambien descartaria
cambios bruscos reales, que en montana existen. El script se limita a
AVISAR de los saltos grandes para revision manual; descartar datos
reales de forma automatica es peor que un aviso que se puede ignorar.

Uso
---
    python scripts/clean_temp_and_recompute.py
    python scripts/clean_temp_and_recompute.py --tmin -20 --tmax 20
    python scripts/clean_temp_and_recompute.py --out data/t2m_recomputed.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

DAILY_CSV = os.path.join('datos jesus', 'correo3', 'meteo_izas_daily.csv')
JESUS_CSV = os.path.join('datos jesus', 'correo2', 't2m_datesUAV.csv')
DATASET_CSV = 'dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv'

TMIN, TMAX = -25.0, 30.0      # rango fisico plausible (media diaria, ~2000 m)
MAX_GAP = 3                    # dias; huecos mas largos no se interpolan
WINDOWS = (7, 15, 30)          # ventanas de media movil
JUMP_WARN = 15.0               # aviso (no filtro) por salto entre dias


# ----------------------------------------------------------------------
def load_daily(path):
    """Carga la serie diaria de la estacion.

    El fichero tiene la cabecera con comillas duplicadas (""day"" en lugar
    de "day"), lo que impide a pandas casarla con las filas de datos. Se
    lee saltando la cabecera y asignando los nombres a mano. La primera
    columna es un indice correlativo, no un dato.
    """
    cols = ['idx', 'day', 'temp_C', 'WS_ms', 'WD_deg', 'RH_perc', 'Rad_Wm']
    d = pd.read_csv(path, skiprows=1, names=cols)
    d['day'] = pd.to_datetime(d['day'], errors='coerce')
    d['temp_C'] = pd.to_numeric(d['temp_C'], errors='coerce')
    return d[['day', 'temp_C']].copy()


def report_range_candidates(raw):
    """Cuenta cuantos valores descartaria cada rango candidato.

    Sirve para elegir el umbral con datos en lugar de a ojo: si un rango
    estrecho descarta muchos mas valores que uno amplio, esos valores
    adicionales son probablemente datos reales de verano/invierno, no
    corrupcion.
    """
    print('Valores de temp_C descartados por cada rango candidato:')
    print(f"  {'rango':>14}{'descartados':>13}{'% del total':>13}")
    n = len(raw)
    for lo, hi in [(-20, 20), (-25, 25), (-25, 30), (-30, 40), (-50, 60)]:
        k = int(((raw < lo) | (raw > hi)).sum())
        print(f'  {f"[{lo}, {hi}]":>14}{k:>13}{100*k/n:>12.3f}%')
    print()
    finite = raw[(raw > -100) & (raw < 100)]
    print(f'  Rango observado descartando basura evidente: '
          f'{finite.min():.1f} a {finite.max():.1f} C')
    print(f'  (sobre {len(finite)} de {n} valores)')
    print()


def clean_series(d, tmin, tmax, max_gap):
    """Filtra, reindexa e interpola huecos cortos.

    Devuelve un DataFrame indexado por dia con columnas:
        temp_C : serie limpia (NaN donde no se pudo reconstruir)
        origen : 'real' | 'interp' | 'falta'
    """
    d = d.dropna(subset=['day']).drop_duplicates('day').set_index('day').sort_index()

    bad = (d['temp_C'] < tmin) | (d['temp_C'] > tmax)
    n_bad = int(bad.sum())
    d.loc[bad, 'temp_C'] = np.nan

    # rejilla diaria completa: los dias ausentes pasan a existir como NaN
    full = pd.date_range(d.index.min(), d.index.max(), freq='D')
    n_missing_rows = len(full) - len(d)
    d = d.reindex(full)

    origen = pd.Series('real', index=d.index)
    origen[d['temp_C'].isna()] = 'falta'

    # interpolar solo huecos de longitud <= max_gap
    interp = d['temp_C'].interpolate(method='linear', limit=max_gap,
                                     limit_area='inside')
    newly = d['temp_C'].isna() & interp.notna()
    origen[newly] = 'interp'
    d['temp_C'] = interp
    d['origen'] = origen

    print(f'Limpieza (rango [{tmin}, {tmax}], huecos interpolables <= {max_gap} d):')
    print(f'  valores fuera de rango descartados : {n_bad}')
    print(f'  dias ausentes en el fichero        : {n_missing_rows}')
    print(f'  dias reales                        : {int((d["origen"]=="real").sum())}')
    print(f'  dias interpolados                  : {int((d["origen"]=="interp").sum())}')
    print(f'  dias sin reconstruir               : {int((d["origen"]=="falta").sum())}')
    print(f'  rango temporal                     : {d.index.min().date()} a {d.index.max().date()}')
    print()
    return d


def warn_jumps(d, thr):
    """Avisa (sin filtrar) de saltos bruscos entre dias consecutivos."""
    real = d[d['origen'] == 'real']['temp_C']
    jumps = real.diff().abs()
    big = jumps[jumps > thr]
    if len(big) == 0:
        print(f'Sin saltos mayores de {thr} C entre dias consecutivos.\n')
        return
    print(f'AVISO: {len(big)} saltos mayores de {thr} C entre dias consecutivos.')
    print('  No se filtran automaticamente (podrian ser reales), pero conviene')
    print('  revisarlos si alguno cae en una ventana de vuelo:')
    for day, val in big.head(15).items():
        print(f'    {day.date()}  salto de {val:.1f} C  (temp={real[day]:.1f})')
    if len(big) > 15:
        print(f'    ... y {len(big)-15} mas')
    print()


def rolling_mean(d, date, n):
    """Media de los n dias naturales anteriores a `date`, sin incluirlo.

    Ventana [D-n, D-1]. Definicion verificada contra t2m_datesUAV.csv.
    Devuelve (media, n_real, n_interp, n_falta).
    """
    start = date - pd.Timedelta(days=n)
    end = date - pd.Timedelta(days=1)
    w = d.loc[start:end]
    if len(w) == 0:
        return np.nan, 0, 0, n
    vals = w['temp_C']
    n_real = int((w['origen'] == 'real').sum())
    n_interp = int((w['origen'] == 'interp').sum())
    n_falta = n - n_real - n_interp
    mean = float(vals.mean()) if vals.notna().any() else np.nan
    return mean, n_real, n_interp, n_falta


def flight_dates(path):
    """Fechas unicas de vuelo del dataset."""
    df = pd.read_csv(path)
    return sorted(pd.to_datetime(df['date'].astype(str), format='%Y%m%d').unique())


def merge_with_reference(rows, jesus_path, tol=1e-6):
    """Fusiona las medias recalculadas con las de referencia de Jesus.

    Por que hace falta
    ------------------
    La validacion revela que las diferencias entre ambas fuentes son
    proporcionales al numero de dias que faltan en meteo_izas_daily.csv:
    donde falta un dia la diferencia es de milesimas, donde faltan diez
    llega a 1.6 C. Es decir, Jesus trabajo sobre una serie MAS COMPLETA
    que la que tenemos: el calculo con datos reales donde aqui hay huecos.

    Por tanto sus valores son mejores donde existan, y el recalculo solo
    aporta en las fechas que su fichero no cubre (2022-04-17), donde
    ademas el pipeline anterior interpolaba linealmente sobre 87 dias y
    asignaba +6.1 C, un valor implausible para mediados de abril.

    Politica: usar Jesus si esta disponible; recalculado en caso
    contrario. Se registra el origen de cada valor para poder declararlo.
    """
    if not os.path.exists(jesus_path):
        print(f'[aviso] no se encuentra {jesus_path}: se usan solo los valores recalculados\n')
        for r in rows:
            for n in WINDOWS:
                r[f'origen_{n}d'] = 'recalculado'
        return rows

    j = pd.read_csv(jesus_path)
    dcol = [c for c in j.columns if 'date' in c.lower()][0]
    j[dcol] = pd.to_datetime(j[dcol])
    j = j.set_index(dcol)

    print('Fusion con t2m_datesUAV.csv (valores de referencia de Jesus):')
    print(f"  {'fecha':<12}{'ventana':>8}{'usado':>13}{'origen':>14}{'dif con recalc':>16}")

    n_ref = n_recalc = 0
    worst_agree = 0.0
    for r in rows:
        in_ref = r['fecha'] in j.index
        for n in WINDOWS:
            col = f't2m_{n}d'
            mine = r[col]
            theirs = j.loc[r['fecha'], col] if (in_ref and col in j.columns) else np.nan

            if pd.notna(theirs):
                dif = abs(mine - theirs) if pd.notna(mine) else np.nan
                r[col] = float(theirs)
                r[f'origen_{n}d'] = 'referencia'
                n_ref += 1
                if pd.notna(dif):
                    if dif > tol:
                        print(f"  {r['fecha'].date()!s:<12}{n:>7}d{theirs:>13.4f}"
                              f"{'referencia':>14}{dif:>16.3e}")
                    else:
                        worst_agree = max(worst_agree, dif)
            else:
                r[f'origen_{n}d'] = 'recalculado'
                n_recalc += 1
                val = f'{mine:.4f}' if pd.notna(mine) else 'NaN'
                print(f"  {r['fecha'].date()!s:<12}{n:>7}d{val:>13}"
                      f"{'RECALCULADO':>14}{'-':>16}")

    print(f'  valores tomados de la referencia : {n_ref}')
    print(f'  valores recalculados             : {n_recalc}')
    print(f'  maxima discrepancia entre los que coinciden: {worst_agree:.2e}')
    print('  (las discrepancias mayores corresponden a fechas con huecos en')
    print('   meteo_izas_daily.csv; en ellas la referencia es mas fiable)')
    print()
    return rows


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Limpia temp_C y recalcula medias moviles')
    ap.add_argument('--daily', default=DAILY_CSV)
    ap.add_argument('--jesus', default=JESUS_CSV)
    ap.add_argument('--dataset', default=DATASET_CSV)
    ap.add_argument('--tmin', type=float, default=TMIN)
    ap.add_argument('--tmax', type=float, default=TMAX)
    ap.add_argument('--max-gap', type=int, default=MAX_GAP)
    ap.add_argument('--out', default='data/t2m_recomputed.csv')
    args = ap.parse_args()

    if not os.path.exists(args.daily):
        raise SystemExit(f'No se encuentra {args.daily}. Ejecuta desde la raiz del repo.')

    raw = load_daily(args.daily)
    print(f'Serie diaria: {len(raw)} filas, '
          f'{raw["day"].min().date()} a {raw["day"].max().date()}\n')

    report_range_candidates(raw['temp_C'])
    d = clean_series(raw, args.tmin, args.tmax, args.max_gap)
    warn_jumps(d, JUMP_WARN)

    dates = flight_dates(args.dataset)
    print(f'Fechas de vuelo en el dataset: {len(dates)}\n')

    rows = []
    for date in dates:
        date = pd.Timestamp(date)
        row = {'fecha': date}
        for n in WINDOWS:
            mean, nr, ni, nf = rolling_mean(d, date, n)
            row[f't2m_{n}d'] = mean
            row[f'n_real_{n}'] = nr
            row[f'n_interp_{n}'] = ni
            row[f'n_falta_{n}'] = nf
        rows.append(row)

    # --- tabla de calidad por fecha (ventana de 30 d, la mas exigente) ---
    print('Calidad de la ventana de 30 dias por fecha de vuelo:')
    print(f"  {'fecha':<12}{'t2m_30d':>10}{'real':>7}{'interp':>8}{'falta':>7}   estado")
    n_ok = n_partial = n_bad = 0
    for r in rows:
        nr, ni, nf = r['n_real_30'], r['n_interp_30'], r['n_falta_30']
        if nf == 0 and ni == 0:
            estado, n_ok = 'completa', n_ok + 1
        elif nf == 0:
            estado, n_partial = f'{ni} d interpolados', n_partial + 1
        else:
            estado, n_bad = f'INCOMPLETA ({nf} d sin dato)', n_bad + 1
        val = f"{r['t2m_30d']:.3f}" if pd.notna(r['t2m_30d']) else 'NaN'
        print(f"  {r['fecha'].date()!s:<12}{val:>10}{nr:>7}{ni:>8}{nf:>7}   {estado}")
    print()
    print(f'  ventanas completas          : {n_ok}')
    print(f'  con algun dia interpolado   : {n_partial}')
    print(f'  incompletas (declarar/excluir): {n_bad}')
    print()

    rows = merge_with_reference(rows, args.jesus)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Medias recalculadas guardadas en: {args.out}')
    print('Incluye las columnas de procedencia (n_real / n_interp / n_falta)')
    print('por ventana, para poder declarar la calidad de cada fecha.')


if __name__ == '__main__':
    main()