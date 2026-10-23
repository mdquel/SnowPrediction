r"""
Paso 2, analisis exploratorio: dibujo de la climatologia + nivel de la red
=========================================================================

Surge DESPUES de ver la lectura con el nivel corregido
(scripts/clim_v1_level_corrected.py): con el nivel perfecto, la climatologia
sola dibuja mejor que la red con climatologia, pero tal cual pierde por el
nivel. Aqui se combinan las dos piezas sin usar el vuelo real:

    hibrido(fecha) = max(0, climatologia + (media_red - media_clim))

* media_red y media_clim: medias de la prediccion de la red y de la
  climatologia sobre TODOS los pixeles volados de la fecha (dato valido), no
  sobre los pixeles con nieve observada. La zona volada se conoce antes de
  tener el resultado, asi que el hibrido es aplicable en la practica.
* El nivel se toma de cada red y semilla: Bclim con climatologia, Bclim con
  ella apagada y B del paso 1.
* Se mide tal cual, con los mismos pixeles y el mismo SPAEF que la
  evaluacion (funcion metrics de clim_v1_level_corrected.py). Referencia:
  la propia red tal cual.

Se presenta como analisis exploratorio, no fijado de antemano.

Uso (desde la raiz del repo)
----------------------------
    python scripts/clim_v1_hybrid.py --b-dir results/sat_v2_instancia
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clim_v1_level_corrected import date_of, load, metrics, mmm  # noqa: E402

SEEDS = (7, 42, 123)


def hybrid(clim_ids, clim_p, net_ids, net_p, valid):
    """Climatologia desplazada por fecha hasta la media que predice la red."""
    idx = {t: i for i, t in enumerate(net_ids)}
    order = np.array([idx[t] for t in clim_ids])
    net_p = net_p[order]
    dates = np.array([date_of(t) for t in clim_ids])
    out = np.empty_like(clim_p)
    for d in np.unique(dates):
        k = dates == d
        v = valid[k]
        shift = float(net_p[k][v].mean() - clim_p[k][v].mean())
        out[k] = np.maximum(clim_p[k] + shift, 0.0)
    return out, net_p


def main():
    ap = argparse.ArgumentParser(description='Dibujo de la climatologia + nivel de la red')
    ap.add_argument('--clim-dir', default='results/clim_v1')
    ap.add_argument('--b-dir', default='results/sat_v2_instancia')
    ap.add_argument('--out', default='analysis/clim_v1_hibrido.json')
    a = ap.parse_args()

    cs = os.path.join(a.clim_dir, 'climatologia_sola', 'climatologia_sola_predictions.npz')
    c_ids, c_p, c_o, c_v = load(cs)
    valid = c_v & np.isfinite(c_o)

    def path(modelo, s, e):
        if modelo == 'B_paso1':
            n = f'resunetpp_sat2_B_s{s}'
            return os.path.join(a.b_dir, n, f'{n}{e}_predictions.npz')
        n = f'resunetpp_clim1_Bclim_s{s}'
        suf = e + ('_sinclim' if modelo == 'Bclim_sin_climatologia' else '')
        return os.path.join(a.clim_dir, n, f'{n}{suf}_predictions.npz')

    res = {'descripcion': 'Exploratorio: max(0, climatologia + (media_red - media_clim)) por fecha; '
                          'medias sobre todos los pixeles volados de la fecha. Metricas tal cual.'}
    for epoca, e in (('mejor_epoca', ''), ('ultima_epoca', '_last')):
        res[epoca] = {}
        for modelo in ('Bclim_con_climatologia', 'Bclim_sin_climatologia', 'B_paso1'):
            red, hib = {}, {}
            for s in SEEDS:
                p = path(modelo, s, e)
                if not os.path.exists(p):
                    print(f'  [aviso] falta {p}')
                    continue
                n_ids, n_p, n_o, _ = load(p)
                h, n_al = hybrid(c_ids, c_p, n_ids, n_p, valid)
                red[s] = metrics(c_ids, n_al, c_o, c_v)['tal_cual']
                hib[s] = metrics(c_ids, h, c_o, c_v)['tal_cual']
                print(f"{epoca:12s} {modelo:24s} s{s:<3d} red R2 {red[s]['R2']:.3f} SPAEF {red[s]['SPAEF']:.3f}"
                      f" | clim + nivel de la red R2 {hib[s]['R2']:.3f} SPAEF {hib[s]['SPAEF']:.3f}", flush=True)
            if red:
                res[epoca][modelo] = {
                    'red_tal_cual': {k: mmm([red[s][k] for s in red]) for k in ('R2', 'SPAEF')},
                    'clim_mas_nivel_de_la_red': {k: mmm([hib[s][k] for s in hib]) for k in ('R2', 'SPAEF')},
                    'semillas': sorted(red)}

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print('\nResumen (media [min-max] de 3 semillas)')
    for epoca in ('mejor_epoca', 'ultima_epoca'):
        print(f'  [{epoca}]')
        for modelo, v in res[epoca].items():
            r, h = v['red_tal_cual'], v['clim_mas_nivel_de_la_red']
            print(f"    nivel de {modelo:24s} red R2 {r['R2']['media']:.3f} | clim + nivel R2 "
                  f"{h['R2']['media']:.3f} [{h['R2']['min']:.3f}-{h['R2']['max']:.3f}] SPAEF "
                  f"{h['SPAEF']['media']:.3f} [{h['SPAEF']['min']:.3f}-{h['SPAEF']['max']:.3f}]")
    print(f'Resultados: {a.out}')


if __name__ == '__main__':
    main()
