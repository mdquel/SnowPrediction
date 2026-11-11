r"""
Red de dos cabezas: techo en 2025 con w y delta ideales (sin GPU)
=================================================================

Peticion de Hernan Diaz (correo del 11-11-2026, punto 2), antes de entrenar:
calcular en 2025 el w y el delta IDEALES de cada tesela, por minimos
cuadrados y dentro de los limites del apartado 8.13, con el nivel observado
de la fecha, y compararlo con la combinacion con ese mismo nivel. La
diferencia es lo maximo que pueden aportar las cabezas.

Para cada tesela de la fecha d (formula del apartado 8.13):

    HS = max(0, L(d) + w * (C - Cm(d)) + delta)

    L(d), Cm(d)  media observada y media de la climatologia sobre todos los
                 pixeles volados de la fecha (definicion de 8.13)
    w, delta     los que minimizan el error cuadratico de la tesela, con
                 w en [0.5, 1.5] y delta en [-0.5, 0.5] m (problema
                 cuadratico convexo en una caja: se resuelve exacto)

Variantes: w y delta; solo delta (w = 1); solo w (delta = 0). Y dos
conjuntos de pixeles para el ajuste: los de la evaluacion (con nieve) y
todos los validos (los de la perdida). El R2 se mide siempre como en la
evaluacion: pixeles con nieve (HS > 0,01 m) y dato valido, acumulado sobre
las 220 teselas, prediccion recortada a 0.

Es un TECHO: usa la nieve observada de cada tesela para elegir w y delta.
La red tendra que deducirlos de sus entradas, y en el test el nivel sera el
de la red B, no el observado.

Controles (si alguno falla, se detiene sin escribir nada)
---------------------------------------------------------
1. La climatologia sola tal cual reproduce el R2 de su metrics.json.
2. La climatologia con el nivel corregido (desplazamiento por fecha calculado
   en los pixeles con nieve, sin recorte) reproduce el R2 de
   analysis/metricas_ampliadas_2025_regen2024.json.
3. Ninguna tesela posterior al 30-09-2025.

Salida: analysis/dos_cabezas_techo_2025.json

Uso (desde la raiz del repo)
----------------------------
    python scripts/dos_cabezas_techo.py
"""

import argparse
import json
import os

import numpy as np

SNOW = 0.01
W_RANGE = (0.5, 1.5)
DELTA_MAX = 0.5
TOL = 1e-3
LIMITE = '20250930'


def r2(o, pred, pix):
    """R2 acumulado como en la evaluacion (prediccion recortada a 0)."""
    so = so2 = sr = 0.0
    n = 0
    for i in range(len(o)):
        m = pix[i]
        if not m.any():
            continue
        oi, pi = o[i][m], np.maximum(pred[i][m], 0.0)
        so += oi.sum(); so2 += (oi ** 2).sum(); sr += ((oi - pi) ** 2).sum(); n += oi.size
    return 1.0 - sr / (so2 - so ** 2 / n)


def r2_sin_recorte(o, pred, pix):
    so = so2 = sr = 0.0
    n = 0
    for i in range(len(o)):
        m = pix[i]
        if not m.any():
            continue
        oi, pi = o[i][m], pred[i][m]
        so += oi.sum(); so2 += (oi ** 2).sum(); sr += ((oi - pi) ** 2).sum(); n += oi.size
    return 1.0 - sr / (so2 - so ** 2 / n)


def ajuste(a, y, usar_w=True, usar_d=True):
    """min ||y - w a - D||^2 con w en W_RANGE y D en [-DELTA_MAX, DELTA_MAX].

    Forma cuadratica convexa en una caja: si el minimo libre esta dentro, es
    la solucion; si no, esta en una arista (minimo en 1D recortado)."""
    def sse(w, d):
        return float(((y - w * a - d) ** 2).sum())
    cand = []
    s = float((a * a).sum())
    if usar_w and usar_d:
        va = a.var()
        if va > 0:
            w = float(((a - a.mean()) * (y - y.mean())).mean() / va)
            d = float(y.mean() - w * a.mean())
            if W_RANGE[0] <= w <= W_RANGE[1] and -DELTA_MAX <= d <= DELTA_MAX:
                return w, d
        for w in W_RANGE:
            cand.append((w, float(np.clip((y - w * a).mean(), -DELTA_MAX, DELTA_MAX))))
        for d in (-DELTA_MAX, DELTA_MAX):
            w = (a * (y - d)).sum() / s if s > 0 else 1.0
            cand.append((float(np.clip(w, *W_RANGE)), d))
    elif usar_d:
        cand.append((1.0, float(np.clip((y - a).mean(), -DELTA_MAX, DELTA_MAX))))
    else:
        cand.append((float(np.clip((a * y).sum() / s, *W_RANGE)) if s > 0 else 1.0, 0.0))
    return min(cand, key=lambda p: sse(*p))


def main():
    ap = argparse.ArgumentParser(description='Techo de la red de dos cabezas en 2025')
    ap.add_argument('--clim-dir', default='results/regen2024/clim_v1/climatologia_sola')
    ap.add_argument('--ampliadas', default='analysis/metricas_ampliadas_2025_regen2024.json')
    ap.add_argument('--out', default='analysis/dos_cabezas_techo_2025.json')
    a = ap.parse_args()

    z = np.load(os.path.join(a.clim_dir, 'climatologia_sola_predictions.npz'))
    ids = [str(t) for t in z['tile_ids']]
    c = np.squeeze(z['preds']).astype(np.float64)        # climatologia, 0 donde no hay
    o = np.squeeze(z['targets']).astype(np.float64)
    v = (np.squeeze(z['valids']) > 0.5) & np.isfinite(o)
    dates = np.array([t.split('_')[0] for t in ids])
    fallos = []
    if any(d > LIMITE for d in dates):
        raise SystemExit('STOP: hay teselas posteriores al 30-09-2025 (temporada reservada).')
    snow = v & (o > SNOW)

    # --- controles 1 y 2
    ref_tal = json.load(open(os.path.join(a.clim_dir, 'climatologia_sola_metrics.json')))['R2']
    r_tal = r2(o, c, snow)
    if abs(r_tal - ref_tal) > TOL:
        fallos.append(f'control 1: R2 tal cual {r_tal:.4f} frente a {ref_tal:.4f}')
    corr = np.empty_like(c)
    for d in np.unique(dates):
        k = dates == d
        sh = o[k][snow[k]].mean() - c[k][snow[k]].mean()
        corr[k] = c[k] + sh
    ref_niv = json.load(open(a.ampliadas))['climatologia_sola']['nivel_corregido']['R2']
    r_niv = r2_sin_recorte(o, corr, snow)
    if abs(r_niv - ref_niv) > TOL:
        fallos.append(f'control 2: R2 con nivel corregido {r_niv:.4f} frente a {ref_niv:.4f}')
    print(f'Control 1 (climatologia sola tal cual): {r_tal:.4f} frente a {ref_tal:.4f}')
    print(f'Control 2 (climatologia, nivel corregido): {r_niv:.4f} frente a {ref_niv:.4f}')
    if fallos:
        for f in fallos:
            print('  ' + f)
        raise SystemExit('STOP: no se escribe nada.')

    # --- nivel observado y media de la climatologia por fecha (definicion de 8.13)
    L = {d: float(o[dates == d][v[dates == d]].mean()) for d in np.unique(dates)}
    Cm = {d: float(c[dates == d][v[dates == d]].mean()) for d in np.unique(dates)}
    base = np.stack([c[i] - Cm[dates[i]] + L[dates[i]] for i in range(len(ids))])
    r_base = r2(o, base, snow)
    res = {'descripcion': ('Techo de la red de dos cabezas en el test 2025: w y delta ideales por '
                           'tesela (minimos cuadrados dentro de los limites de 8.13) con el nivel '
                           'observado de la fecha. Es un techo: usa la nieve observada.'),
           'limites': {'w': list(W_RANGE), 'delta_m': DELTA_MAX},
           'controles': {'climatologia_sola_tal_cual': [round(r_tal, 4), round(ref_tal, 4)],
                         'climatologia_nivel_corregido': [round(r_niv, 4), round(ref_niv, 4)],
                         'superados': True},
           'nivel_por_fecha': {d: {'L_obs_m': round(L[d], 4), 'Cm_m': round(Cm[d], 4)} for d in sorted(L)},
           'combinacion_con_nivel_observado_R2': round(r_base, 4),
           'techo': {}}
    print(f'\nCombinacion con el nivel observado (todos los pixeles volados): R2 {r_base:.4f}')

    for pix_nombre, pix in (('ajuste_en_pixeles_con_nieve', snow), ('ajuste_en_todos_los_validos', v)):
        res['techo'][pix_nombre] = {}
        for var, uw, ud in (('w_y_delta', True, True), ('solo_delta', False, True), ('solo_w', True, False)):
            pred = np.empty_like(c)
            ws, ds = [], []
            for i in range(len(ids)):
                d = dates[i]
                ai, yi, m = c[i] - Cm[d], o[i] - L[d], pix[i]
                w, dl = ajuste(ai[m], yi[m], uw, ud) if m.sum() > 10 else (1.0, 0.0)
                pred[i] = L[d] + w * ai + dl
                ws.append(w); ds.append(dl)
            ws, ds = np.array(ws), np.array(ds)
            r = r2(o, pred, snow)
            res['techo'][pix_nombre][var] = {
                'R2': round(r, 4), 'ganancia_sobre_combinacion': round(r - r_base, 4),
                'w_mediana': round(float(np.median(ws)), 4),
                'w_en_limite_pct': round(float(100 * np.mean((ws <= W_RANGE[0] + 1e-9) | (ws >= W_RANGE[1] - 1e-9))), 1),
                'delta_mediana_m': round(float(np.median(ds)), 4),
                'delta_en_limite_pct': round(float(100 * np.mean(np.abs(ds) >= DELTA_MAX - 1e-9)), 1)}
            t = res['techo'][pix_nombre][var]
            print(f"  {pix_nombre:28s} {var:10s} R2 {r:.4f} (+{r - r_base:.4f}); w mediana {t['w_mediana']:.2f}, "
                  f"en el limite {t['w_en_limite_pct']} %; delta mediana {t['delta_mediana_m']:+.3f} m, "
                  f"en el limite {t['delta_en_limite_pct']} %")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f'\nResultados: {a.out}')


if __name__ == '__main__':
    main()
