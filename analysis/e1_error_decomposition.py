r"""
E1 - Descomposicion del error de prediccion (Murphy, 1988).

Objetivo cientifico
-------------------
Determinar SI el error de los modelos de HS es un error de MAGNITUD
(el modelo sabe donde se acumula la nieve pero no cuanta hay) o un error
de PATRON (el modelo no sabe donde se acumula).

El MSE se descompone exactamente en tres terminos ortogonales:

    MSE = (mu_s - mu_o)^2  +  (sigma_s - sigma_o)^2  +  2*sigma_s*sigma_o*(1 - rho)
          \_____T1_____/      \________T2________/     \__________T3__________/
             sesgo               dispersion                   patron

    T1 (sesgo)      : error de nivel medio      -> "cuanta nieve"
    T2 (dispersion) : error de amplitud/rango   -> "cuanto contraste"
    T3 (patron)     : error de co-localizacion  -> "donde esta la nieve"

Criterio de decision (fijado A PRIORI, antes de ver resultados):

    T1+T2 > 50% del MSE en split temporal, y esa fraccion cae en el
    espacial            -> hipotesis CONFIRMADA (fallo de escala)
    T3 domina en ambos  -> hipotesis REFUTADA  (fallo de patron)
    T1+T2 entre 40-50%  -> NO CONCLUYENTE, ampliar con leave-one-year-out

Metricas complementarias
------------------------
    R2_raw     : R2 sobre los campos tal cual
    R2_pattern : R2 sobre campos estandarizados por tile (z-score).
                 Elimina artificialmente los errores de escala. Si sube
                 mucho respecto a R2_raw -> el patron estaba bien.
    R2_recal   : R2 tras recalibrar con UN escalar por fecha
                 (alpha_d = mu_o(d) / mu_s(d)). Se calcula por FECHA y no
                 por tile para no filtrar informacion del propio tile.
                 Si recupera gran parte del rendimiento -> al modelo solo
                 le falta saber "cuanta nieve hay este anyo".

Uso
---
    python analysis/e1_error_decomposition.py \
        --npz results/.../exp_predictions.npz \
        --label "spatial lambda=0"

    # varios a la vez (comparativa temporal vs espacial):
    python analysis/e1_error_decomposition.py \
        --npz results/A_predictions.npz results/B_predictions.npz \
        --label "temporal lambda=0" "spatial lambda=0" \
        --out analysis/e1_results.json \
        --fig analysis/e1_decomposition.png

Notas de implementacion
-----------------------
* Solo se usan pixeles validos. Por defecto se replica el criterio
  "snow-only" del pipeline (target > 0.01 m), que es el usado para las
  metricas principales del paper. Con --domain full se usa la mascara de
  validez completa (incluye suelo desnudo).
* Se descartan tiles con menos de MIN_VALID_PIXELS pixeles validos o con
  varianza nula (SPAEF/correlacion indefinidos).
* Se verifica que T1+T2+T3 reconstruye el MSE. Si el error relativo
  supera RECON_TOL el script AVISA: indicaria un bug, no un resultado.
"""

import argparse
import json
import os

import numpy as np

MIN_VALID_PIXELS = 10      # mismo umbral que evaluate.py
RECON_TOL = 1e-4           # tolerancia relativa del control de sanidad
SNOW_THRESHOLD = 0.01      # m, mismo umbral snow-only que el pipeline


# ----------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------

def load_npz(path):
    """Carga un .npz de predicciones generado por evaluate_model()."""
    d = np.load(path)
    required = ('tile_ids', 'preds', 'targets')
    for k in required:
        if k not in d:
            raise KeyError(f"{path}: falta la clave '{k}'")
    out = {
        'tile_ids': d['tile_ids'],
        'preds':    d['preds'].astype(np.float64),
        'targets':  d['targets'].astype(np.float64),
    }
    out['valids'] = d['valids'].astype(np.float64) if 'valids' in d else None
    return out


def date_from_tile_id(tile_id):
    """Extrae la fecha (YYYYMMDD) del identificador de tile.

    Formato esperado: '20210202_lidar_tile_0_128.npy' -> '20210202'
    """
    s = str(tile_id)
    head = s.split('_')[0]
    return head if (len(head) == 8 and head.isdigit()) else 'unknown'


def tile_mask(target, valid, domain):
    """Mascara de pixeles a considerar en un tile."""
    if domain == 'full':
        if valid is None:
            raise ValueError("--domain full requiere la clave 'valids' en el .npz")
        return valid > 0.5
    return target > SNOW_THRESHOLD          # snow-only (por defecto)


# ----------------------------------------------------------------------
# Descomposicion de Murphy
# ----------------------------------------------------------------------

def decompose_tile(obs, sim):
    """Descompone el MSE de un tile en sesgo, dispersion y patron.

    Devuelve None si el tile es degenerado (varianza nula en obs o sim).
    """
    mu_o, mu_s = obs.mean(), sim.mean()
    sd_o, sd_s = obs.std(), sim.std()

    if sd_o < 1e-12 or sd_s < 1e-12:
        return None                          # correlacion indefinida

    rho = float(np.corrcoef(obs, sim)[0, 1])
    if not np.isfinite(rho):
        return None

    t1 = (mu_s - mu_o) ** 2                  # sesgo
    t2 = (sd_s - sd_o) ** 2                  # dispersion
    t3 = 2.0 * sd_s * sd_o * (1.0 - rho)     # patron
    mse = float(np.mean((sim - obs) ** 2))

    return {
        'mse': mse, 't1': float(t1), 't2': float(t2), 't3': float(t3),
        'rho': rho, 'mu_o': float(mu_o), 'mu_s': float(mu_s),
        'sd_o': float(sd_o), 'sd_s': float(sd_s), 'n': int(obs.size),
    }


# ----------------------------------------------------------------------
# Metricas globales
# ----------------------------------------------------------------------

def r2(obs, sim):
    """Coeficiente de determinacion sobre vectores concatenados."""
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def analyse(data, domain='snow', label='experiment'):
    """Analisis completo de un experimento."""
    tile_ids = data['tile_ids']
    preds, targets, valids = data['preds'], data['targets'], data['valids']
    n_tiles = len(tile_ids)

    per_tile, skipped = [], 0
    obs_all, sim_all = [], []                 # para R2 crudo
    obs_z, sim_z = [], []                     # para R2 de patron
    by_date = {}                              # fecha -> (sum_obs, sum_sim, n)

    for i in range(n_tiles):
        tgt, prd = targets[i], preds[i]
        val = valids[i] if valids is not None else None
        m = tile_mask(tgt, val, domain)

        if m.sum() < MIN_VALID_PIXELS:
            skipped += 1
            continue

        obs, sim = tgt[m], prd[m]
        dec = decompose_tile(obs, sim)
        if dec is None:
            skipped += 1
            continue

        dec['tile_id'] = str(tile_ids[i])
        dec['date'] = date_from_tile_id(tile_ids[i])
        per_tile.append(dec)

        obs_all.append(obs)
        sim_all.append(sim)

        # campos estandarizados por tile -> patron puro
        obs_z.append((obs - dec['mu_o']) / dec['sd_o'])
        sim_z.append((sim - dec['mu_s']) / dec['sd_s'])

        d = dec['date']
        so, ss, nn = by_date.get(d, (0.0, 0.0, 0))
        by_date[d] = (so + obs.sum(), ss + sim.sum(), nn + obs.size)

    if not per_tile:
        raise RuntimeError(f"{label}: ningun tile valido tras el filtrado")

    # --- agregacion de la descomposicion (ponderada por n pixeles) ---
    w = np.array([t['n'] for t in per_tile], dtype=np.float64)
    w = w / w.sum()
    mse = float(np.sum(w * np.array([t['mse'] for t in per_tile])))
    t1 = float(np.sum(w * np.array([t['t1'] for t in per_tile])))
    t2 = float(np.sum(w * np.array([t['t2'] for t in per_tile])))
    t3 = float(np.sum(w * np.array([t['t3'] for t in per_tile])))

    # --- control de sanidad: T1+T2+T3 debe reconstruir el MSE ---
    recon = t1 + t2 + t3
    rel_err = abs(recon - mse) / mse if mse > 0 else 0.0
    recon_ok = rel_err < RECON_TOL

    # --- metricas complementarias ---
    obs_cat = np.concatenate(obs_all)
    sim_cat = np.concatenate(sim_all)
    r2_raw = r2(obs_cat, sim_cat)
    r2_pattern = r2(np.concatenate(obs_z), np.concatenate(sim_z))

    # recalibracion: UN escalar por fecha (alpha_d = mu_o(d)/mu_s(d))
    alphas = {}
    for d, (so, ss, nn) in by_date.items():
        mo, ms = so / nn, ss / nn
        alphas[d] = float(mo / ms) if abs(ms) > 1e-12 else 1.0

    sim_recal = []
    for t, sim in zip(per_tile, sim_all):
        sim_recal.append(sim * alphas.get(t['date'], 1.0))
    r2_recal = r2(obs_cat, np.concatenate(sim_recal))

    return {
        'label': label,
        'domain': domain,
        'n_tiles_used': len(per_tile),
        'n_tiles_skipped': skipped,
        'n_dates': len(by_date),
        'mse': mse,
        'T1_bias': t1,
        'T2_dispersion': t2,
        'T3_pattern': t3,
        'T1_pct': 100.0 * t1 / mse,
        'T2_pct': 100.0 * t2 / mse,
        'T3_pct': 100.0 * t3 / mse,
        'scale_pct': 100.0 * (t1 + t2) / mse,     # T1+T2 = error de escala
        'reconstruction_ok': bool(recon_ok),
        'reconstruction_rel_error': float(rel_err),
        'R2_raw': r2_raw,
        'R2_pattern': r2_pattern,
        'R2_recalibrated': r2_recal,
        'mean_rho': float(np.mean([t['rho'] for t in per_tile])),
        'mean_bias': float(np.mean([t['mu_s'] - t['mu_o'] for t in per_tile])),
        'scale_factors_by_date': alphas,
    }


# ----------------------------------------------------------------------
# Presentacion
# ----------------------------------------------------------------------

def verdict(res):
    """Aplica el criterio de decision fijado a priori."""
    s = res['scale_pct']
    if s > 50.0:
        return 'ESCALA domina (>50%): compatible con la hipotesis patron-si/magnitud-no'
    if s < 40.0:
        return 'PATRON domina (<40% escala): hipotesis NO respaldada en este experimento'
    return 'NO CONCLUYENTE (escala 40-50%): ampliar evidencia (leave-one-year-out)'


def report(results):
    for r in results:
        print()
        print('=' * 68)
        print(f"  {r['label']}   [dominio: {r['domain']}]")
        print('=' * 68)
        print(f"  Tiles usados / descartados : {r['n_tiles_used']} / {r['n_tiles_skipped']}")
        print(f"  Fechas distintas           : {r['n_dates']}")
        print(f"  MSE total                  : {r['mse']:.4f} m^2")
        print()
        print('  Descomposicion de Murphy:')
        print(f"    T1 sesgo       : {r['T1_bias']:.4f}  ({r['T1_pct']:5.1f} %)")
        print(f"    T2 dispersion  : {r['T2_dispersion']:.4f}  ({r['T2_pct']:5.1f} %)")
        print(f"    T3 patron      : {r['T3_pattern']:.4f}  ({r['T3_pct']:5.1f} %)")
        print(f"    ESCALA (T1+T2) : {r['scale_pct']:5.1f} %")
        print()
        print(f"  R2 crudo                   : {r['R2_raw']:.4f}")
        print(f"  R2 patron (z-score)        : {r['R2_pattern']:.4f}")
        print(f"  R2 recalibrado por fecha   : {r['R2_recalibrated']:.4f}")
        print(f"  Correlacion media por tile : {r['mean_rho']:.4f}")
        print(f"  Sesgo medio (sim - obs)    : {r['mean_bias']:+.4f} m")
        print()
        flag = 'OK' if r['reconstruction_ok'] else 'AVISO - REVISAR'
        print(f"  Control T1+T2+T3 == MSE    : {flag} (err rel {r['reconstruction_rel_error']:.2e})")
        if not r['reconstruction_ok']:
            print('    >> La descomposicion NO reconstruye el MSE: es un BUG, no un resultado.')
        print()
        print(f"  Veredicto: {verdict(r)}")

    if len(results) > 1:
        print()
        print('=' * 68)
        print('  COMPARATIVA')
        print('=' * 68)
        print(f"  {'experimento':<28} {'escala%':>9} {'patron%':>9} {'R2':>8} {'R2 recal':>10}")
        for r in results:
            print(f"  {r['label']:<28} {r['scale_pct']:>8.1f}% {r['T3_pct']:>8.1f}% "
                  f"{r['R2_raw']:>8.3f} {r['R2_recalibrated']:>10.3f}")
        print()
        print('  Lectura: si el % de escala es alto en el split temporal y')
        print('  claramente menor en el espacial, el techo predictivo esta en la')
        print('  extrapolacion de MAGNITUD entre anyos, no en el patron espacial.')


def make_figure(results, path):
    """Barras apiladas con la descomposicion porcentual del MSE."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [aviso] matplotlib no disponible: se omite la figura')
        return

    labels = [r['label'] for r in results]
    t1 = [r['T1_pct'] for r in results]
    t2 = [r['T2_pct'] for r in results]
    t3 = [r['T3_pct'] for r in results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 3.2, 4.6))
    b1 = ax.bar(x, t1, width=0.55, label='T1 sesgo (nivel medio)', color='#c44e52')
    b2 = ax.bar(x, t2, width=0.55, bottom=t1, label='T2 dispersion (amplitud)', color='#dd8452')
    bottom3 = [a + b for a, b in zip(t1, t2)]
    b3 = ax.bar(x, t3, width=0.55, bottom=bottom3, label='T3 patron (co-localizacion)', color='#4c72b0')

    for i, r in enumerate(results):
        ax.text(i, 102, f"escala {r['scale_pct']:.0f}%", ha='center', fontsize=9)

    ax.set_ylabel('Contribucion al MSE (%)')
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(50, color='grey', ls='--', lw=0.8)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title('Descomposicion del MSE (Murphy, 1988)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"  Figura guardada en: {path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='E1 - descomposicion del error (Murphy 1988)')
    p.add_argument('--npz', nargs='+', required=True,
                   help='uno o mas .npz de predicciones')
    p.add_argument('--label', nargs='*', default=None,
                   help='etiquetas legibles, en el mismo orden que --npz')
    p.add_argument('--domain', choices=['snow', 'full'], default='snow',
                   help="'snow' = pixeles con nieve (>1cm, criterio del paper); "
                        "'full' = todos los pixeles validos")
    p.add_argument('--out', default=None, help='ruta del JSON de salida')
    p.add_argument('--fig', default=None, help='ruta de la figura PNG')
    args = p.parse_args()

    labels = args.label if args.label else [os.path.basename(f) for f in args.npz]
    if len(labels) != len(args.npz):
        raise SystemExit('--label debe tener tantos elementos como --npz')

    results = []
    for path, label in zip(args.npz, labels):
        print(f"Procesando: {path}")
        results.append(analyse(load_npz(path), domain=args.domain, label=label))

    report(results)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Resultados guardados en: {args.out}")

    if args.fig:
        make_figure(results, args.fig)


if __name__ == '__main__':
    main()
