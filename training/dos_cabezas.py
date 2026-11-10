r"""
Red de dos cabezas: volcado de w y delta por tesela (protocolo 2026, 8.13)
==========================================================================

"Se guardan w y delta de cada tesela, para comprobar que no se quedan en los
limites." Lo llama main.py despues de cada evaluacion del modelo
resunetpp_dos_cabezas. No interviene en las metricas.
"""

import csv
import os

import numpy as np
import torch


def volcar_w_delta(model, loader, device, path: str, cerca: float = 0.02) -> dict:
    """Escribe un CSV con w, delta, L(d) y Cm(d) de cada tesela del loader.

    Args:
        model:  ResUNetPPDosCabezas con los pesos cargados
        loader: DataLoader de SnowDatasetEval con return_valid y niveles por
                fecha (lotes de 5 elementos: imagen, mascara, validez, ids,
                senales)
        device: dispositivo
        path:   CSV de salida
        cerca:  una tesela cuenta como "en el limite" si w o delta estan a
                menos de esta fraccion de la semianchura de su intervalo

    Returns:
        Resumen (minimo, mediana, maximo y fraccion de teselas en el limite).
    """
    filas = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) != 5:
                raise ValueError('volcar_w_delta espera lotes de 5 elementos '
                                 '(imagen, mascara, validez, ids, senales)')
            images, _, _, ids, signals = batch
            _, w, delta = model.predict_components(images.to(device), signals.to(device))
            for t, wi, di, si in zip(ids, w.cpu().numpy(), delta.cpu().numpy(), signals.numpy()):
                filas.append({'tile_id': str(t), 'date': str(t).split('_')[0],
                              'w': float(wi), 'delta_m': float(di),
                              'nivel_m': float(si[0]), 'clim_media_m': float(si[1])})

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=['tile_id', 'date', 'w', 'delta_m', 'nivel_m', 'clim_media_m'])
        wr.writeheader()
        for f in filas:
            wr.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v) for k, v in f.items()})

    w = np.array([f['w'] for f in filas])
    d = np.array([f['delta_m'] for f in filas])
    w_min, w_max = model.w_range
    tol_w = cerca * 0.5 * (w_max - w_min)
    tol_d = cerca * model.delta_max
    res = {
        'n_teselas': len(filas),
        'w': {'min': float(w.min()), 'mediana': float(np.median(w)), 'max': float(w.max()),
              'frac_en_limite': float(np.mean((w < w_min + tol_w) | (w > w_max - tol_w)))},
        'delta_m': {'min': float(d.min()), 'mediana': float(np.median(d)), 'max': float(d.max()),
                    'frac_en_limite': float(np.mean(np.abs(d) > model.delta_max - tol_d))},
    }
    print(f"  w     por tesela: min {res['w']['min']:.3f}  mediana {res['w']['mediana']:.3f}  "
          f"max {res['w']['max']:.3f}  | en el limite [{w_min}, {w_max}]: "
          f"{100 * res['w']['frac_en_limite']:.1f} % de las teselas")
    print(f"  delta por tesela: min {res['delta_m']['min']:+.3f}  mediana {res['delta_m']['mediana']:+.3f}  "
          f"max {res['delta_m']['max']:+.3f} m | en el limite (+-{model.delta_max} m): "
          f"{100 * res['delta_m']['frac_en_limite']:.1f} % de las teselas")
    print(f"  w y delta guardados en: {path}")
    return res
