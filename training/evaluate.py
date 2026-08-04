import os
import json
import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import (compute_metrics, compute_naive_benchmark,
                           compute_spaef, compute_mspaef,
                           compute_detection_metrics, print_metrics)
from utils.visualization import plot_scatter, plot_predictions


def _predict_with_tta(model: torch.nn.Module,
                      images: torch.Tensor,
                      device: torch.device,
                      eastness_ch: int = 3) -> np.ndarray:
    """
    Prediccion con Test-Time Augmentation (TTA) horizontal.

    Promedia la prediccion original con la prediccion sobre la imagen
    volteada horizontalmente (Eastness negado). El flip H es fisicamente
    valido porque la asimetria E-O del snowpack es mas simetrica que la N-S.

    Args:
        model:       Modelo en modo eval con no_grad activo
        images:      Tensor (B, C, H, W)
        device:      Dispositivo de calculo
        eastness_ch: Indice del canal Eastness (defecto: 3)

    Returns:
        np.ndarray (B, 1, H, W) con predicciones promediadas
    """
    imgs_dev = images.to(device)

    # Prediccion original
    pred_orig = model(imgs_dev).cpu().numpy()  # (B,1,H,W)

    # Imagen H-flip: invertir eje W y negar Eastness
    imgs_flip = torch.flip(imgs_dev, dims=[3]).clone()
    imgs_flip[:, eastness_ch, :, :] = -imgs_flip[:, eastness_ch, :, :]

    # Prediccion flipped (y devolver a orientacion original)
    pred_flip = model(imgs_flip).cpu().numpy()
    pred_flip = pred_flip[:, :, :, ::-1].copy()  # flip W de vuelta

    return (pred_orig + pred_flip) / 2.0


def evaluate_model(model:       torch.nn.Module,
                   test_loader: DataLoader,
                   device:      torch.device,
                   config:      dict) -> dict:
    """
    Evaluacion completa del modelo sobre el conjunto de test.

    Genera metricas, scatter plot y mapas de prediccion/error.

    Args:
        model:       Modelo PyTorch con pesos cargados
        test_loader: DataLoader del conjunto de test (SnowDatasetEval)
        device:      Dispositivo de calculo
        config:      Configuracion completa del YAML
                     Soporta config['evaluation']['tta'] = true para TTA

    Returns:
        Diccionario con metricas (MAE, RMSE, R2, NSE, Bias)
    """
    exp         = config['experiment']['name']
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # TTA: opcional desde el YAML  (evaluation.tta: true)
    use_tta = config.get('evaluation', {}).get('tta', False)
    if use_tta:
        print(f"\nEvaluando: {exp}  [TTA activo] ...")
    else:
        print(f"\nEvaluando: {exp} ...")

    model.eval()
    # snow-only (target > 1cm): metricas principales, comparables con el paper
    all_preds, all_targets = [], []
    # full-domain (todos los pixeles validos, incl. suelo desnudo): mide
    # tambien los falsos positivos que la evaluacion snow-only no ve
    all_preds_full, all_targets_full = [], []
    spaef_per_tile  = []
    mspaef_per_tile = []
    tiles_dropped = 0   # NaN en SPAEF o MSPAEF (tiles degenerados)
    tiles_small   = 0   # < 10 pixeles con nieve
    s_imgs, s_masks, s_preds, s_ids = [], [], [], []

    #mdquel - Volcado opcional de predicciones por tile (para analisis posterior)
    save_preds = config.get('evaluation', {}).get('save_predictions', False)
    dump_ids, dump_preds, dump_targets, dump_valids = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test"):

            # 2/3 elementos (legacy) o 4 (con mascara de validez)
            valids = None
            if len(batch) == 4:
                images, masks, valids, ids = batch
            elif len(batch) == 3:
                images, masks, ids = batch
            else:
                images, masks = batch
                ids = [''] * images.size(0)

            if use_tta:
                outputs = _predict_with_tta(model, images, device)
            else:
                outputs = model(images.to(device)).cpu().numpy()  # (B,1,H,W)
            targets = masks.cpu().numpy()                          # (B,1,H,W)
            valids_np = valids.cpu().numpy() if valids is not None else None
            
            #mdquel - guadarmos las predicciones
            if save_preds:
                dump_ids.extend(list(ids))
                dump_preds.append(outputs[:, 0].astype(np.float32))
                dump_targets.append(targets[:, 0].astype(np.float32))
                if valids_np is not None:
                    dump_valids.append(valids_np[:, 0].astype(np.float32))            

            # Guardar ejemplos para visualizacion (maximo 3 tiles)
            if len(s_imgs) < 3:
                s_imgs.append(images.numpy())
                s_masks.append(targets)
                s_preds.append(outputs)
                s_ids.extend(list(ids)[:3 - len(s_ids)])

            tgt_flat = targets.squeeze(1).flatten()
            out_flat = outputs.squeeze(1).flatten()

            # snow-only: pixeles con nieve real (> 0.01 m)
            snow = tgt_flat > 0.01
            if snow.sum() > 0:
                all_preds.extend(out_flat[snow].tolist())
                all_targets.extend(tgt_flat[snow].tolist())

            # full-domain: todos los pixeles con dato LiDAR (incl. suelo desnudo)
            if valids_np is not None:
                vflat = valids_np.squeeze(1).flatten() > 0.5
                if vflat.sum() > 0:
                    all_preds_full.extend(out_flat[vflat].tolist())
                    all_targets_full.extend(tgt_flat[vflat].tolist())

            # SPAEF y MSPAEF por tile (sobre pixeles con nieve de cada tile);
            # solo se cuentan si AMBOS son validos -> mismo conjunto de tiles
            for b in range(targets.shape[0]):
                tgt_tile = targets[b, 0].flatten()
                out_tile = outputs[b, 0].flatten()
                v = tgt_tile > 0.01
                if v.sum() < 10:
                    tiles_small += 1
                    continue
                spaef_val = compute_spaef(tgt_tile[v], out_tile[v])
                mspaef_val = compute_mspaef(tgt_tile[v], out_tile[v])
                if np.isnan(spaef_val) or np.isnan(mspaef_val):
                    tiles_dropped += 1
                    continue
                spaef_per_tile.append(spaef_val)
                mspaef_per_tile.append(mspaef_val)

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    metrics = compute_metrics(y_true, y_pred)

    # SPAEF y MSPAEF sobre el MISMO conjunto de tiles; std con ddof=1
    # (consistente con la std entre seeds del paper) y conteo de descartes.
    def _std1(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    if spaef_per_tile:
        metrics['SPAEF']         = round(float(np.mean(spaef_per_tile)), 4)
        metrics['SPAEF_std']     = round(_std1(spaef_per_tile), 4)
        metrics['MSPAEF']        = round(float(np.mean(mspaef_per_tile)), 4)
        metrics['MSPAEF_std']    = round(_std1(mspaef_per_tile), 4)
        metrics['SPAEF_n_tiles'] = len(spaef_per_tile)
        metrics['MSPAEF_n_tiles'] = len(mspaef_per_tile)  # mismo conjunto
        metrics['tiles_dropped_nan'] = tiles_dropped
        metrics['tiles_too_small']   = tiles_small
    else:
        metrics['SPAEF'] = float('nan')
        metrics['MSPAEF'] = float('nan')

    # full-domain + deteccion (solo si el dataset aporto la mascara de validez)
    if all_targets_full:
        yf_true = np.array(all_targets_full)
        yf_pred = np.array(all_preds_full)
        mfull = compute_metrics(yf_true, yf_pred)
        metrics['R2_full']   = mfull['R2']
        metrics['MAE_full']  = mfull['MAE']
        metrics['RMSE_full'] = mfull['RMSE']
        metrics['Bias_full'] = mfull['Bias']
        metrics.update(compute_detection_metrics(yf_true, yf_pred, thr=0.01))
        metrics['n_pixels_full'] = int(len(yf_true))

    print_metrics(metrics, title=f"Test Set - {exp}")

    # --- Guardar metricas en disco ---
    metrics_path = os.path.join(results_dir, f"{exp}_metrics.json")
    metrics_out  = {
        'experiment':  exp,
        'timestamp':   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_pixels':    int(len(y_true)),
        'tta':         use_tta,
        **metrics,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metricas guardadas en: {metrics_path}")

    #mdquel - guardado de las predicciones
    if save_preds and dump_ids:
        preds_path = os.path.join(results_dir, f"{exp}_predictions.npz")
        arrays = {
            'tile_ids': np.array(dump_ids),
            'preds':    np.concatenate(dump_preds,   axis=0),
            'targets':  np.concatenate(dump_targets, axis=0),
        }
        if dump_valids:
            arrays['valids'] = np.concatenate(dump_valids, axis=0)
        np.savez_compressed(preds_path, **arrays)
        print(f"  Predicciones guardadas en: {preds_path}")

    # Figuras
    scatter_path = os.path.join(results_dir, f"{exp}_scatter.png")
    maps_path    = os.path.join(results_dir, f"{exp}_maps.png")

    plot_scatter(y_true, y_pred, metrics, save_path=scatter_path)

    if s_imgs:
        plot_predictions(
            images=np.concatenate(s_imgs,  axis=0),
            masks= np.concatenate(s_masks, axis=0),
            preds= np.concatenate(s_preds, axis=0),
            tile_ids=s_ids[:3],
            n_samples=3,
            save_path=maps_path,
        )

    return metrics


def run_naive_benchmark(train_df:  pd.DataFrame,
                         test_df:   pd.DataFrame,
                         masks_dir: str) -> dict:
    """
    Calcula el benchmark naive: predice siempre la media del train.

    Args:
        train_df:  DataFrame del split de entrenamiento
        test_df:   DataFrame del split de test
        masks_dir: Directorio con los .npy de mascaras

    Returns:
        Diccionario con metricas del predictor naive
    """
    def _load_values(df: pd.DataFrame) -> np.ndarray:
        values = []
        for _, row in df.iterrows():
            path = os.path.join(masks_dir, row['tile_id'])
            if os.path.exists(path):
                m     = np.load(path).flatten().astype(float)
                valid = m[(m != -9999) & np.isfinite(m) & (m >= 0)]
                values.extend(valid.tolist())
        return np.array(values)

    print("Calculando benchmark naive...")
    train_vals = _load_values(train_df)
    test_vals  = _load_values(test_df)

    metrics = compute_naive_benchmark(train_vals, test_vals)
    print_metrics(metrics, title="Benchmark Naive (Media Train)")
    return metrics


# ----------------------------------------------------------------------
# Utilidad: compilar todos los _metrics.json en una tabla resumen
# ----------------------------------------------------------------------

def compile_results_table(results_root: str,
                           save_path: str = None) -> pd.DataFrame:
    """
    Recorre results_root buscando *_metrics.json y construye una tabla
    comparativa de todos los experimentos.

    Args:
        results_root: Directorio raiz donde estan las carpetas de resultados.
        save_path:    Si se proporciona, guarda la tabla como CSV en esa ruta.

    Returns:
        DataFrame con una fila por experimento.
    """
    rows = []
    for dirpath, _, filenames in os.walk(results_root):
        for fname in filenames:
            if fname.endswith('_metrics.json'):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    rows.append(data)
                except Exception as e:
                    print(f"  [WARN] No se pudo leer {fpath}: {e}")

    if not rows:
        print("No se encontraron archivos _metrics.json.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ordenar por R2 descendente
    if 'R2' in df.columns:
        df = df.sort_values('R2', ascending=False).reset_index(drop=True)

    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"Tabla resumen guardada en: {save_path}")

    return df
