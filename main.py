"""
Snow Depth Prediction - Articulo 1
===================================
Punto de entrada principal del proyecto.

Uso:
    # Entrenar
    python main.py --config configs/unet_v5_5m_mae.yaml --mode train

    # Evaluar (requiere modelo entrenado)
    python main.py --config configs/unet_v5_5m_mae.yaml --mode evaluate

    # Entrenar y evaluar en secuencia
    python main.py --config configs/unet_v5_5m_mae.yaml --mode both

    # Ablation: solo canal SCE
    python main.py --config configs/unet_v5_5m_mae_sce.yaml --mode both
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import SnowDataset, SnowDatasetEval, load_splits

# Raiz del repositorio (directorio donde vive este archivo)
REPO_ROOT = Path(__file__).resolve().parent


def _resolve(path: str) -> Path:
    """Resuelve paths relativos desde la raiz del repositorio."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
from models.unet import build_model
from training.train import train_model, get_device
from training.evaluate import evaluate_model, run_naive_benchmark


# ----------------------------------------------------------------------
# Carga de configuracion
# ----------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Helpers de rutas
# ----------------------------------------------------------------------

def _get_paths(config: dict) -> tuple:
    cfg       = config['data']
    root      = cfg['root']
    csv_path  = os.path.join(root, cfg['csv_file'])
    imgs_dir  = os.path.join(root, cfg['images_dir'])
    masks_dir = os.path.join(root, cfg['masks_dir'])
    return csv_path, imgs_dir, masks_dir


# ----------------------------------------------------------------------
# Modos de ejecucion
# ----------------------------------------------------------------------

def run_train(config: dict):
    csv_path, imgs_dir, masks_dir = _get_paths(config)
    cfg_data  = config['data']
    cfg_tr    = config['training']
    use_sce   = cfg_data.get('use_sce', False)

    train_df, val_df, _ = load_splits(
        csv_path,
        source=cfg_data['source'],
        split_type=cfg_data['split_type'],
    )

    # Opcion: usar train+val como entrenamiento (maximizar datos)
    use_all_pretest = cfg_data.get('use_all_pretest', False)
    if use_all_pretest:
        combined = pd.concat([train_df, val_df]).reset_index(drop=True)
        print(f"  MODO use_all_pretest: train={len(train_df)} + val={len(val_df)} "
              f"-> {len(combined)} tiles de entrenamiento (val=monitor solo)")
        train_df = combined

    use_aug        = cfg_data.get('augmentation', False)
    aug_mode       = cfg_data.get('augmentation_mode', 'hv')
    n_channels     = config.get('model', {}).get('in_channels', None)
    channel_indices = cfg_data.get('channel_indices', None)

    train_ds = SnowDataset(train_df, imgs_dir, masks_dir,
                           use_sce=use_sce, augment=use_aug,
                           n_channels=n_channels)
    norm_extended = cfg_data.get('norm_extended', True)
    norm_version  = cfg_data.get('norm_version', None)
    masked_loss   = cfg_tr.get('masked_loss', False)
    train_ds.augment_mode   = aug_mode
    train_ds.channel_indices = channel_indices
    train_ds.norm_extended   = norm_extended
    train_ds.norm_version    = norm_version
    train_ds.return_valid    = masked_loss
    val_ds   = SnowDataset(val_df,   imgs_dir, masks_dir,
                           use_sce=use_sce, augment=False,
                           n_channels=n_channels)
    val_ds.channel_indices   = channel_indices
    val_ds.norm_extended     = norm_extended
    val_ds.norm_version      = norm_version
    val_ds.return_valid      = masked_loss

    # --- Senales de escala externas (E4) ---
    # Escalares por fecha que alimentan la cabeza de escala del modelo dual.
    # Si la config no las declara, los datasets no las cargan y el
    # comportamiento es el de siempre.
    sig_csv = cfg_data.get('scale_signals_csv')
    sig_cols = cfg_data.get('scale_signal_cols')
    if sig_csv and sig_cols:
        print(f"Senales de escala: {sig_cols}  <- {sig_csv}")
        train_ds.load_scale_signals(sig_csv, sig_cols)
        val_ds.load_scale_signals(sig_csv, sig_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_tr['batch_size'],
        shuffle=True,
        num_workers=cfg_tr['num_workers'],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_tr['batch_size'],
        shuffle=False,
        num_workers=cfg_tr['num_workers'],
        pin_memory=True,
    )

    # Re-fijar semilla justo antes de inicializar el modelo para
    # garantizar reproducibilidad independientemente del orden de
    # operaciones previas (dataloaders, splits, etc.)
    _seed = cfg_tr.get('seed', None)
    if _seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(_seed)
        _np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

    model = build_model(config)
    train_model(model, train_loader, val_loader, config)


def run_evaluate(config: dict):
    csv_path, imgs_dir, masks_dir = _get_paths(config)
    cfg_data  = config['data']
    cfg_tr    = config['training']
    cfg_out   = config['output']
    use_sce   = cfg_data.get('use_sce', False)

    train_df, _, test_df = load_splits(
        csv_path,
        source=cfg_data['source'],
        split_type=cfg_data['split_type'],
    )

    n_channels      = config.get('model', {}).get('in_channels', None)
    channel_indices = cfg_data.get('channel_indices', None)
    test_ds = SnowDatasetEval(test_df, imgs_dir, masks_dir, use_sce=use_sce,
                              n_channels=n_channels)
    test_ds.channel_indices = channel_indices
    test_ds.norm_extended   = cfg_data.get('norm_extended', True)
    test_ds.norm_version    = cfg_data.get('norm_version', None)
    # Devolver la mascara de validez para calcular metricas full-domain y de
    # deteccion (falsos positivos) ademas de las snow-only.
    test_ds.return_valid    = True

    sig_csv = cfg_data.get('scale_signals_csv')
    sig_cols = cfg_data.get('scale_signal_cols')
    if sig_csv and sig_cols:
        print(f"Senales de escala: {sig_cols}  <- {sig_csv}")
        test_ds.load_scale_signals(sig_csv, sig_cols)    
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg_tr['batch_size'],
        shuffle=False,
        num_workers=cfg_tr['num_workers'],
    )

    # Cargar modelo
    device     = get_device(cfg_tr['device'])
    model      = build_model(config).to(device)
    model_path = os.path.join(cfg_out['models_dir'],
                              f"{cfg_out['model_name']}.pth")

    if not os.path.exists(model_path):
        print(f"\nERROR: No se encontro el modelo en:\n  {model_path}")
        print("Ejecuta primero con --mode train\n")
        return

    # Evaluar checkpoint best (mejor val loss)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Pesos cargados desde: {model_path}")
    evaluate_model(model, test_loader, device, config)

    # Evaluar checkpoint last (ultima epoca) si existe
    last_model_path = model_path.replace('.pth', '_last.pth')
    if os.path.exists(last_model_path):
        print(f"\n--- Evaluando checkpoint ultima epoca ---")
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        print(f"Pesos cargados desde: {last_model_path}")
        config_last = {**config, 'experiment': {'name': config['experiment']['name'] + '_last'}}
        evaluate_model(model, test_loader, device, config_last)

    # Benchmark naive como referencia
    run_naive_benchmark(train_df, test_df, masks_dir)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Snow Depth Prediction con U-Net'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Ruta al archivo YAML de configuracion (ej: configs/unet_v5_5m_mae.yaml)',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate', 'both'],
        help='Modo de ejecucion: train | evaluate | both  (default: train)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semilla aleatoria global (reproducibilidad)',
    )
    args   = parser.parse_args()
    config = load_config(args.config)

    # Semilla aleatoria (--seed sobreescribe config si existe)
    seed = args.seed if args.seed is not None else config.get('training', {}).get('seed', None)
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        config['training']['seed'] = seed   # propagar al historial

    # Resolver rutas relativas contra la raiz del repositorio
    config['data']['root']          = str(_resolve(config['data']['root']))
    config['output']['models_dir']  = str(_resolve(config['output']['models_dir']))
    config['output']['results_dir'] = str(_resolve(config['output']['results_dir']))

    print("\n" + "=" * 60)
    print(f"  Experimento : {config['experiment']['name']}")
    print(f"  Modo        : {args.mode}")
    print(f"  Config      : {args.config}")
    if seed is not None:
        print(f"  Semilla     : {seed}")
    print("=" * 60)

    if args.mode in ('train', 'both'):
        run_train(config)

    if args.mode in ('evaluate', 'both'):
        run_evaluate(config)


if __name__ == '__main__':
    main()
