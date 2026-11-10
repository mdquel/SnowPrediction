r"""
Red de dos cabezas: control de arranque con los datos reales (antes de entrenar)
================================================================================

Protocolo 2026, apartado 8.13: "Al empezar, w = 1 y delta = 0, de modo que la
salida es exactamente la combinacion". Este script lo comprueba con el modelo
recien construido (sin entrenar), las teselas reales y la tabla de niveles de
la semilla. Se ejecuta una vez por semilla ANTES de entrenar; si falla, no se
entrena.

Controles
---------
1. Arranque = combinacion. En las teselas de test, la salida del modelo sin
   entrenar es igual, pixel a pixel, a la combinacion (hybrid, de
   scripts/clim_v1_hybrid.py) calculada con las predicciones guardadas de la
   red B de la misma semilla; con la mejor epoca de B y con la ultima.
   Tolerancia 1e-4 m (float32 frente a float64).
2. w = 1 y delta = 0 en todas las teselas de test.
3. Perdida solo con climatologia. En validacion y en una muestra del
   entrenamiento, ningun pixel sin climatologia cuenta en la perdida. Se
   informa de la fraccion de pixeles validos que se excluye.

Informativo
-----------
Perdida de validacion al arrancar: es el error de la combinacion con el nivel
observado en 2024, y la referencia de la curva de entrenamiento (la red solo
aporta si baja de ahi).

Salida: analysis/dos_cabezas/control_inicio_s<semilla>.json

Uso (desde la raiz del repo)
----------------------------
    python scripts/dos_cabezas_control_inicio.py --config configs/dos_cabezas/resunetpp_dc1_s7.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_AQUI, '..'))
sys.path.insert(0, _AQUI)
import main as M  # noqa: E402
from clim_v1_hybrid import hybrid  # noqa: E402
from clim_v1_level_corrected import load  # noqa: E402
from data.dataset import SnowDataset, SnowDatasetEval, load_splits  # noqa: E402
from models.unet import build_model  # noqa: E402
from training.train import _val_epoch, get_device, get_loss_fn  # noqa: E402

TOL_M = 1e-4
MUESTRA_TRAIN = 20      # una de cada 20 teselas de entrenamiento para el control 3


def _comun(ds, config):
    cfg = config['data']
    ds.channel_indices = cfg.get('channel_indices')
    ds.norm_extended = cfg.get('norm_extended', True)
    ds.norm_version = cfg.get('norm_version')
    ds.extra_channels_dir = M._extra_dir(config)
    M._set_clim(ds, config)
    return ds


def ds_test(df, config, imgs, masks):
    """Como main.run_evaluate."""
    ds = _comun(SnowDatasetEval(df, imgs, masks, use_sce=False,
                                n_channels=config['model']['in_channels']), config)
    ds.return_valid = True
    M._set_date_levels(ds, config)
    return ds


def ds_perdida(df, config, imgs, masks, excluir=True):
    """Como main.run_train (sin aumento ni apagado de la climatologia)."""
    ds = _comun(SnowDataset(df, imgs, masks, use_sce=False, augment=False,
                            n_channels=config['model']['in_channels']), config)
    ds.return_valid = True
    M._set_date_levels(ds, config)
    if excluir:
        ds.loss_requires_channel = config['data'].get('loss_requires_channel')
    return ds


def main():
    ap = argparse.ArgumentParser(description='Control de arranque de la red de dos cabezas')
    ap.add_argument('--config', required=True)
    ap.add_argument('--b-dir', default='results/regen2024/sat_v2')
    ap.add_argument('--clim-npz',
                    default='results/regen2024/clim_v1/climatologia_sola/climatologia_sola_predictions.npz')
    ap.add_argument('--out-dir', default='analysis/dos_cabezas')
    a = ap.parse_args()

    config = M.load_config(a.config)
    config['data']['root'] = str(M._resolve(config['data']['root']))
    cfg_data, cfg_tr = config['data'], config['training']
    if config['model']['architecture'] != 'resunetpp_dos_cabezas':
        raise SystemExit('STOP: la config no es de la red de dos cabezas')
    seed = cfg_tr['seed']
    csv_path, imgs, masks = M._get_paths(config)
    train_df, val_df, test_df = load_splits(csv_path, source=cfg_data['source'],
                                            split_type=cfg_data['split_type'])

    torch.manual_seed(seed)
    device = get_device(cfg_tr['device'])
    model = build_model(config).to(device).eval()
    bs, nw = cfg_tr['batch_size'], cfg_tr['num_workers']
    fallos, res = [], {'config': a.config, 'semilla': seed, 'tolerancia_m': TOL_M}

    # --- controles 1 y 2: test, mejor epoca y ultima de la red B -------------------
    c_ids, c_p, c_o, c_v = load(a.clim_npz)
    valid = c_v & np.isfinite(c_o)
    pos = {t: i for i, t in enumerate(c_ids)}
    test = ds_test(test_df, config, imgs, masks)
    loader = DataLoader(test, batch_size=bs, shuffle=False, num_workers=nw)
    res['arranque_frente_a_combinacion'] = {}
    for epoca, e, last in (('mejor_epoca', '', False), ('ultima_epoca', '_last', True)):
        name = f'resunetpp_sat2_B_s{seed}'
        npz = os.path.join(a.b_dir, name, f'{name}{e}_predictions.npz')
        n_ids, n_p, _, _ = load(npz)
        hib, _ = hybrid(c_ids, c_p, n_ids, n_p, valid)
        M._set_date_levels(test, config, last=last)
        dif_max, n, w_all, d_all = 0.0, 0, [], []
        with torch.no_grad():
            for images, _, _, ids, signals in loader:
                hs, w, delta = model.predict_components(images.to(device), signals.to(device))
                hs = hs[:, 0].cpu().numpy().astype(np.float64)
                for t, h in zip(ids, hs):
                    dif_max = max(dif_max, float(np.abs(h - hib[pos[str(t)]]).max()))
                    n += 1
                w_all.append(w.cpu().numpy())
                d_all.append(delta.cpu().numpy())
        w_all, d_all = np.concatenate(w_all), np.concatenate(d_all)
        ok12 = bool(np.all(w_all == 1.0) and np.all(d_all == 0.0))
        res['arranque_frente_a_combinacion'][epoca] = {
            'teselas': n, 'diferencia_maxima_m': dif_max, 'w_igual_a_1_y_delta_igual_a_0': ok12}
        print(f'Control 1 ({epoca}): {n} teselas de test, diferencia maxima con la combinacion '
              f'{dif_max:.2e} m -> {"SUPERADO" if dif_max <= TOL_M else "NO SUPERADO"}')
        print(f'Control 2 ({epoca}): w = 1 y delta = 0 en todas -> {"SUPERADO" if ok12 else "NO SUPERADO"}')
        if n != len(c_ids):
            fallos.append(f'{epoca}: {n} teselas evaluadas y {len(c_ids)} en la combinacion')
        if dif_max > TOL_M:
            fallos.append(f'{epoca}: el arranque difiere de la combinacion en {dif_max:.3e} m')
        if not ok12:
            fallos.append(f'{epoca}: w o delta no arrancan en 1 y 0')

    # --- control 3: la perdida no cuenta pixeles sin climatologia -----------------------
    disp = cfg_data['channel_indices'].index(cfg_data['loss_requires_channel'])
    res['perdida_solo_con_climatologia'] = {}
    for nombre, df in (('validacion', val_df),
                       ('entrenamiento_muestra', train_df.iloc[::MUESTRA_TRAIN])):
        con = ds_perdida(df, config, imgs, masks, excluir=True)
        sin = ds_perdida(df, config, imgs, masks, excluir=False)
        mal = tot = dentro = 0
        for i in range(len(con)):
            x, _, v, _ = con[i]
            _, _, v0, _ = sin[i]
            mal += int(((v > 0) & (x[disp:disp + 1] <= 0)).sum())
            dentro += int(v.sum())
            tot += int(v0.sum())
        excl = 1.0 - dentro / max(tot, 1)
        res['perdida_solo_con_climatologia'][nombre] = {
            'teselas': len(con), 'pixeles_sin_climatologia_en_la_perdida': mal,
            'fraccion_de_pixeles_validos_excluida': round(excl, 6)}
        print(f'Control 3 ({nombre}): {len(con)} teselas, {mal} pixeles sin climatologia en la perdida, '
              f'{100 * excl:.2f} % de los pixeles validos excluido -> {"SUPERADO" if mal == 0 else "NO SUPERADO"}')
        if mal:
            fallos.append(f'{nombre}: {mal} pixeles sin climatologia cuentan en la perdida')

    # --- informativo: perdida de validacion al arrancar ---------------------------------
    val = ds_perdida(val_df, config, imgs, masks, excluir=True)
    criterion = get_loss_fn(cfg_tr['loss'], lambda_pearson=cfg_tr.get('lambda_pearson', 0.5))
    val_loss = _val_epoch(model, DataLoader(val, batch_size=bs, shuffle=False, num_workers=nw),
                          criterion, device)
    res['perdida_validacion_al_arrancar'] = round(float(val_loss), 6)
    print(f'Perdida de validacion al arrancar (combinacion con el nivel observado): {val_loss:.4f}')

    res['fallos'], res['superado'] = fallos, not fallos
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f'control_inicio_s{seed}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f'\nControl de arranque: {"SUPERADO" if not fallos else "NO SUPERADO"}. Resumen: {out}')
    if fallos:
        for f in fallos:
            print('  ' + f)
        raise SystemExit('STOP: no entrenar hasta resolverlo.')


if __name__ == '__main__':
    main()
