r"""
Referencia: la climatologia sola como prediccion (paso 2; sin entrenar)
=======================================================================

Evalua el canal de climatologia (clim_v1) como si fuera la salida de un
modelo, con el MISMO codigo de evaluacion (training/evaluate.py), las mismas
teselas de test y los mismos pixeles que la red. Responde a la pregunta del
paso 2: ¿la red con climatologia supera a la climatologia?

* La prediccion de cada pixel es su climatologia (temporadas 2021-2023).
* Donde no hay climatologia (clim_disp = 0) la prediccion es 0 m. En el test
  la cobertura es del 97-100 % (analysis/clim_v1_canales.json); se declara.
* Sin nivel corregido: es la climatologia tal cual, como la red tal cual.

Uso (desde la raiz del repo)
----------------------------
    python scripts/eval_clim_only.py --config configs/clim_v1/resunetpp_clim1_Bclim_s7.yaml
"""

import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import main as M  # noqa: E402
from data.dataset import SnowDatasetEval, load_splits  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import get_device  # noqa: E402


class ClimOnly(nn.Module):
    """Devuelve el canal de climatologia de la entrada como prediccion."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, *args, **kwargs):
        return x[:, self.k:self.k + 1]


def main():
    ap = argparse.ArgumentParser(description='Climatologia sola como prediccion')
    ap.add_argument('--config', required=True)
    ap.add_argument('--name', default='climatologia_sola')
    a = ap.parse_args()

    config = M.load_config(a.config)
    config['data']['root'] = str(M._resolve(config['data']['root']))
    out = os.path.join(os.path.dirname(config['output']['results_dir']), a.name)
    config['output']['results_dir'] = str(M._resolve(out))
    config['experiment'] = {'name': a.name}

    csv_path, imgs_dir, masks_dir = M._get_paths(config)
    cfg_data = config['data']
    _, _, test_df = load_splits(csv_path, source=cfg_data['source'],
                                split_type=cfg_data['split_type'])
    idx = cfg_data['channel_indices']
    k = idx.index(cfg_data['clim_channels'][0])

    ds = SnowDatasetEval(test_df, imgs_dir, masks_dir, use_sce=False,
                         n_channels=config['model']['in_channels'])
    ds.channel_indices = idx
    ds.norm_extended = cfg_data.get('norm_extended', True)
    ds.norm_version = cfg_data.get('norm_version', None)
    ds.extra_channels_dir = M._extra_dir(config)
    M._set_clim(ds, config)
    ds.return_valid = True

    loader = DataLoader(ds, batch_size=config['training']['batch_size'], shuffle=False,
                        num_workers=config['training']['num_workers'])
    device = get_device(config['training']['device'])
    model = ClimOnly(k).to(device).eval()
    print(f'Climatologia sola: canal {cfg_data["clim_channels"][0]} (posicion {k} de la entrada)')
    evaluate_model(model, loader, device, config)


if __name__ == '__main__':
    main()
