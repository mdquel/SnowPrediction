r"""
ResUNet++ de doble cabeza con senales de escala EXTERNAS.
=========================================================

Por que existe
--------------
E3 probo una doble cabeza donde la escala se estimaba desde el bottleneck,
es decir, desde la topografia. Fallo: la fraccion de error de escala
AUMENTO en 3 de 4 folds (60.6% -> 64.5%). El motivo es claro en
retrospectiva: se le pidio a la red estimar cuanta nieve hay este anyo
mirando un terreno que es identico todos los anyos. La informacion no
estaba en sus entradas.

Esta version le da la senal que si la contiene. La cabeza de escala
recibe cuatro escalares por fecha de vuelo:

    snow_pct   cobertura de nieve satelital  (r parcial +0.678 con la
               magnitud real, descontada la estacionalidad)
    pdd_15d    grados-dia positivos, 15 dias (r anomalia intra-anual -0.672)
    pdd_30d    grados-dia positivos, 30 dias (r anomalia intra-anual -0.690)
    ppAcc_mm   precipitacion acumulada       (r parcial +0.334)

Las dos familias son complementarias: la cobertura distingue anyos
nevados de secos, los grados-dia siguen la fusion dentro de una
temporada.

Diferencia con el experimento meteo del manuscrito
--------------------------------------------------
Aquel replicaba cada escalar en una matriz de 256x256 y lo anadia como
canal de imagen: la red gastaba todo el encoder convolucional en volver
a reducir a un numero algo que ya era un numero. Aqui los escalares
entran DIRECTAMENTE en la cabeza de escala, concatenados al vector del
global average pooling del bottleneck.

Ademas aquel experimento usaba ventanas de temperatura media, que segun
el analisis de senales no llevan informacion de magnitud (r parcial entre
-0.135 y -0.335). De ahi su resultado negativo.

Arquitectura
------------
    patron  <- decoder completo (contexto espacial, 22 canales topograficos)
    escala  <- [GAP(bottleneck) ; senales externas] -> MLP -> (mu, log sigma)
    HS = patron * sigma + mu
"""

import torch
import torch.nn as nn

from models.resunetpp_dual import ResUNetPPDual


class ResUNetPPDualScale(ResUNetPPDual):
    """Doble cabeza cuya cabeza de escala recibe senales externas.

    Args:
        n_scale_signals: numero de escalares externos por muestra.
                         Con 0 el comportamiento es identico a
                         ResUNetPPDual (util como control).
        (el resto, heredados de ResUNetPPDual)
    """

    def __init__(self,
                 in_channels: int = 22,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0,
                 num_groups: int = 8,
                 scale_reduction: int = 4,
                 log_sigma_clamp: tuple = (-4.0, 3.0),
                 normalize_pattern: bool = False,
                 n_scale_signals: int = 4):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         features=features,
                         dropout_p=dropout_p,
                         num_groups=num_groups,
                         scale_reduction=scale_reduction,
                         log_sigma_clamp=log_sigma_clamp,
                         normalize_pattern=normalize_pattern)

        f4 = (features or [64, 128, 256, 512])[3]
        mid = max(f4 // scale_reduction, 8)
        self.n_scale_signals = n_scale_signals

        # Cabeza de escala: bottleneck agregado + senales externas.
        # El GAP reduce el bottleneck (B, f4, 32, 32) a (B, f4); a ese
        # vector se le concatenan los escalares antes de la MLP.
        self.scale_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.scale_head = nn.Sequential(
            nn.Linear(f4 + n_scale_signals, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 2),
        )
        # Arranque en mu=0, sigma=1: el entrenamiento empieza donde estaria
        # el modelo original y se desvia desde ahi. Hace la comparacion
        # limpia y evita que la cabeza domine la salida desde el principio.
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)

    def _scale_from(self, b, signals):
        """(mu, sigma) a partir del bottleneck y las senales externas."""
        v = self.scale_pool(b)                       # (B, f4)
        if self.n_scale_signals > 0:
            if signals is None:
                signals = v.new_zeros(v.size(0), self.n_scale_signals)
            v = torch.cat([v, signals.to(v.dtype)], dim=1)
        out = self.scale_head(v)                     # (B, 2)
        mu = out[:, 0:1].view(-1, 1, 1, 1)
        sigma = torch.exp(out[:, 1:2].view(-1, 1, 1, 1)
                          .clamp(*self.log_sigma_clamp))
        return mu, sigma

    def forward(self, x: torch.Tensor, signals: torch.Tensor = None):
        b, skips = self._encode(x)
        pattern = self._decode(b, skips)
        mu, sigma = self._scale_from(b, signals)

        if self.normalize_pattern:
            m = pattern.mean(dim=(2, 3), keepdim=True)
            s = pattern.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            pattern = (pattern - m) / s

        hs = pattern * sigma + mu

        if self.training:
            return hs, mu.view(-1), sigma.view(-1)
        return hs

    def predict_components(self, x: torch.Tensor, signals: torch.Tensor = None):
        """Devuelve (hs, patron, mu, sigma) tambien en modo eval."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            b, skips = self._encode(x)
            pattern = self._decode(b, skips)
            mu, sigma = self._scale_from(b, signals)
            if self.normalize_pattern:
                m = pattern.mean(dim=(2, 3), keepdim=True)
                s = pattern.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                pattern = (pattern - m) / s
            hs = pattern * sigma + mu
        if was_training:
            self.train()
        return hs, pattern, mu.view(-1), sigma.view(-1)