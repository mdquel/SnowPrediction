r"""
ResUNet++ de doble cabeza: desacoplamiento patron / escala.
============================================================

Motivacion empirica
-------------------
El analisis de descomposicion del error (Murphy 1988) sobre cuatro folds
leave-one-year-out mostro que, al extrapolar a un anyo no visto:

  * El error de ESCALA (sesgo + dispersion) domina el MSE en los cuatro
    folds (56.8%, 59.5%, 61.7%, 64.6%).
  * La correlacion espacial media por tile se mantiene practicamente
    constante (0.625-0.626) sea cual sea la anomalia del anyo de test.
  * El sesgo sigue la anomalia del anyo de forma monotona (Spearman = -1.00,
    n=4): el modelo infrapredice los anyos nevados y sobrepredice los secos.
  * Un unico factor de reescalado por fecha recupera hasta +0.28 de R2.

Es decir: el modelo aprende DONDE se acumula la nieve, pero no CUANTA hay
ese anyo. Son dos subproblemas de naturaleza distinta:

  * el patron espacial es una funcion (aproximadamente estable) del terreno
  * la magnitud depende del regimen meteorologico de la temporada

Esta arquitectura introduce ese conocimiento como sesgo inductivo: separa
explicitamente ambas predicciones en dos cabezas.

    HS(x, y) = patron(x, y) * sigma + mu

  * cabeza de PATRON : el decoder completo (contexto espacial fino)
  * cabeza de ESCALA : global average pooling sobre el bottleneck ASPP,
                       que es la representacion mas agregada de la red,
                       -> dos escalares por tile (mu, log sigma)

Notas de implementacion
-----------------------
* La cabeza de escala predice log(sigma), no sigma. La desviacion tipica
  debe ser positiva; prediciendo el logaritmo y exponenciando, la
  positividad esta garantizada por construccion en lugar de depender de
  que la red aprenda a no cruzar el cero. El clamp acota sigma al rango
  [~0.018, ~20] m, holgado para profundidad de nieve.

* En modo entrenamiento forward() devuelve la tupla (hs, mu, sigma) para
  que la loss pueda supervisar las dos cabezas por separado. Sin esa
  supervision la red puede ignorar la separacion: la cabeza de patron
  aprenderia valores absolutos y la de escala se quedaria en mu=0,
  sigma=1, degenerando al modelo original con mas parametros.

* En modo eval() devuelve solo el mapa HS, de modo que es intercambiable
  con ResUNetPP en el pipeline de evaluacion sin tocar nada.

* La salida del decoder NO se normaliza explicitamente por tile. Se confia
  en que la supervision de mu y sigma empuje a la red a la separacion
  deseada. Si al entrenar se observa que la cabeza de escala no aprende
  (mu_pred casi constante), la alternativa es normalizar el campo dentro
  del tile antes de reescalar; ver normalize_pattern.
"""

import torch
import torch.nn as nn

from models.resunet import ResUNetPP


class ResUNetPPDual(ResUNetPP):
    """ResUNet++ con cabeza de escala desacoplada.

    Args:
        in_channels:       canales de entrada
        out_channels:      canales de salida (1 = HS en metros)
        features:          filtros por nivel [f1, f2, f3, f4]
        dropout_p:         dropout en el decoder
        num_groups:        grupos de GroupNorm
        scale_reduction:   factor de reduccion de la MLP de escala
        log_sigma_clamp:   limites de log(sigma) antes de exponenciar
        normalize_pattern: si True, estandariza el campo del decoder dentro
                           de cada tile antes de reescalarlo. Fuerza la
                           separacion por construccion, a costa de una
                           operacion no lineal extra en el grafo.
    """

    def __init__(self,
                 in_channels: int = 22,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0,
                 num_groups: int = 8,
                 scale_reduction: int = 4,
                 log_sigma_clamp: tuple = (-4.0, 3.0),
                 normalize_pattern: bool = False):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         features=features,
                         dropout_p=dropout_p,
                         num_groups=num_groups)

        f4 = (features or [64, 128, 256, 512])[3]
        mid = max(f4 // scale_reduction, 8)

        # Cabeza de escala: del bottleneck ASPP a dos escalares por tile.
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f4, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 2),
        )
        self.log_sigma_clamp = log_sigma_clamp
        self.normalize_pattern = normalize_pattern

        # Inicializacion: mu ~ 0, log sigma ~ 0 (sigma ~ 1). Al empezar, la
        # salida equivale al campo del decoder sin reescalar, de modo que el
        # entrenamiento arranca en el mismo punto que el modelo original.
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)

    # ------------------------------------------------------------------
    def _encode(self, x):
        """Encoder + bottleneck ASPP. Devuelve (bottleneck, skips)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck_in(self.pool(e3))
        b = self.aspp(b)
        b = self.bottleneck_out(b)
        return b, (e1, e2, e3)

    def _decode(self, b, skips):
        """Decoder con attention gates. Devuelve el campo de patron."""
        e1, e2, e3 = skips
        a3 = self.att3(b, e3)
        d3 = self.dec3(torch.cat([self.upsample(b), a3], dim=1))
        a2 = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([self.upsample(d3), a2], dim=1))
        a1 = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([self.upsample(d2), a1], dim=1))
        return self.head(d1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        b, skips = self._encode(x)
        pattern = self._decode(b, skips)                      # (B, 1, H, W)

        scale = self.scale_head(b)                            # (B, 2)
        mu = scale[:, 0:1].view(-1, 1, 1, 1)
        log_sigma = scale[:, 1:2].view(-1, 1, 1, 1)
        sigma = torch.exp(log_sigma.clamp(*self.log_sigma_clamp))

        if self.normalize_pattern:
            m = pattern.mean(dim=(2, 3), keepdim=True)
            s = pattern.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            pattern = (pattern - m) / s

        hs = pattern * sigma + mu

        if self.training:
            # La loss necesita mu y sigma para supervisar la cabeza de escala.
            return hs, mu.view(-1), sigma.view(-1)
        return hs

    # ------------------------------------------------------------------
    def predict_components(self, x: torch.Tensor):
        """Devuelve (hs, patron, mu, sigma) tambien en modo eval.

        Util para analisis: permite comparar el patron aprendido con el
        campo observado estandarizado, y contrastar mu/sigma predichos
        frente a los reales por tile.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            b, skips = self._encode(x)
            pattern = self._decode(b, skips)
            scale = self.scale_head(b)
            mu = scale[:, 0:1].view(-1, 1, 1, 1)
            sigma = torch.exp(scale[:, 1:2].view(-1, 1, 1, 1)
                              .clamp(*self.log_sigma_clamp))
            if self.normalize_pattern:
                m = pattern.mean(dim=(2, 3), keepdim=True)
                s = pattern.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                pattern = (pattern - m) / s
            hs = pattern * sigma + mu
        if was_training:
            self.train()
        return hs, pattern, mu.view(-1), sigma.view(-1)


if __name__ == '__main__':
    m = ResUNetPPDual(in_channels=22, features=[64, 128, 256, 512])
    x = torch.randn(2, 22, 256, 256)

    m.train()
    hs, mu, sigma = m(x)
    print(f'train -> hs {tuple(hs.shape)}  mu {tuple(mu.shape)}  sigma {tuple(sigma.shape)}')
    print(f'         mu={mu.tolist()}  sigma={sigma.tolist()}  (esperado mu~0, sigma~1)')

    m.eval()
    print(f'eval  -> hs {tuple(m(x).shape)}')
    print(f'params: {m.count_parameters():,}')
