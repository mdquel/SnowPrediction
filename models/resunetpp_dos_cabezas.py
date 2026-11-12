r"""
ResUNet++ de dos cabezas sobre la climatologia (protocolo 2026, apartado 8.13)
===============================================================================

Que es
------
La version aprendida de la combinacion (scripts/clim_v1_hybrid.py). La
combinacion desplaza la climatologia, fecha a fecha, hasta el nivel medio que
predice la red B. Esta red parte exactamente de ahi y solo aprende a
corregirla. Para cada tesela de la fecha d:

    HS(x) = max(0, L(d) + w * (C(x) - Cm(d)) + delta)

    C(x)     climatologia del pixel (m); canal `clim_index` de la entrada
             (0 donde no hay climatologia, como en la combinacion)
    Cm(d)    media de la climatologia en la fecha (m)
    L(d)     nivel de la fecha (m)
    w        cuanto se acentua (> 1) o se suaviza (< 1) el dibujo de la
             climatologia dentro de la tesela
    delta    desplazamiento de la tesela respecto al nivel de la fecha (m)

Las dos cabezas dan w y delta: UN valor de cada una por tesela.

Por que asi (leccion de E3, models/resunetpp_dual.py)
-----------------------------------------------------
E3 separo patron y escala, pero la cabeza de escala tenia que estimar cuanta
nieve hay ese anyo mirando un terreno que es igual todos los anyos, y fallo.
Aqui la red no tiene que estimar ni el dibujo ni el nivel:

  * el dibujo lo pone la climatologia, C(x) - Cm(d);
  * el nivel L(d) viene de fuera: en entrenamiento y validacion es la media
    observada de la fecha; en el test, la media que predice la red B;
  * la red solo corrige, y con dos restricciones que impiden la degeneracion
    que senalo Hernan (correo del 27-10-2026: "si la correccion puede tomar
    cualquier valor por pixel, la red puede poner w a 0 e ignorar la
    climatologia"):

      1. w y delta son un valor por tesela: dentro de la tesela el dibujo es
         siempre el de la climatologia, la red no puede dibujar por su cuenta;
      2. estan acotados: w en [0.5, 1.5] y delta en [-0.5, +0.5] m.

Arranque
--------
La ultima capa de cada cabeza se inicializa a cero, de modo que al empezar
w = 1 y delta = 0 y la salida es EXACTAMENTE la combinacion:
max(0, C(x) + L(d) - Cm(d)). Se comprueba con datos reales antes de entrenar
(scripts/dos_cabezas_control_inicio.py).

Arquitectura
------------
    entrada (28 canales, los de la red con climatologia)
      -> codificador y cuello ASPP de ResUNet++ (models/resunet.py, sin cambios)
      -> media espacial del cuello (un vector por tesela)
      -> se le anaden L(d) y Cm(d), en metros (8.15: el contraste depende
         de cuanta nieve hay)
      -> cabeza w     : Linear -> ReLU -> Dropout -> Linear(1) -> acotado
      -> cabeza delta : Linear -> ReLU -> Dropout -> Linear(1) -> acotado

No hay decodificador: la salida de la red son dos escalares por tesela, y el
mapa a 1 m sale de la formula. El dropout del modelo (en ResUNet++ estaba en
el decodificador) se aplica en la capa oculta de las cabezas.

Las acotaciones son suaves (tangente hiperbolica), para que el gradiente no
se anule de golpe en los limites:

    w     = w_centro + w_semiancho * tanh(a + a0)     a0 tal que w = 1 si a = 0
    delta = delta_max * tanh(b)

Entradas de forward
-------------------
    x        (B, C, H, W)  canales de entrada ya normalizados; la climatologia
                           entra en metros (data/dataset.py no la escala)
    signals  (B, 2)        [L(d), Cm(d)] en metros, por este orden
                           (data.date_level_cols de la config)

Devuelve HS (B, 1, H, W) en metros, tanto en entrenamiento como en
evaluacion, asi que usa la misma perdida y la misma evaluacion que el resto
de modelos. predict_components devuelve ademas w y delta de cada tesela.

Variante dc2 (protocolo 2026, 8.16): codificador congelado y cabezas sin L ni Cm
---------------------------------------------------------------------------------
En dc1 (codificador entrenado desde cero, 21 M de parametros, cabezas con
L(d) y Cm(d)) la mejor epoca fue la 1 o la 3 y despues la red memorizo: la
perdida de entrenamiento bajo a la mitad y la de validacion no. Con tres
argumentos opcionales (por defecto, dc1 sin cambios):

    encoder_in_channels  el codificador solo ve los primeros N canales de la
                         entrada (26: los de la red B, en su mismo orden)
    encoder_init         pesos de la red B (state_dict de ResUNetPP); se
                         copian enc1-enc3, bottleneck_in, aspp y
                         bottleneck_out, y tienen que estar todos
    freeze_encoder       el codificador no se entrena (requires_grad False) y
                         se queda en modo evaluacion aunque se llame a train()
    head_signals         False: las cabezas solo ven el cuello; L(d) y Cm(d)
                         se siguen usando en la formula

Asi solo se entrenan las dos cabezas (unos 130 000 parametros) sobre lo que
la red B ya sabe ver de cada tesela. El arranque sigue siendo la combinacion.
"""

import math

import torch
import torch.nn as nn

from models.resunet import ASPPModule, ResidualBlock


class ResUNetPPDosCabezas(nn.Module):
    """Dos cabezas (w y delta por tesela) sobre la climatologia.

    Args:
        in_channels:    canales de entrada
        out_channels:   debe ser 1 (HS en metros)
        features:       filtros por nivel [f1, f2, f3, f4]
        dropout_p:      dropout en la capa oculta de las cabezas
        num_groups:     grupos de GroupNorm
        clim_index:     posicion del canal de climatologia DENTRO de la entrada
                        (despues de channel_indices), en metros
        w_range:        limites de w (el 1 tiene que quedar dentro)
        delta_max:      limite de |delta| en metros
        head_reduction: la capa oculta de cada cabeza tiene f4 / head_reduction
    """

    SIGNALS = ('nivel_m', 'clim_media_m')   # orden de las columnas de signals

    def __init__(self,
                 in_channels: int = 28,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0,
                 num_groups: int = 8,
                 clim_index: int = 26,
                 w_range: tuple = (0.5, 1.5),
                 delta_max: float = 0.5,
                 head_reduction: int = 4,
                 encoder_in_channels: int = None,
                 encoder_init: str = None,
                 freeze_encoder: bool = False,
                 head_signals: bool = True):
        super().__init__()
        if out_channels != 1:
            raise ValueError('ResUNetPPDosCabezas solo admite out_channels = 1')
        if not 0 <= clim_index < in_channels:
            raise ValueError(f'clim_index = {clim_index} fuera de los {in_channels} canales de entrada')
        w_min, w_max = float(w_range[0]), float(w_range[1])
        if not w_min < 1.0 < w_max:
            raise ValueError(f'w_range = {w_range}: el 1 tiene que quedar dentro (arranque en la combinacion)')
        if delta_max <= 0:
            raise ValueError('delta_max tiene que ser positivo')

        if features is None:
            features = [64, 128, 256, 512]
        f1, f2, f3, f4 = features
        enc_in = in_channels if encoder_in_channels is None else int(encoder_in_channels)
        if not 0 < enc_in <= in_channels:
            raise ValueError(f'encoder_in_channels = {enc_in} fuera de 1..{in_channels}')
        if freeze_encoder and not encoder_init:
            raise ValueError('freeze_encoder sin encoder_init: se congelaria un codificador aleatorio')
        self.encoder_in_channels = enc_in
        self.head_signals = bool(head_signals)

        # --- Codificador y cuello ASPP: los mismos bloques que ResUNetPP ---
        self.enc1 = ResidualBlock(enc_in, f1, num_groups)
        self.enc2 = ResidualBlock(f1,          f2, num_groups)
        self.enc3 = ResidualBlock(f2,          f3, num_groups)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_in  = ResidualBlock(f3, f4, num_groups)
        self.aspp           = ASPPModule(f4, f4, num_groups)
        self.bottleneck_out = ResidualBlock(f4, f4, num_groups)

        # --- Las dos cabezas: un escalar por tesela cada una ---
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        mid = max(f4 // head_reduction, 8)

        def cabeza():
            return nn.Sequential(
                nn.Linear(f4 + (len(self.SIGNALS) if self.head_signals else 0), mid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Linear(mid, 1),
            )

        self.head_w = cabeza()
        self.head_delta = cabeza()
        # Arranque en la combinacion: salida cruda 0 -> w = 1, delta = 0.
        for head in (self.head_w, self.head_delta):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.clim_index = int(clim_index)
        self.w_range = (w_min, w_max)
        self.delta_max = float(delta_max)
        self._w_centro = 0.5 * (w_min + w_max)
        self._w_semiancho = 0.5 * (w_max - w_min)
        # desplazamiento de la salida cruda para que 0 -> w = 1 con cualquier w_range
        self._a0 = math.atanh((1.0 - self._w_centro) / self._w_semiancho)

        self.encoder_init = encoder_init
        self.freeze_encoder = bool(freeze_encoder)
        if encoder_init:
            self._load_encoder(encoder_init)
        if self.freeze_encoder:
            for mod in self._encoder_modules():
                for p in mod.parameters():
                    p.requires_grad_(False)
                mod.eval()

    # ------------------------------------------------------------------
    ENCODER = ('enc1', 'enc2', 'enc3', 'bottleneck_in', 'aspp', 'bottleneck_out')

    def _encoder_modules(self):
        return [getattr(self, n) for n in self.ENCODER]

    def _load_encoder(self, path: str):
        """Copia el codificador y el cuello ASPP de un state_dict de ResUNetPP."""
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        propio = {k: v for k, v in self.state_dict().items() if k.split('.')[0] in self.ENCODER}
        faltan = [k for k in propio if k not in sd]
        if faltan:
            raise ValueError(f'{path}: faltan {len(faltan)} pesos del codificador (p. ej. {faltan[0]})')
        malos = [k for k in propio if tuple(sd[k].shape) != tuple(propio[k].shape)]
        if malos:
            raise ValueError(f'{path}: forma distinta en {malos[0]}: {tuple(sd[malos[0]].shape)} '
                             f'frente a {tuple(propio[malos[0]].shape)}')
        self.load_state_dict({k: sd[k] for k in propio}, strict=False)
        print(f'  codificador cargado de {path} ({len(propio)} tensores)')

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'freeze_encoder', False):
            for mod in self._encoder_modules():
                mod.eval()
        return self

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x[:, :self.encoder_in_channels])
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck_in(self.pool(e3))
        b = self.aspp(b)
        return self.bottleneck_out(b)

    def _w_delta(self, x: torch.Tensor, signals: torch.Tensor):
        """w y delta de cada tesela, ya acotados. Forma (B, 1, 1, 1).

        Las cabezas reciben el cuello de la red y, si head_signals, los dos
        escalares de la fecha, L(d) y Cm(d) (protocolo 2026, 8.15 y 8.16)."""
        if self.freeze_encoder:
            with torch.no_grad():
                v = self.gap(self._encode(x))                           # (B, f4)
        else:
            v = self.gap(self._encode(x))                               # (B, f4)
        if self.head_signals:
            v = torch.cat([v, signals.to(v.dtype)], dim=1)              # (B, f4 + 2)
        w = self._w_centro + self._w_semiancho * torch.tanh(self.head_w(v) + self._a0)
        delta = self.delta_max * torch.tanh(self.head_delta(v))
        return w.view(-1, 1, 1, 1), delta.view(-1, 1, 1, 1)

    def _check(self, x, signals):
        if signals is None:
            raise ValueError('ResUNetPPDosCabezas necesita signals = [L(d), Cm(d)] por tesela '
                             '(data.date_levels_csv y data.date_level_cols en la config)')
        if signals.dim() != 2 or signals.size(1) != 2 or signals.size(0) != x.size(0):
            raise ValueError(f'signals debe ser (B, 2); recibido {tuple(signals.shape)}')

    def _hs(self, x, signals, w, delta):
        signals = signals.to(x.dtype)
        nivel = signals[:, 0].view(-1, 1, 1, 1)
        clim_media = signals[:, 1].view(-1, 1, 1, 1)
        clim = x[:, self.clim_index:self.clim_index + 1]                # (B, 1, H, W), metros
        return torch.relu(nivel + w * (clim - clim_media) + delta)

    def forward(self, x: torch.Tensor, signals: torch.Tensor = None) -> torch.Tensor:
        self._check(x, signals)
        w, delta = self._w_delta(x, signals)
        return self._hs(x, signals, w, delta)

    # ------------------------------------------------------------------
    def predict_components(self, x: torch.Tensor, signals: torch.Tensor = None):
        """Devuelve (hs, w, delta) en modo evaluacion y sin gradiente.

        w y delta tienen forma (B,). Sirve para guardar las correcciones de
        cada tesela y comprobar que no se quedan en los limites (8.13).
        """
        self._check(x, signals)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            w, delta = self._w_delta(x, signals)
            hs = self._hs(x, signals, w, delta)
        if was_training:
            self.train()
        return hs, w.view(-1), delta.view(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    torch.manual_seed(0)
    m = ResUNetPPDosCabezas(in_channels=28, features=[64, 128, 256, 512], dropout_p=0.077)
    x = torch.randn(2, 28, 256, 256)
    x[:, 26] = torch.rand(2, 256, 256) * 2.0            # climatologia (m)
    s = torch.tensor([[1.10, 0.80], [0.40, 0.90]])      # [L(d), Cm(d)]
    m.train()
    hs = m(x, s)
    comb = torch.relu(x[:, 26:27] + (s[:, 0] - s[:, 1]).view(-1, 1, 1, 1))
    print(f'salida {tuple(hs.shape)}; diferencia maxima con la combinacion al arrancar: '
          f'{(hs - comb).abs().max().item():.2e} (esperado 0)')
    _, w, d = m.predict_components(x, s)
    print(f'w = {w.tolist()} (esperado 1), delta = {d.tolist()} (esperado 0)')
    print(f'parametros: {m.count_parameters():,}')
