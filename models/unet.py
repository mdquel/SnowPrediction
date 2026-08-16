import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resunet import ResUNetPP
from models.unet_gn import UNetGN
from models.resunetpp_dual import ResUNetPPDual
from models.resunetpp_dual_scale import ResUNetPPDualScale


class DoubleConv(nn.Module):
    """Bloque de dos Conv2d 3x3 con BatchNorm y ReLU."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate para U-Net (Oktay et al., 2018).
    Pondera las skip connections segun relevancia espacial.
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: señal del decoder (gating signal)
            x: skip connection del encoder
        """
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear',
                             align_corners=True)
        psi = F.relu(self.W_g(g_up) + self.W_x(x), inplace=True)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    U-Net para prediccion de profundidad de nieve.

    Arquitectura encoder-decoder con skip connections simetricas.

    Entrada : (B, in_channels, 256, 256)
    Salida  : (B, 1, 256, 256)  -> mapa de profundidad en metros
    """

    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0):
        """
        Args:
            in_channels:  Numero de canales de entrada (5 sin SCE, 6 con SCE)
            out_channels: Numero de canales de salida (1 = snow depth)
            features:     Lista con el numero de filtros por nivel del encoder
            dropout_p:    Probabilidad de dropout en el decoder (0.0 = sin dropout)
        """
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder  (nombres identicos al .pth del alumno)
        self.dconv_down1 = DoubleConv(in_channels,  features[0])
        self.dconv_down2 = DoubleConv(features[0],  features[1])
        self.dconv_down3 = DoubleConv(features[1],  features[2])
        self.dconv_down4 = DoubleConv(features[2],  features[3])  # bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder  (nombres identicos al .pth del alumno)
        self.dconv_up3 = DoubleConv(features[3] + features[2], features[2],
                                    dropout_p=dropout_p)
        self.dconv_up2 = DoubleConv(features[2] + features[1], features[1],
                                    dropout_p=dropout_p)
        self.dconv_up1 = DoubleConv(features[1] + features[0], features[0],
                                    dropout_p=dropout_p)

        self.conv_last = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.dconv_down1(x)
        e2 = self.dconv_down2(self.pool(e1))
        e3 = self.dconv_down3(self.pool(e2))
        b  = self.dconv_down4(self.pool(e3))

        # Decoder con skip connections
        d3 = self.dconv_up3(torch.cat([self.upsample(b),  e3], dim=1))
        d2 = self.dconv_up2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dconv_up1(torch.cat([self.upsample(d2), e1], dim=1))

        return self.conv_last(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al. 2018) para prediccion de profundidad de nieve.

    Las Attention Gates ponderan las skip connections, permitiendo al modelo
    centrarse en las regiones mas relevantes (ej: zonas de acumulacion topografica).
    Especialmente util con pocos datos de entrenamiento.
    """

    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1,
                 features: list = None,
                 dropout_p: float = 0.0):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.enc1 = DoubleConv(in_channels,  features[0])
        self.enc2 = DoubleConv(features[0],  features[1])
        self.enc3 = DoubleConv(features[1],  features[2])
        self.enc4 = DoubleConv(features[2],  features[3])  # bottleneck

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Attention gates (g=decoder signal, x=skip connection)
        self.att3 = AttentionGate(features[3], features[2], features[2] // 2)
        self.att2 = AttentionGate(features[2], features[1], features[1] // 2)
        self.att1 = AttentionGate(features[1], features[0], features[0] // 2)

        # Decoder
        self.dec3 = DoubleConv(features[3] + features[2], features[2],
                               dropout_p=dropout_p)
        self.dec2 = DoubleConv(features[2] + features[1], features[1],
                               dropout_p=dropout_p)
        self.dec1 = DoubleConv(features[1] + features[0], features[0],
                               dropout_p=dropout_p)

        self.conv_last = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.enc4(self.pool(e3))

        # Decoder con attention gates en skip connections
        a3 = self.att3(b,  e3)
        d3 = self.dec3(torch.cat([self.upsample(b),  a3], dim=1))

        a2 = self.att2(d3, e2)
        d2 = self.dec2(torch.cat([self.upsample(d3), a2], dim=1))

        a1 = self.att1(d2, e1)
        d1 = self.dec1(torch.cat([self.upsample(d2), a1], dim=1))

        return self.conv_last(d1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> nn.Module:
    """
    Construye el modelo segun la configuracion YAML.

    Arquitecturas disponibles:
      'unet'         - U-Net estandar (features=[64,128,256,512])
      'unet_small'   - U-Net ligero (features=[32,64,128,256]) para datasets pequenos
      'attention_unet' - Attention U-Net (Oktay 2018) con attention gates en skips

    Parametros opcionales en config.model:
      features: [32, 64, 128, 256]  # sobreescribe los filtros por defecto
      dropout_p: 0.2                # dropout en el decoder (0.0 = desactivado)

    Args:
        config: Diccionario completo cargado desde el YAML

    Returns:
        Modelo PyTorch listo para usar
    """
    cfg    = config['model']
    arch   = cfg['architecture']
    in_ch  = cfg['in_channels']
    out_ch = cfg['out_channels']

    # Parametros opcionales
    dropout_p = cfg.get('dropout_p', 0.0)

    if arch == 'unet':
        features = cfg.get('features', [64, 128, 256, 512])
        model = UNet(in_channels=in_ch, out_channels=out_ch,
                     features=features, dropout_p=dropout_p)

    elif arch == 'unet_gn':
        features = cfg.get('features', [64, 128, 256, 512])
        num_groups = cfg.get('num_groups', 8)
        model = UNetGN(in_channels=in_ch, out_channels=out_ch,
                       features=features, dropout_p=dropout_p,
                       num_groups=num_groups)

    elif arch == 'unet_small':
        features = cfg.get('features', [32, 64, 128, 256])
        model = UNet(in_channels=in_ch, out_channels=out_ch,
                     features=features, dropout_p=dropout_p)

    elif arch == 'attention_unet':
        features = cfg.get('features', [64, 128, 256, 512])
        model = AttentionUNet(in_channels=in_ch, out_channels=out_ch,
                              features=features, dropout_p=dropout_p)

    elif arch == 'resunetpp':
        features  = cfg.get('features', [64, 128, 256, 512])
        num_groups = cfg.get('num_groups', 8)
        model = ResUNetPP(in_channels=in_ch, out_channels=out_ch,
                          features=features, dropout_p=dropout_p,
                          num_groups=num_groups)
    elif arch == 'resunetpp_dual':
        features  = cfg.get('features', [64, 128, 256, 512])
        num_groups = cfg.get('num_groups', 8)
        model = ResUNetPPDual(in_channels=in_ch, out_channels=out_ch,
                              features=features, dropout_p=dropout_p,
                              num_groups=num_groups,
                              normalize_pattern=cfg.get('normalize_pattern', False))
    elif arch == 'resunetpp_dual_scale':
        features  = cfg.get('features', [64, 128, 256, 512])
        num_groups = cfg.get('num_groups', 8)
        n_signals = len(config.get('data', {}).get('scale_signal_cols', [])) or 4
        model = ResUNetPPDualScale(in_channels=in_ch, out_channels=out_ch,
                                   features=features, dropout_p=dropout_p,
                                   num_groups=num_groups,
                                   normalize_pattern=cfg.get('normalize_pattern', False),
                                   n_scale_signals=n_signals)
                                   
    else:
        raise ValueError(f"Arquitectura desconocida: '{arch}'. "
                         f"Disponibles: 'unet', 'unet_gn', 'unet_small', "
                         f"'attention_unet', 'resunetpp'")

    n_params = model.count_parameters()
    print(f"Modelo     : {arch}")
    print(f"Canales    : {in_ch} entrada -> {out_ch} salida")
    print(f"Parametros : {n_params:,}")
    if dropout_p > 0:
        print(f"Dropout    : {dropout_p} (en decoder)")
    return model
