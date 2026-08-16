import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SnowDataset(Dataset):
    """
    Dataset de tiles de nieve para entrenamiento y validacion.

    Canales del .npy (dataset v5, 6 canales):
        [0] DEM        - Modelo Digital de Elevacion (metros)
        [1] Slope      - Pendiente (grados)
        [2] Northness  - cos(aspect), orientacion norte [-1, 1]
        [3] Eastness   - sin(aspect), orientacion este  [-1, 1]
        [4] TPI        - Topographic Position Index
        [5] SCE        - Snow Cover Extent satelite (opcional)

    Canales adicionales del .npy (dataset v6, 17 canales):
        [6-13] Sx_DIR_100  - Wind Shelter Index, 8 direcciones, radio 100m
        [14]   Pers_15d    - Fraccion de dias nevados en los ultimos 15 dias
        [15]   Pers_30d    - Fraccion de dias nevados en los ultimos 30 dias
        [16]   Pers_60d    - Fraccion de dias nevados en los ultimos 60 dias
    """

    NORM = {
        'dem_mean':  2100.0,
        'dem_std':   1000.0,
        'slope_max':   90.0,
        'tpi_max':   9200.0,   # Rango real observado: ~[-9155, +9141]
        'sx_max':      90.0,   # Angulo maximo teorico del horizonte
    }

    # Normalizacion v3: constantes empiricas derivadas del TRAIN set (split
    # temporal, 4241 tiles, solo pixeles validos). Ver
    # scripts/compute_norm_v3_stats.py y results/norm_v3/norm_v3_stats.json.
    # Motivacion: tpi_max=9200 y dem_std=1000 de NORM estan calibradas sobre
    # artefactos nodata y aplastan la senal real (TPI real ~[-5,+5] m a 1m,
    # ~[-15,+15] m a 5m; std real del DEM ~130 m). Ademas el remuestreo del
    # DEM 5m genera valores intermedios entre -9999 y el terreno real que
    # pasan el filtro ==-9999; los umbrales *_valid los detectan y esos
    # pixeles se neutralizan a 0 tras normalizar (mismo convenio que nodata).
    NORM_V3 = {
        'dem_mean':      2015.0,   # media train TEMPORAL (rango cuenca 1712-2312)
        'dem_std':        130.0,   # std train temporal
        'dem_min_valid': 1700.0,   # por debajo = artefacto de remuestreo
        'tpi1_scale':       5.0,   # |p99.5| del TPI 1m
        'tpi1_max_valid':  50.0,   # |TPI 1m| mayor = artefacto
        'tpi5_scale':      15.0,   # grueso del TPI 5m
        'tpi5_max_valid': 100.0,   # |TPI 5m| mayor = artefacto
        'clip':             3.0,   # saturacion (winsorizing) de extremos reales
    }

    # Constantes para el SPLIT ESPACIAL: derivadas del train ESPACIAL, no del
    # temporal. Las bandas espaciales cubren un rango de elevacion mas estrecho
    # (DEM std 106 vs 134). Usar las constantes temporales aqui seria una fuga:
    # se calcularon sobre tiles que incluyen la banda de test espacial. TPI es
    # robusto entre splits, solo cambian mean/std del DEM. Ver
    # scripts/compute_norm_v3_stats.py --split spatial.
    NORM_V3_SPATIAL = {
        'dem_mean':      1945.0,   # media train ESPACIAL
        'dem_std':        106.0,   # std train espacial
        'dem_min_valid': 1700.0,
        'tpi1_scale':       5.0,
        'tpi1_max_valid':  50.0,
        'tpi5_scale':      15.0,
        'tpi5_max_valid': 100.0,
        'clip':             3.0,
    }

    def __init__(self, dataframe: pd.DataFrame, images_dir: str,
                 masks_dir: str, use_sce: bool = False,
                 augment: bool = False, n_channels: int = None):
        """
        Args:
            dataframe:   DataFrame con metadatos de tiles (tile_id, ...)
            images_dir:  Directorio con los .npy de imagenes
            masks_dir:   Directorio con los .npy de mascaras (snow depth)
            use_sce:     Si True incluye SCE (canal 5). Ignorado si n_channels>5.
            augment:     Si True aplica data augmentation en tiempo real
            n_channels:  Numero de canales a cargar. Si None se deduce de use_sce.
                         Usar 17 para dataset v6 (Sx + persistencia).
        """
        self.df           = dataframe.reset_index(drop=True)
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.use_sce      = use_sce
        self.augment      = augment
        self.augment_mode = 'hv'   # se sobreescribe desde main.py si es necesario

        if n_channels is not None:
            self.n_channels = n_channels
        else:
            self.n_channels = 6 if use_sce else 5

        # Indices de canales a cargar. Si None, carga los primeros n_channels.
        # Para dataset v6 (33 canales con huecos) usar [0..13, 30..32].
        self.channel_indices = None   # se puede sobreescribir desde main.py

        # Normalizacion de los canales "extendidos" (topografia 5m 17-21 y
        # meteorologia 22+). True = comportamiento actual (se normalizan).
        # False = comportamiento antiguo: esos canales entran en CRUDO, tal como
        # se entrenaron los experimentos de 22ch previos al parche de _normalize.
        # Se fija desde main.py via config data.norm_extended (default True).
        self.norm_extended = True     # se puede sobreescribir desde main.py

        # Version de normalizacion. None = comportamiento legacy gobernado por
        # norm_extended ('vieja'/'nueva'). 'v3' = normalizacion empirica con
        # exclusion de artefactos (ver NORM_V3). Se fija desde main.py via
        # config data.norm_version.
        self.norm_version = None      # se puede sobreescribir desde main.py

        # Si True, __getitem__ devuelve (image, mask, valid) donde valid marca
        # los pixeles del target con dato LiDAR real (los NaN de la mascara son
        # nodata; _clean los convierte a 0 y sin este flag la loss los veria
        # como "0 m de nieve" reales). Lo activa main.py via
        # config training.masked_loss. Default False = 2-tupla legacy.
        self.return_valid = False     # se puede sobreescribir desde main.py

        # --- Senales de escala externas (E4) ---
        # Escalares por FECHA de vuelo (cobertura de nieve, grados-dia,
        # precipitacion acumulada) que alimentan la cabeza de escala del
        # modelo dual. No son canales de imagen: entran directamente en la
        # cabeza, sin pasar por las convoluciones.
        #
        # Motivacion: E3 fallo porque la cabeza de escala tenia que estimar
        # la magnitud desde la topografia, que es identica todos los anyos.
        # Aqui se le da una senal que si varia entre anyos y correlaciona
        # con la magnitud real (cobertura de nieve, r parcial +0.678).
        #
        # Se normalizan con constantes FIJAS, iguales en todos los folds:
        # si cada fold usase las suyas, el modelo veria escalas distintas y
        # los resultados no serian comparables entre folds.
        self.scale_signals = None       # dict {fecha_str: np.ndarray}
        self.scale_signal_cols = None   # lista de nombres, para trazabilidad
        self.n_scale_signals = 0

    # Rangos observados sobre las 23 fechas admisibles (build_scale_signals.py).
    # Cada senal se lleva a [0, 1] con (x - min) / (max - min).
    SCALE_NORM = {
        'snow_pct':  (0.0, 100.0),      # porcentaje, rango natural
        'pdd_15d':   (0.0, 110.0),      # observado 0.99 - 103.10
        'pdd_30d':   (0.0, 170.0),      # observado 5.10 - 160.08
        'ppAcc_mm':  (0.0, 2000.0),     # observado 353.86 - 1895.86
    }

    def load_scale_signals(self, csv_path: str, cols: list):
        """Carga las senales de escala desde el CSV consolidado.

        El CSV tiene una fila por fecha de vuelo. Se indexan por fecha en
        formato YYYYMMDD, que es como aparece en el tile_id.

        Las fechas no admisibles (sin las cuatro senales) no deberian
        aparecer en el dataframe si se uso un CSV de fold de E4; si aun
        asi apareciese alguna, se avisa y se le asignan ceros.
        """
        sig = pd.read_csv(csv_path)
        sig = sig[sig['admisible'] == True]
        table = {}
        for _, r in sig.iterrows():
            key = pd.to_datetime(r['fecha']).strftime('%Y%m%d')
            vals = []
            for c in cols:
                lo, hi = self.SCALE_NORM.get(c, (0.0, 1.0))
                vals.append((float(r[c]) - lo) / (hi - lo))
            table[key] = np.array(vals, dtype=np.float32)
        self.scale_signals = table
        self.scale_signal_cols = list(cols)
        self.n_scale_signals = len(cols)

        # comprobar cobertura de las fechas presentes en este split
        dates = {str(d) for d in self.df['date'].astype(str)}
        faltan = sorted(dates - set(table.keys()))
        if faltan:
            print(f"  [AVISO] {len(faltan)} fechas sin senales de escala: "
                  f"{', '.join(faltan[:5])}{' ...' if len(faltan) > 5 else ''}")
            print(f"          se les asignaran ceros; revisa el CSV del fold.")

    def _get_scale(self, tile_id: str) -> np.ndarray:
        """Senales de escala del tile, a partir de su fecha."""
        key = str(tile_id).split('_')[0]
        if self.scale_signals is None:
            return np.zeros(0, dtype=np.float32)
        return self.scale_signals.get(
            key, np.zeros(self.n_scale_signals, dtype=np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tile_id  = self.df.iloc[idx]['tile_id']
        img_path = os.path.join(self.images_dir, tile_id)
        msk_path = os.path.join(self.masks_dir,  tile_id)

        try:
            image = np.load(img_path).astype(np.float32)
            mask  = np.load(msk_path).astype(np.float32)
        except Exception as e:
            print(f"Error cargando {tile_id}: {e}")
            if self.return_valid:
                return (torch.zeros((self.n_channels, 256, 256)),
                        torch.zeros((1, 256, 256)),
                        torch.zeros((1, 256, 256)))
            return (torch.zeros((self.n_channels, 256, 256)),
                    torch.zeros((1, 256, 256)))

        # Validez del target ANTES de _clean (que destruye la marca de nodata
        # convirtiendo NaN -> 0). NaN = fuera de la huella LiDAR.
        valid = (np.isfinite(mask) & (mask > -100)).astype(np.float32)

        image, mask = self._clean(image, mask)
        image = self._normalize(image)
        if self.channel_indices is not None:
            image = image[self.channel_indices, :, :]
        else:
            image = image[:self.n_channels, :, :]

        if self.augment:
            image, mask, valid = self._augment(image, mask, valid)

        # Senales de escala (E4): se anaden al final para no romper el
        # desempaquetado de los flujos que no las usan.
        if self.n_scale_signals > 0:
            scale = torch.from_numpy(self._get_scale(tile_id))
            if self.return_valid:
                return (torch.from_numpy(image.copy()),
                        torch.from_numpy(mask.copy()).unsqueeze(0),
                        torch.from_numpy(valid.copy()).unsqueeze(0),
                        scale)
            return (torch.from_numpy(image.copy()),
                    torch.from_numpy(mask.copy()).unsqueeze(0),
                    scale)

        if self.return_valid:
            return (torch.from_numpy(image.copy()),
                    torch.from_numpy(mask.copy()).unsqueeze(0),
                    torch.from_numpy(valid.copy()).unsqueeze(0))
        return torch.from_numpy(image.copy()), torch.from_numpy(mask.copy()).unsqueeze(0)

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------
    def _clean(self, image: np.ndarray, mask: np.ndarray):
        image[image == -9999] = 0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        mask[mask <= -100] = 0
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        # Profundidad de nieve negativa es fisicamente imposible: el ruido
        # LiDAR pequeno (~[-0.1, 0)) se recorta a 0 (suelo desnudo). La mascara
        # de validez ya se calculo antes de _clean, asi que estos pixeles
        # siguen siendo validos y entran a la loss como 0 m (correcto).
        mask = np.maximum(mask, 0.0)
        return image, mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.norm_version in ('v3', 'v3_spatial'):
            const = (self.NORM_V3_SPATIAL if self.norm_version == 'v3_spatial'
                     else self.NORM_V3)
            return self._normalize_v3(image, const)
        n = image.shape[0]  # usar tamano real, no self.n_channels

        # Canales base (presentes en todos los datasets)
        image[0] = (image[0] - self.NORM['dem_mean']) / self.NORM['dem_std']
        image[1] = image[1] / self.NORM['slope_max']
        # Canal 2 (northness) y 3 (eastness) ya estan en [-1, 1]
        # TPI: rango real ~[-9000, +9000] -> normalizamos con su rango real
        image[4] = np.clip(image[4] / self.NORM['tpi_max'], -1.0, 1.0)

        # Canal 5 (SCE): codigos 0/10/11 -> binario [0, 1]
        if n >= 6:
            image[5] = (image[5] > 5).astype(np.float32)

        # Canales 6-13 (Sx): angulos de horizonte en grados -> [-1, 1]
        if n >= 14:
            image[6:14] = np.clip(image[6:14] / self.NORM['sx_max'], -1.0, 1.0)

        # Canales 14-16 (persistencia nival): ya en [0, 1], no requieren normalizacion

        # Canales 17-21 (topografia 5m): misma normalizacion que 1m
        # (solo si norm_extended; los experimentos 22ch antiguos los usaban crudos)
        if n >= 22 and self.norm_extended:
            image[17] = (image[17] - self.NORM['dem_mean']) / self.NORM['dem_std']
            image[18] = image[18] / self.NORM['slope_max']
            # 19 (northness_5m) y 20 (eastness_5m) ya en [-1, 1]
            image[21] = np.clip(image[21] / self.NORM['tpi_max'], -1.0, 1.0)

        # Canales 22-25 (meteorologia correo2): escalares por fecha
        # t2m_7d, t2m_15d, t2m_30d en degC -> dividir por 10 (rango aprox -5 a +10)
        # ppAcc_from_oct en mm -> dividir por 1000 (rango aprox 0 a 2000)
        if n >= 26 and self.norm_extended:
            image[22] = image[22] / 10.0
            image[23] = image[23] / 10.0
            image[24] = image[24] / 10.0
            image[25] = image[25] / 1000.0

        # Canales 26-34 (meteorologia correo3): escalares por fecha
        # ws_7d, ws_15d, ws_30d en m/s -> dividir por 10 (rango 0-9)
        # rh_7d, rh_15d, rh_30d en % -> dividir por 100 (rango 0-100)
        # rad_7d, rad_15d, rad_30d en W/m2 -> dividir por 600 (rango 0-600)
        if n >= 35 and self.norm_extended:
            image[26] = image[26] / 10.0
            image[27] = image[27] / 10.0
            image[28] = image[28] / 10.0
            image[29] = image[29] / 100.0
            image[30] = image[30] / 100.0
            image[31] = image[31] / 100.0
            image[32] = image[32] / 600.0
            image[33] = image[33] / 600.0
            image[34] = image[34] / 600.0

        return image

    def _normalize_v3(self, image: np.ndarray, const: dict = None) -> np.ndarray:
        """Normalizacion v3: constantes empiricas del train set (NORM_V3 para
        el split temporal, NORM_V3_SPATIAL para el espacial) y exclusion de
        artefactos de remuestreo.

        Para DEM y TPI (1m y 5m): los pixeles fuera del rango fisico plausible
        (sangrado de nodata en el remuestreo) se detectan ANTES de escalar y se
        neutralizan a 0 DESPUES de escalar, igual que el resto de nodata. Los
        valores validos se escalan con constantes empiricas y los extremos
        reales raros se saturan a +-clip (winsorizing).
        NOTA: _clean ya convirtio los -9999 exactos en 0; esos pixeles quedan
        dentro de los umbrales de plausibilidad del TPI (0 es plausible) y su
        valor escalado sigue siendo 0, que es el convenio de "sin informacion".
        """
        V = const if const is not None else self.NORM_V3
        n = image.shape[0]
        clip = V['clip']

        def _dem(ch):
            invalid = ch < V['dem_min_valid']
            out = np.clip((ch - V['dem_mean']) / V['dem_std'], -clip, clip)
            out[invalid] = 0.0
            return out

        def _tpi(ch, scale, max_valid):
            invalid = np.abs(ch) > max_valid
            out = np.clip(ch / scale, -clip, clip)
            out[invalid] = 0.0
            return out

        # Topografia 1m
        image[0] = _dem(image[0])
        image[1] = image[1] / self.NORM['slope_max']
        # 2 (northness) y 3 (eastness) ya en [-1, 1]
        image[4] = _tpi(image[4], V['tpi1_scale'], V['tpi1_max_valid'])

        # Canal 5 (SCE): codigos 0/10/11 -> binario [0, 1]
        if n >= 6:
            image[5] = (image[5] > 5).astype(np.float32)

        # Canales 6-13 (Sx): angulos de horizonte en grados -> [-1, 1].
        # ~0.34% de pixeles traen -FLT_MAX (artefacto de calculo); el clip de
        # las normas legacy los disfrazaba de -1 ("exposicion maxima"). Aqui
        # se detectan (|Sx|>90 es fisicamente imposible) y se neutralizan a 0.
        if n >= 14:
            sx = image[6:14]
            invalid_sx = np.abs(sx) > self.NORM['sx_max']
            sx = np.clip(sx / self.NORM['sx_max'], -1.0, 1.0)
            sx[invalid_sx] = 0.0
            image[6:14] = sx

        # Canales 14-16 (persistencia nival): ya en [0, 1]

        # Topografia 5m
        if n >= 22:
            image[17] = _dem(image[17])
            image[18] = image[18] / self.NORM['slope_max']
            # 19 (northness_5m) y 20 (eastness_5m) ya en [-1, 1]
            image[21] = _tpi(image[21], V['tpi5_scale'], V['tpi5_max_valid'])

        # Canales meteo (22+): mismos escalados que la norm extendida
        if n >= 26:
            image[22] = image[22] / 10.0
            image[23] = image[23] / 10.0
            image[24] = image[24] / 10.0
            image[25] = image[25] / 1000.0
        if n >= 35:
            image[26] = image[26] / 10.0
            image[27] = image[27] / 10.0
            image[28] = image[28] / 10.0
            image[29] = image[29] / 100.0
            image[30] = image[30] / 100.0
            image[31] = image[31] / 100.0
            image[32] = image[32] / 600.0
            image[33] = image[33] / 600.0
            image[34] = image[34] / 600.0

        return image

    def _augment(self, image: np.ndarray,
                 mask: np.ndarray,
                 valid: np.ndarray = None) -> tuple:
        """
        Aplica flips aleatorios en tiempo real.

        Modos controlados por self.augment_mode:
          'hv'  - Flip H y V independientes (prob 0.5 c/u) -> 4 variantes
                  ATENCION: el flip V invierte la asimetria N-S del snowpack
                  (laderas N tienen mas nieve que S en los Pirineos).
                  Puede introducir patrones fisicamente incorrectos.
          'h'   - Solo flip horizontal (prob 0.5) -> 2 variantes
                  Conserva la asimetria N-S. Recomendado para cuencas con
                  gradiente N-S dominante en acumulacion de nieve.

        Correcciones de canal:
          Flip H: negar Eastness (ch 3)  — el eje E-O se invierte
          Flip V: negar Northness (ch 2) — el eje N-S se invierte
        """
        flip_h = random.random() > 0.5
        flip_v = (self.augment_mode == 'hv') and (random.random() > 0.5)

        if flip_h:
            image = np.flip(image, axis=2)   # invertir W
            mask  = np.flip(mask,  axis=1)
            if valid is not None:
                valid = np.flip(valid, axis=1)
            image[3] = -image[3]             # Eastness -> negar

        if flip_v:
            image = np.flip(image, axis=1)   # invertir H
            mask  = np.flip(mask,  axis=0)
            if valid is not None:
                valid = np.flip(valid, axis=0)
            image[2] = -image[2]             # Northness -> negar

        return image, mask, valid


class SnowDatasetEval(SnowDataset):
    """
    Igual que SnowDataset pero devuelve tambien el tile_id,
    necesario para el bucle de evaluacion.

    Si return_valid=True devuelve (image, mask, valid, tile_id): valid marca
    los pixeles con dato LiDAR real (no nodata), lo que permite calcular
    metricas full-domain (incluyendo suelo desnudo) y de deteccion, ademas de
    las snow-only. Si False (legacy) devuelve (image, mask, tile_id).
    """

    def __getitem__(self, idx):
        tile_id  = self.df.iloc[idx]['tile_id']
        img_path = os.path.join(self.images_dir, tile_id)
        msk_path = os.path.join(self.masks_dir,  tile_id)

        try:
            image = np.load(img_path).astype(np.float32)
            mask  = np.load(msk_path).astype(np.float32)
        except Exception as e:
            print(f"Error cargando {tile_id}: {e}")
            if self.return_valid:
                return (torch.zeros((self.n_channels, 256, 256)),
                        torch.zeros((1, 256, 256)),
                        torch.zeros((1, 256, 256)), tile_id)
            return (torch.zeros((self.n_channels, 256, 256)),
                    torch.zeros((1, 256, 256)), tile_id)

        # Validez del target ANTES de _clean (nodata = NaN)
        valid = (np.isfinite(mask) & (mask > -100)).astype(np.float32)

        image, mask = self._clean(image, mask)
        image = self._normalize(image)
        if self.channel_indices is not None:
            image = image[self.channel_indices, :, :]
        else:
            image = image[:self.n_channels, :, :]

        # Senales de escala (E4): se anaden al final para no romper el
        # desempaquetado de los flujos que no las usan.
        if self.n_scale_signals > 0:
            scale = torch.from_numpy(self._get_scale(tile_id))
            if self.return_valid:
                return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                        torch.from_numpy(valid).unsqueeze(0), tile_id, scale)
            return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                    tile_id, scale)

        if self.return_valid:
            return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                    torch.from_numpy(valid).unsqueeze(0), tile_id)
        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0), tile_id

        if self.n_scale_signals > 0:
            scale = torch.from_numpy(self._get_scale(tile_id))
            if self.return_valid:
                return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                        torch.from_numpy(valid).unsqueeze(0), tile_id, scale)
            return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                    tile_id, scale)

        if self.return_valid:
            return (torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0),
                    torch.from_numpy(valid).unsqueeze(0), tile_id)
        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0), tile_id


# ----------------------------------------------------------------------
# Funcion de utilidad para cargar los splits del CSV
# ----------------------------------------------------------------------
def load_splits(csv_path: str,
                source: str = 'lidar',
                split_type: str = 'temporal'):
    """
    Carga el CSV de metadatos y devuelve los tres splits.

    Args:
        csv_path:   Ruta al CSV generado por los notebooks a0x
        source:     'lidar' | 'pleiades' | 'grass' | 'all'
        split_type: 'temporal' (por anyo) | 'spatial' (por posicion)

    Returns:
        Tupla (train_df, val_df, test_df)
    """
    df = pd.read_csv(csv_path)

    if source != 'all':
        df = df[df['source'] == source].reset_index(drop=True)

    if split_type == 'temporal':
        col = 'exp_temporal_split'
    elif split_type == 'spatial':
        col = 'exp_spatial_split'
    else:
        raise ValueError(f"split_type desconocido: '{split_type}'. "
                         f"Usa 'temporal' o 'spatial'.")

    train_df = df[df[col] == 'train'].reset_index(drop=True)
    val_df   = df[df[col] == 'val'].reset_index(drop=True)
    test_df  = df[df[col] == 'test'].reset_index(drop=True)

    print(f"\nSplit '{split_type}' | Fuente: '{source}'")
    print(f"  Train : {len(train_df):>5} tiles  "
          f"({train_df['year'].unique().tolist() if len(train_df) else []})")
    print(f"  Val   : {len(val_df):>5} tiles  "
          f"({val_df['year'].unique().tolist() if len(val_df) else []})")
    print(f"  Test  : {len(test_df):>5} tiles  "
          f"({test_df['year'].unique().tolist() if len(test_df) else []})")

    return train_df, val_df, test_df
