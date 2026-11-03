r"""
Regenera las teselas del 7 y del 14 de marzo de 2024 con los mapas corregidos
=============================================================================

Enmienda 2 del protocolo (analysis/protocolo_test_2026.md, 8.7): Jesús
Revuelto corrigió los mapas del 7 y del 14 de marzo de 2024 (eran idénticos).
Están en validación, así que intervienen en la elección de la mejor época.
Este script regenera SOLO las teselas de esas dos fechas, con el mismo código
que data/generate_dataset_v4_ms_sx200.py, sin tocar el resto del dataset.

Pasos
-----
A. Conversión. Los mapas corregidos vienen en otro formato (rejilla del MDE,
   EPSG:25830, sin dato = -9999). Se escriben copias con el formato de los
   originales (sin dato = NaN, sin CRS, float64, misma transformación) en
   <nuevos>/convertidos/SD_<fecha>_1m.tif. Sin esto, el generador descartaría
   toda tesela con un -9999 (filtro min < -100) y la interpolación mezclaría
   -9999 con valores reales.
B. Control (obligatorio, antes de escribir nada). Con la misma función se
   regenera una fecha que NO cambia (por defecto 2024-02-13, validación) a
   partir de su mapa original, y las teselas tienen que salir IDÉNTICAS a las
   que ya están en el dataset (mismos nombres, mismos canales, misma máscara).
   Si no, el script se detiene.
C. Regeneración. Para cada fecha corregida:
   - las teselas antiguas (images, masks, sat_v2 y clim_v1 si existen) se
     MUEVEN a <dataset>/_reemplazadas_mapas2024/, no se borran;
   - se escriben las teselas nuevas (mismo nombre de fichero que daría el
     generador: <fecha>_lidar_tile_<ty>_<tx>.npy);
   - dataset_v4_ms_sx200.csv: se sustituyen las filas de esas fechas; la
     versión anterior se guarda como dataset_v4_ms_sx200_v1.csv;
   - data/tile_georef.csv: se sustituyen sus filas con la posición real en la
     rejilla del MDE (mismo cálculo que scripts/tile_georef.py, origen
     truncado); la anterior se guarda como data/tile_georef_v1.csv; las filas
     nuevas se validan contra el MDE (diferencia mediana < 0,01 m).
D. Resumen en analysis/regen_mapas_2024.json (huellas de los mapas, teselas
   antes y después, controles).

Después hay que generar los canales de satélite y de climatología de las
teselas nuevas (data/patch_sat_channels.py y data/patch_clim_channels.py, que
solo escriben las que faltan) y rehacer el manifiesto.

Uso (desde la raíz de repo-moises; necesita opencv, como el generador)
----------------------------------------------------------------------
    python data/regen_mapas_2024.py --nuevos "Articulo 1/Data/izas/LiDAR/SnowDepth_2024_corregidos"
    python data/regen_mapas_2024.py --nuevos ... --solo-control     # solo A y B
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dataset_v4_ms_sx200 as G  # noqa: E402

FECHAS = ('20240307', '20240314')
CONTROL = '20240213'
TOL_GEOREF = 0.01


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# A. Conversión al formato de los mapas originales
# --------------------------------------------------------------------------
def convertir(src_path, dst_path):
    with rasterio.open(src_path) as s:
        a = s.read(1).astype(np.float64)
        nod = s.nodata
        prof = dict(driver='GTiff', height=s.height, width=s.width, count=1,
                    dtype='float64', transform=s.transform, nodata=np.nan)
    a[a == nod] = np.nan
    a[a < -100] = np.nan
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.open(dst_path, 'w', **prof) as d:      # sin CRS, como los originales
        d.write(a, 1)
    return {'pixeles_validos': int(np.isfinite(a).sum()), 'nodata_original': nod}


# --------------------------------------------------------------------------
# Teselas de una fecha: COPIA del cuerpo del bucle de G.main() (bloques A-H)
# --------------------------------------------------------------------------
def teselas_fecha(fp, date_str, ctx, src_dem):
    sx_arrays, ctx_arrays, transform_5m, sce_index = ctx
    dem_crs = src_dem.crs
    year = int(date_str[:4])
    sce_date_str = G.DATE_FIXES_LIDAR.get(date_str, date_str)
    target_date = datetime.strptime(sce_date_str, '%Y%m%d')
    stride = G.STRIDE_TRAIN if year in G.TRAIN_YEARS else G.STRIDE_EVALTEST

    # A. DEM crop (1m)
    with rasterio.open(fp) as src_target:
        win_raw = src_dem.window(*src_target.bounds)
        win = win_raw.round_offsets().round_lengths()
        transform_crop = src_dem.window_transform(win)
        dem_crop = src_dem.read(1, window=win, boundless=True, fill_value=G.NODATA_VAL)
        h, w = dem_crop.shape
        sd_crop = np.zeros((h, w), dtype=np.float32)
        reproject(source=rasterio.band(src_target, 1), destination=sd_crop,
                  src_transform=src_target.transform, src_crs=src_target.crs,
                  dst_transform=transform_crop, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

    # B. SCE
    sce_path = G.SCE_DIR / f"izas_{sce_date_str}.tif"
    sce_crop = np.zeros((h, w), dtype=np.float32)
    if sce_path.exists():
        with rasterio.open(sce_path) as src_sce:
            sce_raw = src_sce.read(1).astype(np.float32)
            reproject(source=sce_raw, destination=sce_crop,
                      src_transform=src_sce.transform, src_crs=src_sce.crs,
                      dst_transform=transform_crop, dst_crs=dem_crs,
                      resampling=Resampling.bilinear)

    # C. Topografía fina
    slope, northness, eastness = G.calculate_topography(dem_crop, pixel_size=1.0)
    tpi = G.calculate_tpi(dem_crop)

    # D. Sx
    row_off, col_off = int(win.row_off), int(win.col_off)
    win_h, win_w = int(win.height), int(win.width)
    sx_crops = []
    for sx_arr in sx_arrays:
        r0, r1 = max(row_off, 0), min(row_off + win_h, sx_arr.shape[0])
        c0, c1 = max(col_off, 0), min(col_off + win_w, sx_arr.shape[1])
        patch = np.full((win_h, win_w), 0.0, dtype=np.float32)
        patch[r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off] = sx_arr[r0:r1, c0:c1]
        sx_crops.append(patch)

    # E. Persistencia
    persistence = G.compute_persistence(target_date=target_date, sce_index=sce_index,
                                        dst_transform=transform_crop, dst_crs=dem_crs,
                                        dst_shape=(h, w))

    # F. Contexto 5 m
    bounds_crop = array_bounds(h, w, transform_crop)
    ctx_channels = G.extract_context_patch(ctx_arrays, transform_5m, bounds_crop, h, w)

    # G. Pila de 22 canales
    stack = np.concatenate([
        np.stack([dem_crop, slope, northness, eastness, tpi, sce_crop] + sx_crops
                 + [persistence[0], persistence[1], persistence[2]], axis=0).astype(np.float32),
        ctx_channels], axis=0)

    # H. Teselas (mismos filtros)
    out = []
    for ty in range(0, h - G.TILE_SIZE + 1, stride):
        for tx in range(0, w - G.TILE_SIZE + 1, stride):
            tile_img = stack[:, ty:ty + G.TILE_SIZE, tx:tx + G.TILE_SIZE]
            tile_mask = sd_crop[ty:ty + G.TILE_SIZE, tx:tx + G.TILE_SIZE]
            if np.mean(tile_img[0] == G.NODATA_VAL) > 0.10:
                continue
            if np.min(tile_mask) < -100:
                continue
            valid_frac = np.mean(tile_mask > -100)
            if valid_frac < G.MIN_VALID_FRAC:
                continue
            out.append((f"{date_str}_lidar_tile_{ty}_{tx}.npy", tile_img.copy(),
                        tile_mask.copy(), round(float(valid_frac), 4), ty, tx))
    origen = (win_raw.row_off, win_raw.col_off)
    return out, origen


def dif_max(a, b):
    """Diferencia absoluta máxima (NaN en el mismo sitio cuenta como igual)."""
    if a.shape != b.shape:
        return float('inf')
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float('inf')
    return float(np.max(np.abs(a[~na].astype(np.float64) - b[~nb].astype(np.float64)))) if (~na).any() else 0.0


def main():
    ap = argparse.ArgumentParser(description='Regenera las teselas de marzo de 2024 con los mapas corregidos')
    ap.add_argument('--nuevos', required=True, help='carpeta con SD_20240307.tif y SD_20240314.tif corregidos')
    ap.add_argument('--dataset', default=str(G.OUTPUT_DIR))
    ap.add_argument('--georef', default='data/tile_georef.csv')
    ap.add_argument('--control', default=CONTROL)
    ap.add_argument('--resumen', default='analysis/regen_mapas_2024.json')
    ap.add_argument('--solo-control', action='store_true')
    ap.add_argument('--tol', type=float, default=0.0,
                    help='diferencia máxima admitida en el control (0 = idéntico)')
    a = ap.parse_args()

    ds = Path(a.dataset)
    csv_path = ds / 'dataset_v4_ms_sx200.csv'
    res = {'descripcion': 'Regeneración de las teselas del 7 y del 14 de marzo de 2024 (enmienda 2, 8.7).',
           'mapas_corregidos': {}, 'control': {}, 'fechas': {}}

    # A. conversión
    conv_dir = os.path.join(a.nuevos, 'convertidos')
    convertidos = {}
    for d in FECHAS:
        src = os.path.join(a.nuevos, f'SD_{d}.tif')
        dst = os.path.join(conv_dir, f'SD_{d}_1m.tif')
        info = convertir(src, dst)
        info['md5_original'] = md5(src)
        res['mapas_corregidos'][d] = info
        convertidos[d] = dst
        print(f'Convertido {d}: {info["pixeles_validos"]} píxeles válidos', flush=True)

    # capas estáticas (una sola vez)
    sx_arrays, _, _ = G.load_sx_arrays()
    ctx_arrays, transform_5m, _ = G.load_context_5m()
    sce_index = G.build_sce_index()
    ctx = (sx_arrays, ctx_arrays, transform_5m, sce_index)
    df = pd.read_csv(csv_path, dtype={'date': str})

    with rasterio.open(G.DEM_1M_PATH) as src_dem:
        # B. control
        fp = G.SNOW_DIR / f'SD_{a.control}_1m.tif'
        tiles, _ = teselas_fecha(fp, a.control, ctx, src_dem)
        existentes = sorted(df.loc[df.date == a.control, 'tile_id'])
        nombres = sorted(t[0] for t in tiles)
        dmax, distintas = 0.0, []
        if nombres == existentes:
            for t in tiles:
                d_img = dif_max(t[1], np.load(ds / 'images' / t[0]))
                d_msk = dif_max(t[2], np.load(ds / 'masks' / t[0]))
                dmax = max(dmax, d_img, d_msk)
                if max(d_img, d_msk) > a.tol:
                    distintas.append(t[0])
        else:
            distintas = ['(nombres distintos)']
        res['control'] = {'fecha': a.control, 'teselas_existentes': len(existentes),
                          'teselas_regeneradas': len(nombres), 'diferencia_maxima': dmax,
                          'tolerancia': a.tol, 'distintas': distintas[:20],
                          'superado': nombres == existentes and not distintas}
        print(f"Control {a.control}: {len(nombres)} teselas regeneradas frente a {len(existentes)} existentes; "
              f"distintas: {len(distintas)}; diferencia máxima {dmax:.3g}", flush=True)
        if not res['control']['superado']:
            raise SystemExit('STOP: la regeneración no reproduce las teselas existentes. No se escribe nada.')
        if a.solo_control:
            print('Solo control: superado. No se escribe nada.')
            return

        # C. regeneración
        geo = pd.read_csv(a.georef, dtype={'date': str})
        dem_arr = src_dem.read(1).astype(np.float64)
        nuevas_csv, nuevas_geo, backup = [], [], ds / '_reemplazadas_mapas2024'
        for d in FECHAS:
            tiles, (ro, co) = teselas_fecha(convertidos[d], d, ctx, src_dem)
            viejas = list(df.loc[df.date == d, 'tile_id'])
            for sub in ('images', 'masks', 'sat_v2', 'clim_v1'):
                for t in viejas:
                    p = ds / sub / t
                    if p.exists():
                        (backup / sub).mkdir(parents=True, exist_ok=True)
                        shutil.move(str(p), str(backup / sub / t))
            difs = []
            for name, img, mask, vf, ty, tx in tiles:
                np.save(ds / 'images' / name, img)
                np.save(ds / 'masks' / name, mask)
                nuevas_csv.append({'tile_id': name, 'date': d, 'year': 2024, 'source': 'lidar',
                                   'valid_frac': vf, 'exp_temporal_split': 'val'})
                rr, cc = math.floor(ro) + ty, math.floor(co) + tx
                nuevas_geo.append({'tile_id': name, 'date': d, 'ty': ty, 'tx': tx,
                                   'row_real': rr, 'col_real': cc, 'exp_spatial_split': ''})
                ref = dem_arr[rr:rr + G.TILE_SIZE, cc:cc + G.TILE_SIZE]
                ok = np.isfinite(img[0]) & np.isfinite(ref) & (img[0] > -100) & (ref > -100)
                if src_dem.nodata is not None:
                    ok &= ref != src_dem.nodata
                if ref.shape == img[0].shape and ok.sum() >= 100:
                    difs.append(float(np.median(np.abs(img[0][ok] - ref[ok]))))
            vf = [t[3] for t in tiles]
            res['fechas'][d] = {'teselas_antes': len(viejas), 'teselas_despues': len(tiles),
                                'valid_frac_media': round(float(np.mean(vf)), 4) if vf else None,
                                'georef_dif_mediana_max_m': round(max(difs), 4) if difs else None}
            print(f'{d}: {len(viejas)} teselas antes, {len(tiles)} después', flush=True)
            if not difs or max(difs) > TOL_GEOREF:
                raise SystemExit(f'STOP: la georreferencia de {d} no cuadra con el MDE. '
                                 f'Teselas antiguas en {backup}; revisar antes de seguir.')

        # CSV y georreferencia (con copia de la versión anterior)
        shutil.copy(csv_path, ds / 'dataset_v4_ms_sx200_v1.csv')
        shutil.copy(a.georef, a.georef.replace('.csv', '_v1.csv'))
        df2 = pd.concat([df[~df.date.isin(FECHAS)], pd.DataFrame(nuevas_csv)], ignore_index=True)
        geo2 = pd.concat([geo[~geo.date.isin(FECHAS)], pd.DataFrame(nuevas_geo)], ignore_index=True)
        df2.to_csv(csv_path, index=False)
        geo2.to_csv(a.georef, index=False)
        res['totales'] = {'teselas_antes': len(df), 'teselas_despues': len(df2),
                          'val_antes': int((df.exp_temporal_split == 'val').sum()),
                          'val_despues': int((df2.exp_temporal_split == 'val').sum())}

    os.makedirs(os.path.dirname(a.resumen) or '.', exist_ok=True)
    json.dump(res, open(a.resumen, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(json.dumps(res['totales'], ensure_ascii=False))
    print(f'Resumen: {a.resumen}. Siguiente: patch_sat_channels.py y patch_clim_channels.py (solo teselas nuevas).')


if __name__ == '__main__':
    main()
