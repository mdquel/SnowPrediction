r"""
Descarga selectiva del dataset de Kaggle, fichero a fichero
===========================================================

Kaggle no genera el zip completo de articulo-1-data (43 GB): la descarga del
dataset entero devuelve "404 - No gcs url found". Los ficheros sueltos si se
pueden bajar, asi que este script baja solo lo que necesita el entrenamiento:

    dataset_v4_ms_sx200/...   images/, masks/ y los dos CSV del split
    sat_v2/...                los .npy de los canales satelitales v2

y deja fuera Data/izas/LiDAR. Conserva las rutas de Kaggle bajo <destino>, de
modo que vast_run_sat_v2.sh los encuentra igual que si viniesen del zip.

* En paralelo con procesos (--procesos, 8 por defecto), con reintentos. Con
  hilos el cliente de Kaggle se queda bloqueado y no baja nada.
* --limite N baja solo los N primeros (prueba de velocidad).
* Relanzable: salta los ficheros que ya estan con el tamanyo esperado.
* Si Kaggle entrega un fichero comprimido (<nombre>.zip), lo descomprime.
* La integridad la comprueba despues el manifiesto (MD5 de cada fichero).

Requiere el cliente kaggle 1.6.x (pip install "kaggle==1.6.17").

Uso
---
    python scripts/kaggle_fetch_files.py martinezdequel/articulo-1-data /workspace/kaggle_data
"""

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from kaggle.api.kaggle_api_extended import KaggleApi

PREFIJOS = ('dataset_v4_ms_sx200/', 'sat_v2/')
CSVS = ('dataset_v4_ms_sx200.csv', 'dataset_v4_ms_sx200_spatial.csv')
_API = None


def api():
    """Un cliente por proceso."""
    global _API
    if _API is None:
        _API = KaggleApi()
        _API.authenticate()
    return _API


def wanted(name):
    if not name.startswith(PREFIJOS):
        return False
    parts = name.split('/')
    return ('images' in parts or 'masks' in parts or name.startswith('sat_v2/')
            or os.path.basename(name) in CSVS)


def listar(ds):
    files, tok, pag = [], None, 0
    while True:
        r = api().dataset_list_files(ds, page_token=tok, page_size=1000)
        for f in r.files:
            if wanted(f.name):
                files.append((f.name, int(getattr(f, 'totalBytes', 0) or 0)))
        pag += 1
        tok = getattr(r, 'nextPageToken', None) or getattr(r, 'next_page_token', None)
        print(f'  pagina {pag}: {len(files)} ficheros seleccionados', flush=True)
        if not tok:
            return files


def bajar(ds, name, size, out, intentos=5):
    dest = os.path.join(out, name)
    if os.path.exists(dest) and (size == 0 or os.path.getsize(dest) == size):
        return 'ya'
    d = os.path.dirname(dest)
    os.makedirs(d, exist_ok=True)
    for i in range(intentos):
        try:
            api().dataset_download_file(ds, name, path=d, force=True, quiet=True)
            z = dest + '.zip'
            if os.path.exists(z):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
                os.remove(z)
            if os.path.exists(dest):
                return 'ok'
            raise RuntimeError('no aparece el fichero tras la descarga')
        except Exception as e:  # noqa: BLE001
            if i == intentos - 1:
                return f'ERROR {e}'
            time.sleep(2 * (i + 1))


def main():
    ap = argparse.ArgumentParser(description='Descarga selectiva de Kaggle')
    ap.add_argument('dataset')
    ap.add_argument('destino')
    ap.add_argument('--procesos', type=int, default=8)
    ap.add_argument('--limite', type=int, default=0)
    a = ap.parse_args()

    print('Listando ficheros del dataset')
    files = listar(a.dataset)
    if a.limite:
        files = files[:a.limite]
    gb = sum(s for _, s in files) / 1e9
    print(f'{len(files)} ficheros, {gb:.1f} GB')

    t0, n, err = time.time(), 0, []
    with ProcessPoolExecutor(a.procesos) as ex:
        futs = {ex.submit(bajar, a.dataset, f, s, a.destino): f for f, s in files}
        for fut in as_completed(futs):
            n += 1
            r = fut.result()
            if r.startswith('ERROR'):
                err.append((futs[fut], r))
            if n % 100 == 0 or n == len(files):
                print(f'  {n}/{len(files)}  ({(time.time() - t0) / 60:.1f} min, {len(err)} errores)',
                      flush=True)
    for f, r in err[:20]:
        print(f'  {f}: {r}')
    if err:
        sys.exit(f'STOP: {len(err)} ficheros sin bajar; relanza el script para reintentarlos')
    gb_s = sum(s for _, s in files) / 1e9
    print(f'Descarga completa: {gb_s:.1f} GB en {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
