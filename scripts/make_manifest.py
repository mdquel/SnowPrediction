r"""
Manifiesto del dataset: huella MD5 de cada fichero, para comprobar copias
==========================================================================

Sirve para asegurar que la copia del dataset que se usa para entrenar (Kaggle,
Vast.ai) es exactamente la del disco local con la que se obtuvieron los
resultados del articulo. Sin dependencias fuera de la biblioteca estandar.

Tres modos:

    crear      recorre el dataset y escribe un CSV con subcarpeta, nombre,
               tamanyo y MD5 de cada fichero
    comparar   compara dos manifiestos (p. ej. el local y el de Kaggle)
    comprobar  calcula el manifiesto de una copia y lo compara con uno dado;
               termina con error si algo no coincide (para Vast.ai)

Subcarpetas por defecto: images, masks y sat_v2 (las que existan), mas los
ficheros sueltos de la raiz (los CSV del split).

Uso (desde la raiz del repo)
----------------------------
    python scripts/make_manifest.py crear dataset_v4_ms_sx200 analysis/manifest_dataset_v4_ms_sx200.csv
    python scripts/make_manifest.py comparar analysis/manifest_dataset_v4_ms_sx200.csv manifest_kaggle.csv
    python scripts/make_manifest.py comprobar /ruta/a/dataset_v4_ms_sx200 analysis/manifest_dataset_v4_ms_sx200.csv
"""

import argparse
import csv
import hashlib
import os
import sys
import time

SUBDIRS = ('images', 'masks', 'sat_v2')
FIELDS = ('subdir', 'name', 'bytes', 'md5')


def md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def build(root, subdirs=SUBDIRS):
    rows, t0 = [], time.time()
    jobs = [('.', n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n))]
    for sd in subdirs:
        d = os.path.join(root, sd)
        if os.path.isdir(d):
            jobs += [(sd, n) for n in sorted(os.listdir(d))]
    for i, (sd, n) in enumerate(jobs, 1):
        p = os.path.join(root, n) if sd == '.' else os.path.join(root, sd, n)
        rows.append({'subdir': sd, 'name': n, 'bytes': os.path.getsize(p), 'md5': md5(p)})
        if i % 1000 == 0:
            print(f'  {i}/{len(jobs)} ficheros ({(time.time() - t0) / 60:.1f} min)', flush=True)
    return rows


def write(rows, out):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read(path):
    with open(path, encoding='utf-8') as f:
        return {(r['subdir'], r['name']): r for r in csv.DictReader(f)}


def compare(ref, other, only_subdirs=None):
    """Diferencias por subcarpeta; solo las subcarpetas presentes en ambos
    (o las pedidas), para poder comparar con una copia parcial."""
    subs = only_subdirs or sorted({k[0] for k in ref} & {k[0] for k in other})
    rep, ok = {}, True
    for sd in subs:
        a = {k: v for k, v in ref.items() if k[0] == sd}
        b = {k: v for k, v in other.items() if k[0] == sd}
        falta = sorted(k[1] for k in a.keys() - b.keys())
        sobra = sorted(k[1] for k in b.keys() - a.keys())
        distinto = sorted(k[1] for k in a.keys() & b.keys() if a[k]['md5'] != b[k]['md5'])
        rep[sd] = {'ref': len(a), 'copia': len(b), 'faltan': falta, 'sobran': sobra,
                   'distintos': distinto}
        ok &= not (falta or sobra or distinto)
    return ok, rep, subs


def show(ok, rep, subs):
    for sd in subs:
        r = rep[sd]
        print(f"  {sd:8s} referencia {r['ref']:5d}  copia {r['copia']:5d}  faltan {len(r['faltan'])}  "
              f"sobran {len(r['sobran'])}  distintos {len(r['distintos'])}")
        for k in ('faltan', 'sobran', 'distintos'):
            if r[k]:
                print(f"      {k} (primeros 10): {r[k][:10]}")
    print('COINCIDEN' if ok else 'NO COINCIDEN')


def main():
    ap = argparse.ArgumentParser(description='Manifiesto MD5 del dataset')
    sub = ap.add_subparsers(dest='modo', required=True)
    c = sub.add_parser('crear');     c.add_argument('root');     c.add_argument('out')
    k = sub.add_parser('comparar');  k.add_argument('ref');      k.add_argument('otro')
    v = sub.add_parser('comprobar'); v.add_argument('root');     v.add_argument('ref')
    for p in (k, v):
        p.add_argument('--subdirs', nargs='*', help='limitar la comparacion a estas subcarpetas')
    a = ap.parse_args()

    if a.modo == 'crear':
        rows = build(a.root)
        write(rows, a.out)
        print(f'{len(rows)} ficheros -> {a.out}')
    elif a.modo == 'comparar':
        ok, rep, subs = compare(read(a.ref), read(a.otro), a.subdirs)
        show(ok, rep, subs)
        sys.exit(0 if ok else 1)
    else:
        ref = read(a.ref)
        subs = a.subdirs or sorted({s for s, _ in ref})
        other = {(r['subdir'], r['name']): r for r in build(a.root, [s for s in subs if s != '.'])}
        ok, rep, subs = compare(ref, other, subs)
        show(ok, rep, subs)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
