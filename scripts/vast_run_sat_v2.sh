#!/usr/bin/env bash
# =============================================================================
# Experimento de canales satelitales v2 en Vast.ai (paso 1)
# =============================================================================
#
# Lanza en serie las 9 ejecuciones de configs/sat_v2 (brazos base, A y B, con
# las semillas 7, 42 y 123) en el split temporal, y deja todo lo necesario para
# analizarlas: metricas, predicciones, pesos, registros, tiempos y entorno.
#
# Pasos:
#   1. dependencias (sin reinstalar torch, que ya trae la imagen)
#   2. codigo: clona la rama del fork
#   3. datos: descarga de Kaggle martinezdequel/articulo-1-data y localiza
#      dataset_v4_ms_sx200 (images, masks, CSV) y sat_v2
#   4. COMPRUEBA EL MANIFIESTO (analysis/manifest_dataset_v4_ms_sx200.csv,
#      creado en el disco local): si alguna tesela no coincide, se para
#   5. registra GPU, CUDA y versiones
#   6. entrena y evalua las 9 ejecuciones, intercaladas por semilla
#      (base, A, B con s7; luego s42; luego s123), saltando las ya hechas
#   7. resume las metricas en results/sat_v2/resumen.json
#   8. empaqueta results/sat_v2 en un .tar.gz para descargarlo
#
# Requisitos en la instancia:
#   * ~/.kaggle/kaggle.json (token de Kaggle) con permisos 600
#   * GITHUB_TOKEN en el entorno solo si el fork es privado
#   * disco: unos 120 GB libres (zip de Kaggle + datos descomprimidos + resultados)
#
# Uso:
#   bash vast_run_sat_v2.sh            # todo
#   ONLY="base_s7" bash vast_run_sat_v2.sh   # una sola ejecucion (prueba)
# =============================================================================
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO_URL="${REPO_URL:?pon REPO_URL=github.com/<usuario>/<repo>.git}"
BRANCH="${BRANCH:-feat/save-predictions}"
KAGGLE_DS="martinezdequel/articulo-1-data"
REPO="$WORK/SnowPrediction"
DATA="$WORK/kaggle_data"
RUNS=(base_s7 A_s7 B_s7 base_s42 A_s42 B_s42 base_s123 A_s123 B_s123)
[ -n "${ONLY:-}" ] && RUNS=($ONLY)

log() { echo "[$(date '+%F %T')] $*"; }

# --- 1. dependencias ----------------------------------------------------------
log "Dependencias"
python -c "import torch; assert torch.cuda.is_available(), 'sin GPU visible'; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
pip install -q pandas scikit-learn scipy matplotlib seaborn tqdm pyyaml opencv-python-headless kaggle

# --- 2. codigo ----------------------------------------------------------------
if [ ! -d "$REPO/.git" ]; then
  log "Clonando $BRANCH"
  if [ -n "${GITHUB_TOKEN:-}" ]; then URL="https://${GITHUB_TOKEN}@${REPO_URL}"; else URL="https://${REPO_URL}"; fi
  git clone --branch "$BRANCH" --single-branch "$URL" "$REPO"
fi
cd "$REPO"
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q origin "$BRANCH"
COMMIT=$(git rev-parse --short HEAD)
log "Codigo en el commit $COMMIT"
[ -f analysis/manifest_dataset_v4_ms_sx200.csv ] || { echo "STOP: falta analysis/manifest_dataset_v4_ms_sx200.csv en el repositorio"; exit 1; }

# --- 3. datos -----------------------------------------------------------------
[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "STOP: falta ~/.kaggle/kaggle.json"; exit 1; }
chmod 600 "$HOME/.kaggle/kaggle.json"
mkdir -p "$DATA"
if [ ! -f "$DATA/.descomprimido" ]; then
  log "Descargando $KAGGLE_DS"
  kaggle datasets download -d "$KAGGLE_DS" -p "$DATA"
  log "Descomprimiendo"
  (cd "$DATA" && unzip -q ./*.zip && rm -f ./*.zip && touch .descomprimido)
fi

# carpeta con images/, masks/ y el CSV temporal; y carpeta sat_v2 con los .npy
SRC=$(find "$DATA" -type f -name dataset_v4_ms_sx200.csv -printf '%h\n' | while read -r d; do
        [ -d "$d/images" ] && [ -d "$d/masks" ] && echo "$d"; done | head -1)
SAT=$(find "$DATA" -type d -name sat_v2 | while read -r d; do
        ls "$d"/*.npy >/dev/null 2>&1 && echo "$d"; done | head -1)
[ -n "$SRC" ] || { echo "STOP: no encuentro dataset_v4_ms_sx200 (images, masks y CSV)"; exit 1; }
[ -n "$SAT" ] || { echo "STOP: no encuentro la carpeta sat_v2 con los .npy"; exit 1; }
log "Dataset: $SRC"
log "sat_v2 : $SAT"

# disposicion que esperan las configs: <repo>/dataset_v4_ms_sx200/{images,masks,CSV,sat_v2}
DS="$REPO/dataset_v4_ms_sx200"
mkdir -p "$DS"
for x in images masks dataset_v4_ms_sx200.csv dataset_v4_ms_sx200_spatial.csv; do ln -sfn "$SRC/$x" "$DS/$x"; done
ln -sfn "$SAT" "$DS/sat_v2"

# --- 4. manifiesto -------------------------------------------------------------
log "Comprobando el manifiesto (images, masks, sat_v2)"
python scripts/make_manifest.py comprobar "$DS" analysis/manifest_dataset_v4_ms_sx200.csv --subdirs images masks sat_v2
python - <<'EOF'
import csv, hashlib, os, sys
ref = {r['name']: r['md5'] for r in csv.DictReader(open('analysis/manifest_dataset_v4_ms_sx200.csv', encoding='utf-8'))
       if r['subdir'] == '.'}
for n in ('dataset_v4_ms_sx200.csv', 'dataset_v4_ms_sx200_spatial.csv'):
    h = hashlib.md5(open(os.path.join('dataset_v4_ms_sx200', n), 'rb').read()).hexdigest()
    ok = ref.get(n) == h
    print(f'  {n}: {"OK" if ok else "NO COINCIDE"}')
    if not ok:
        sys.exit('STOP: los CSV del split no coinciden con el manifiesto')
EOF

# --- 5. entorno ------------------------------------------------------------------
OUT="$REPO/results/sat_v2"
mkdir -p "$OUT/logs"
{
  echo "fecha_inicio: $(date -u '+%F %T') UTC"
  echo "commit: $COMMIT"
  echo "kaggle_dataset: $KAGGLE_DS"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  python -c "import torch, platform; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cudnn:', torch.backends.cudnn.version()); print('python:', platform.python_version())"
  echo "cpu: $(nproc) nucleos"
} > "$OUT/entorno.txt"
cat "$OUT/entorno.txt"
[ -f "$OUT/tiempos.csv" ] || echo "ejecucion,inicio_utc,fin_utc,minutos,gpu" > "$OUT/tiempos.csv"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# --- 6. ejecuciones ----------------------------------------------------------------
for r in "${RUNS[@]}"; do
  name="resunetpp_sat2_${r}"
  cfg="configs/sat_v2/${name}.yaml"
  if [ -f "$OUT/$name/${name}_metrics.json" ]; then log "$name ya hecha, se salta"; continue; fi
  log "=== $name ==="
  t0=$(date +%s); i0=$(date -u '+%F %T')
  python main.py --config "$cfg" --mode both 2>&1 | tee "$OUT/logs/${name}.log"
  t1=$(date +%s)
  echo "$name,$i0,$(date -u '+%F %T'),$(( (t1 - t0) / 60 )),$GPU" >> "$OUT/tiempos.csv"
  [ -f "$OUT/$name/${name}_metrics.json" ] || { echo "STOP: $name no ha dejado metrics.json"; exit 1; }
done

# --- 7. resumen --------------------------------------------------------------------
log "Resumen"
python - <<'EOF'
import json, os
import numpy as np
OUT = 'results/sat_v2'
arms, seeds = ('base', 'A', 'B'), (7, 42, 123)
keys = ('R2', 'SPAEF', 'MAE', 'RMSE', 'Bias', 'R2_full')
res = {'por_ejecucion': {}, 'por_brazo': {}, 'diferencias_de_medias': {}}
for a in arms:
    vals = {k: [] for k in keys}
    for s in seeds:
        n = f'resunetpp_sat2_{a}_s{s}'
        p = os.path.join(OUT, n, f'{n}_metrics.json')
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        res['por_ejecucion'][n] = {k: m.get(k) for k in keys}
        for k in keys:
            if m.get(k) is not None:
                vals[k].append(m[k])
    res['por_brazo'][a] = {k: {'media': round(float(np.mean(v)), 4), 'min': round(float(np.min(v)), 4),
                               'max': round(float(np.max(v)), 4), 'n': len(v)}
                           for k, v in vals.items() if v}
for x, y in (('A', 'base'), ('B', 'A'), ('B', 'base')):
    bx, by = res['por_brazo'].get(x, {}), res['por_brazo'].get(y, {})
    res['diferencias_de_medias'][f'{x}-{y}'] = {k: round(bx[k]['media'] - by[k]['media'], 4)
                                                 for k in ('R2', 'SPAEF') if k in bx and k in by}
res['referencia_repeticion'] = 'delta R2 de repetir el mismo entrenamiento: 0.0125 (hallazgo 14)'
json.dump(res, open(os.path.join(OUT, 'resumen.json'), 'w'), indent=2)
for a, v in res['por_brazo'].items():
    print(f"  {a:5s} R2 {v['R2']['media']:.3f} [{v['R2']['min']:.3f}-{v['R2']['max']:.3f}]  "
          f"SPAEF {v['SPAEF']['media']:.3f} [{v['SPAEF']['min']:.3f}-{v['SPAEF']['max']:.3f}]  (n={v['R2']['n']})")
for k, v in res['diferencias_de_medias'].items():
    print(f'  {k}: {v}')
EOF

# --- 8. empaquetado -------------------------------------------------------------------
echo "fecha_fin: $(date -u '+%F %T') UTC" >> "$OUT/entorno.txt"
tar -czf "$WORK/resultados_sat_v2.tar.gz" -C "$REPO/results" sat_v2
log "Resultados en $WORK/resultados_sat_v2.tar.gz ($(du -h "$WORK/resultados_sat_v2.tar.gz" | cut -f1))"
