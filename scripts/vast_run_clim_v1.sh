#!/usr/bin/env bash
# =============================================================================
# Experimento de climatologia v1 en Vast.ai (paso 2)
# =============================================================================
#
# Supone la instancia ya preparada en el paso 1 (vast_run_sat_v2.sh): repo en
# /workspace/SnowPrediction con dataset_v4_ms_sx200/{images,masks,CSV,sat_v2}
# enlazados, y clim_v1 generado con data/patch_clim_channels.py.
#
#   1. enlaza clim_v1 y comprueba el manifiesto (images, masks, sat_v2)
#   2. registra el entorno
#   3. entrena y evalua Bclim con las semillas 7, 42 y 123 (salta las hechas);
#      cada una se evalua con climatologia y con ella apagada (_sinclim)
#   4. evalua la climatologia sola en las mismas teselas de test
#   5. resume: Bclim con y sin climatologia, B del paso 1 y climatologia sola
#   6. empaqueta results/clim_v1
#
# Uso:
#   ONLY="s7" bash scripts/vast_run_clim_v1.sh    # una semilla (prueba)
#   bash scripts/vast_run_clim_v1.sh              # las tres
# =============================================================================
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO="$WORK/SnowPrediction"
CLIM_SRC="${CLIM_SRC:-$WORK/kaggle_data/dataset_v4_ms_sx200/dataset_v4_ms_sx200/clim_v1}"
SEEDS=(s7 s42 s123)
[ -n "${ONLY:-}" ] && SEEDS=($ONLY)
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"
COMMIT=$(git rev-parse --short HEAD)
DS="$REPO/dataset_v4_ms_sx200"

# --- 1. datos -------------------------------------------------------------------
[ "$(ls "$CLIM_SRC" | wc -l)" -eq 4892 ] || { echo "STOP: $CLIM_SRC no tiene 4892 teselas"; exit 1; }
ln -sfn "$CLIM_SRC" "$DS/clim_v1"
for x in images masks sat_v2 clim_v1; do [ -e "$DS/$x" ] || { echo "STOP: falta $DS/$x"; exit 1; }; done
log "Comprobando el manifiesto (images, masks, sat_v2)"
python scripts/make_manifest.py comprobar "$DS" analysis/manifest_dataset_v4_ms_sx200.csv --subdirs images masks sat_v2
python scripts/generate_clim_v1_configs.py

# --- 2. entorno -------------------------------------------------------------------
OUT="$REPO/results/clim_v1"
mkdir -p "$OUT/logs"
{
  echo "fecha_inicio: $(date -u '+%F %T') UTC"
  echo "commit: $COMMIT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  python -c "import torch, platform; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cudnn:', torch.backends.cudnn.version()); print('python:', platform.python_version())"
  echo "cpu: $(nproc) hilos visibles"
} >> "$OUT/entorno.txt"
[ -f "$OUT/tiempos.csv" ] || echo "ejecucion,inicio_utc,fin_utc,minutos,gpu" > "$OUT/tiempos.csv"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# --- 3. entrenamientos -----------------------------------------------------------
for s in "${SEEDS[@]}"; do
  name="resunetpp_clim1_Bclim_${s}"
  if [ -f "$OUT/$name/${name}_sinclim_metrics.json" ]; then log "$name ya hecha, se salta"; continue; fi
  log "=== $name ==="
  t0=$(date +%s); i0=$(date -u '+%F %T')
  python main.py --config "configs/clim_v1/${name}.yaml" --mode both 2>&1 | tee "$OUT/logs/${name}.log"
  echo "$name,$i0,$(date -u '+%F %T'),$(( ($(date +%s) - t0) / 60 )),$GPU" >> "$OUT/tiempos.csv"
  for f in "${name}_metrics.json" "${name}_sinclim_metrics.json"; do
    [ -f "$OUT/$name/$f" ] || { echo "STOP: falta $OUT/$name/$f"; exit 1; }
  done
done

# --- 4. climatologia sola --------------------------------------------------------
if [ ! -f "$OUT/climatologia_sola/climatologia_sola_metrics.json" ]; then
  log "Climatologia sola"
  python scripts/eval_clim_only.py --config configs/clim_v1/resunetpp_clim1_Bclim_s7.yaml 2>&1 | tee "$OUT/logs/climatologia_sola.log"
fi

# --- 5. resumen ---------------------------------------------------------------------
log "Resumen"
python - <<'EOF'
import json, os
import numpy as np
seeds, keys = (7, 42, 123), ('R2', 'SPAEF', 'MAE', 'RMSE', 'Bias')
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
def arm(path_of):
    rows = {s: load(path_of(s)) for s in seeds}
    rows = {s: m for s, m in rows.items() if m}
    if not rows:
        return None
    out = {'n': len(rows), 'por_semilla': {s: {k: m.get(k) for k in keys} for s, m in rows.items()}}
    for k in ('R2', 'SPAEF'):
        v = [m[k] for m in rows.values()]
        out[k] = {'media': round(float(np.mean(v)), 4), 'min': round(float(min(v)), 4),
                  'max': round(float(max(v)), 4)}
    return out
C = 'results/clim_v1/resunetpp_clim1_Bclim_s{s}/resunetpp_clim1_Bclim_s{s}{suf}_metrics.json'
Bp = 'results/sat_v2/resunetpp_sat2_B_s{s}/resunetpp_sat2_B_s{s}{suf}_metrics.json'
res = {}
for lect, suf in (('mejor_epoca', ''), ('ultima_epoca', '_last')):
    res[lect] = {
        'Bclim_con_climatologia': arm(lambda s: C.format(s=s, suf=suf)),
        'Bclim_sin_climatologia': arm(lambda s: C.format(s=s, suf=suf + '_sinclim')),
        'B_paso1': arm(lambda s: Bp.format(s=s, suf=suf)),
    }
cs = load('results/clim_v1/climatologia_sola/climatologia_sola_metrics.json')
res['climatologia_sola'] = {k: cs.get(k) for k in keys} if cs else None
res['referencia_repeticion'] = 'delta R2 de repetir el mismo entrenamiento: 0.0125 (hallazgo 14)'
json.dump(res, open('results/clim_v1/resumen.json', 'w'), indent=2)
for lect in ('mejor_epoca', 'ultima_epoca'):
    print(f'  [{lect}]')
    for a, v in res[lect].items():
        if v:
            print(f"    {a:24s} R2 {v['R2']['media']:.3f} [{v['R2']['min']:.3f}-{v['R2']['max']:.3f}]  "
                  f"SPAEF {v['SPAEF']['media']:.3f} [{v['SPAEF']['min']:.3f}-{v['SPAEF']['max']:.3f}]  (n={v['n']})")
if cs:
    print(f"  climatologia sola        R2 {cs['R2']:.3f}  SPAEF {cs['SPAEF']:.3f}")
EOF

# --- 6. empaquetado -------------------------------------------------------------------
echo "fecha_fin: $(date -u '+%F %T') UTC" >> "$OUT/entorno.txt"
tar -czf "$WORK/resultados_clim_v1.tar.gz" -C "$REPO/results" clim_v1
log "Resultados en $WORK/resultados_clim_v1.tar.gz ($(du -h "$WORK/resultados_clim_v1.tar.gz" | cut -f1))"
