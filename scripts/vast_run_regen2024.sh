#!/usr/bin/env bash
# =============================================================================
# Reentrenamiento de B y de la red con climatologia tras regenerar las teselas
# de marzo de 2024 (protocolo 2026, apartados 8.7 y 8.9)
# =============================================================================
#
# Supone la instancia de los pasos 1 y 2 y las teselas ya regeneradas con
# data/regen_mapas_2024.py, con sat_v2 y clim_v1 de las teselas nuevas
# (patch_sat_channels.py y patch_clim_channels.py).
#
#   1. datos: enlaza images, masks, sat_v2, clim_v1 y los CSV en
#      <repo>/dataset_v4_ms_sx200 y comprueba 4926 teselas (465 de validacion)
#   2. manifiesto nuevo y comparacion con el anterior: solo pueden cambiar las
#      teselas del 07-03-2024 y del 14-03-2024 y el CSV del split temporal
#   3. configuraciones: copias de las de B (sat_v2) y de la red con
#      climatologia (clim_v1) cambiando SOLO las carpetas de salida
#      (results/regen2024); los nombres de los modelos no cambian
#   4. entrena y evalua las 6 ejecuciones, intercaladas por semilla (salta
#      las hechas)
#   5. climatologia sola (control: debe coincidir con la anterior)
#   6. combinacion (climatologia + nivel de la nueva B)
#   7. resumen antes y despues (mejor epoca y ultima)
#   8. reune los ficheros que hay que llevar al repositorio y empaqueta
#
# Uso (desde /workspace/SnowPrediction):
#   ONLY="B_s7" bash scripts/vast_run_regen2024.sh       # una ejecucion (prueba)
#   nohup bash scripts/vast_run_regen2024.sh > /workspace/regen2024.out 2>&1 &
# =============================================================================
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO="$WORK/SnowPrediction"
SRC="${SRC:-$WORK/kaggle_data/dataset_v4_ms_sx200/dataset_v4_ms_sx200}"
N_TESELAS=4926
RUNS=(B_s7 Bclim_s7 B_s42 Bclim_s42 B_s123 Bclim_s123)
[ -n "${ONLY:-}" ] && RUNS=($ONLY)
log() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO"
COMMIT=$(git rev-parse --short HEAD)
OUT="$REPO/results/regen2024"
mkdir -p "$OUT/logs"
log "Codigo en el commit $COMMIT; espacio libre: $(df -h "$WORK" | awk 'NR==2{print $4}')"

# --- 1. datos -------------------------------------------------------------------
[ -f analysis/regen_mapas_2024.json ] || { echo "STOP: falta analysis/regen_mapas_2024.json"; exit 1; }
for x in images masks sat_v2 clim_v1; do
  n=$(ls "$SRC/$x" | wc -l)
  [ "$n" -eq "$N_TESELAS" ] || { echo "STOP: $SRC/$x tiene $n teselas (se esperan $N_TESELAS)"; exit 1; }
done
DS="$REPO/dataset_v4_ms_sx200"
mkdir -p "$DS"
for x in images masks sat_v2 clim_v1 dataset_v4_ms_sx200.csv dataset_v4_ms_sx200_spatial.csv; do
  if [ -e "$DS/$x" ] && [ ! -L "$DS/$x" ]; then echo "STOP: $DS/$x existe y no es un enlace"; exit 1; fi
  ln -sfn "$SRC/$x" "$DS/$x"
done
python - <<'EOF'
import sys
import pandas as pd
d = pd.read_csv('dataset_v4_ms_sx200/dataset_v4_ms_sx200.csv', dtype={'date': str})
n = d.exp_temporal_split.value_counts().to_dict()
print(f'  CSV: {len(d)} teselas; por conjunto {n}')
print(f"  marzo de 2024: {d[d.date.isin(['20240307', '20240314'])].groupby('date').size().to_dict()}")
if len(d) != 4926 or n.get('train') != 4241 or n.get('val') != 465 or n.get('test') != 220:
    sys.exit('STOP: el CSV no es el regenerado (4926 = 4241 + 465 + 220)')
EOF

# --- 2. manifiesto ------------------------------------------------------------------
MAN_OLD=analysis/manifest_dataset_v4_ms_sx200.csv
MAN_NEW=analysis/manifest_dataset_v4_ms_sx200_regen2024.csv
[ -f "$MAN_OLD" ] || { echo "STOP: falta $MAN_OLD"; exit 1; }
if [ ! -f "$MAN_NEW" ]; then
  log "Manifiesto nuevo (images, masks, sat_v2 y CSV)"
  python scripts/make_manifest.py crear "$DS" "$MAN_NEW"
fi
python - "$MAN_OLD" "$MAN_NEW" <<'EOF'
import csv, sys
def read(p):
    return {(r['subdir'], r['name']): r['md5'] for r in csv.DictReader(open(p, encoding='utf-8'))}
old, new = read(sys.argv[1]), read(sys.argv[2])
def permitido(k):
    sd, n = k
    return (sd in ('images', 'masks', 'sat_v2') and n[:8] in ('20240307', '20240314')) \
        or k == ('.', 'dataset_v4_ms_sx200.csv')
falta = [k for k in old.keys() - new.keys()]
sobra = [k for k in new.keys() - old.keys()]
dist = [k for k in old.keys() & new.keys() if old[k] != new[k]]
mal = [k for k in falta + sobra + dist if not permitido(k)]
print(f'  manifiesto: faltan {len(falta)}, nuevas {len(sobra)}, distintas {len(dist)}; '
      f'fuera de marzo de 2024: {len(mal)}')
if mal:
    print('   ', mal[:20])
    sys.exit('STOP: han cambiado ficheros que no son del 7 o del 14 de marzo de 2024')
iguales = sum(1 for k in old.keys() & new.keys() if old[k] == new[k])
print(f'  ficheros identicos al manifiesto anterior: {iguales}')
EOF

# --- 3. configuraciones -----------------------------------------------------------------
python scripts/generate_sat_v2_configs.py
python scripts/generate_clim_v1_configs.py
python - <<'EOF'
import os
import yaml
os.makedirs('configs/regen2024', exist_ok=True)
for s in (7, 42, 123):
    for src, sub in ((f'configs/sat_v2/resunetpp_sat2_B_s{s}.yaml', 'sat_v2'),
                     (f'configs/clim_v1/resunetpp_clim1_Bclim_s{s}.yaml', 'clim_v1')):
        c = yaml.safe_load(open(src, encoding='utf-8'))
        name = c['output']['model_name']
        c['output'] = {'models_dir': 'results/regen2024/weights',
                       'results_dir': f'results/regen2024/{sub}/{name}',
                       'model_name': name}
        dst = f'configs/regen2024/{name}.yaml'
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(f'# Protocolo 2026, 8.9: copia de {src}; solo cambian las carpetas de salida\n')
            yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
        print(f'  {dst}')
EOF

# --- 4. entorno y entrenamientos ---------------------------------------------------------
{
  echo "fecha_inicio: $(date -u '+%F %T') UTC"
  echo "commit: $COMMIT"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  python -c "import torch, platform; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cudnn:', torch.backends.cudnn.version()); print('python:', platform.python_version())"
  echo "cpu: $(nproc) hilos visibles"
} >> "$OUT/entorno.txt"
[ -f "$OUT/tiempos.csv" ] || echo "ejecucion,inicio_utc,fin_utc,minutos,gpu" > "$OUT/tiempos.csv"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

for r in "${RUNS[@]}"; do
  case "$r" in
    B_*)     name="resunetpp_sat2_${r}";  sub=sat_v2;  need=("${name}_metrics.json") ;;
    Bclim_*) name="resunetpp_clim1_${r}"; sub=clim_v1; need=("${name}_metrics.json" "${name}_sinclim_metrics.json") ;;
    *) echo "STOP: ejecucion desconocida $r"; exit 1 ;;
  esac
  dir="$OUT/$sub/$name"
  if [ -f "$dir/${need[-1]}" ]; then log "$name ya hecha, se salta"; continue; fi
  log "=== $name ==="
  t0=$(date +%s); i0=$(date -u '+%F %T')
  python main.py --config "configs/regen2024/${name}.yaml" --mode both 2>&1 | tee "$OUT/logs/${name}.log"
  echo "$name,$i0,$(date -u '+%F %T'),$(( ($(date +%s) - t0) / 60 )),$GPU" >> "$OUT/tiempos.csv"
  for f in "${need[@]}"; do [ -f "$dir/$f" ] || { echo "STOP: falta $dir/$f"; exit 1; }; done
done
[ -n "${ONLY:-}" ] && { log "ONLY=$ONLY: hecho; no se hacen los pasos 5-8"; exit 0; }

# --- 5. climatologia sola (control) ------------------------------------------------------
CS="$OUT/clim_v1/climatologia_sola/climatologia_sola_metrics.json"
if [ ! -f "$CS" ]; then
  log "Climatologia sola"
  python scripts/eval_clim_only.py --config configs/regen2024/resunetpp_clim1_Bclim_s7.yaml 2>&1 | tee "$OUT/logs/climatologia_sola.log"
fi
python - <<'EOF'
import json, os
new = json.load(open('results/regen2024/clim_v1/climatologia_sola/climatologia_sola_metrics.json'))
p = 'results/clim_v1/climatologia_sola/climatologia_sola_metrics.json'
if os.path.exists(p):
    old = json.load(open(p))
    d = max(abs(new[k] - old[k]) for k in ('R2', 'SPAEF', 'MAE', 'RMSE') if k in new and k in old)
    print(f'  climatologia sola: diferencia maxima con la anterior {d:.2e}', '(IGUAL)' if d < 1e-6 else '(AVISO: distinta)')
else:
    print('  [aviso] no esta la climatologia sola anterior; no se compara')
EOF

# --- 6. combinacion --------------------------------------------------------------------
log "Combinacion con la nueva B"
python scripts/clim_v1_hybrid.py --clim-dir results/regen2024/clim_v1 --b-dir results/regen2024/sat_v2 \
       --out analysis/clim_v1_hibrido_regen2024.json

# --- 7. resumen antes y despues ------------------------------------------------------------
log "Resumen antes y despues"
python - <<'EOF'
import json, os
import numpy as np
seeds = (7, 42, 123)
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
def arm(fmt):
    v = {s: load(fmt.format(s=s)) for s in seeds}
    v = {s: m for s, m in v.items() if m}
    if not v:
        return None
    return {k: {'media': round(float(np.mean([m[k] for m in v.values()])), 4),
                'por_semilla': {s: round(m[k], 4) for s, m in v.items()}} for k in ('R2', 'SPAEF')}
B = '{root}/resunetpp_sat2_B_s{{s}}/resunetpp_sat2_B_s{{s}}{suf}_metrics.json'
C = '{root}/resunetpp_clim1_Bclim_s{{s}}/resunetpp_clim1_Bclim_s{{s}}{suf}_metrics.json'
def hib(p, lect):
    h = load(p)
    try:
        x = h[lect]['B_paso1']['clim_mas_nivel_de_la_red']
        return {k: {'media': x[k]['media']} for k in ('R2', 'SPAEF')}
    except (TypeError, KeyError):
        return None
res = {'referencia_repeticion': 'delta R2 de repetir el mismo entrenamiento: 0.0125 (hallazgo 14)'}
for lect, suf in (('mejor_epoca', ''), ('ultima_epoca', '_last')):
    res[lect] = {}
    for nombre, antes, despues in (
        ('B', arm(B.format(root='results/sat_v2', suf=suf)), arm(B.format(root='results/regen2024/sat_v2', suf=suf))),
        ('Bclim', arm(C.format(root='results/clim_v1', suf=suf)), arm(C.format(root='results/regen2024/clim_v1', suf=suf))),
        ('Bclim_apagada', arm(C.format(root='results/clim_v1', suf=suf + '_sinclim')),
                          arm(C.format(root='results/regen2024/clim_v1', suf=suf + '_sinclim'))),
        ('combinacion', hib('analysis/clim_v1_hibrido.json', lect), hib('analysis/clim_v1_hibrido_regen2024.json', lect)),
    ):
        res[lect][nombre] = {'antes': antes, 'despues': despues}
json.dump(res, open('results/regen2024/resumen_antes_despues.json', 'w'), indent=2)
f = lambda x, k: f"{x[k]['media']:.3f}" if x else '  -  '
for lect in ('mejor_epoca', 'ultima_epoca'):
    print(f'  [{lect}]                R2 antes  despues   SPAEF antes  despues')
    for nombre, v in res[lect].items():
        print(f"    {nombre:16s}        {f(v['antes'], 'R2')}    {f(v['despues'], 'R2')}      "
              f"{f(v['antes'], 'SPAEF')}    {f(v['despues'], 'SPAEF')}")
EOF

# --- 8. ficheros para el repositorio y empaquetado ---------------------------------------
P="$OUT/para_el_repo"
mkdir -p "$P/analysis" "$P/data" "$P/dataset_v4_ms_sx200"
for f in regen_mapas_2024.json sat_v2_canales_regen2024.json clim_v1_canales_regen2024.json \
         clim_v1_canales.json clim_v1_hibrido_regen2024.json manifest_dataset_v4_ms_sx200_regen2024.csv; do
  [ -f "analysis/$f" ] && cp "analysis/$f" "$P/analysis/" || echo "  [aviso] no esta analysis/$f"
done
cp data/tile_georef.csv data/tile_georef_v1.csv "$P/data/"
cp "$SRC/dataset_v4_ms_sx200.csv" "$SRC/dataset_v4_ms_sx200_v1.csv" "$P/dataset_v4_ms_sx200/"
echo "fecha_fin: $(date -u '+%F %T') UTC" >> "$OUT/entorno.txt"
tar -czf "$WORK/resultados_regen2024.tar.gz" -C "$REPO/results" regen2024
log "Resultados en $WORK/resultados_regen2024.tar.gz ($(du -h "$WORK/resultados_regen2024.tar.gz" | cut -f1))"
