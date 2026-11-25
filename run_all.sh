#!/usr/bin/env bash
# =============================================================================
# run_all.sh — run the full thesis pipeline in an isolated run directory.
#
# Every run works on a fresh copy of the code and input datasets under
# runs/<timestamp>/, so the committed results/ folders are never overwritten
# and two runs can be compared side by side. Run ./run_all.sh --help for usage.
# =============================================================================
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

ALL_STEPS=(preprocess layer2 features latent layer1 baselines plots)
VALID_PRIORS="lasso gaussian horseshoe spike_and_slab"
VALID_HIERARCHIES="DENDRO TREE H2 H3"

MODE="shipped"
PRIORS="gaussian lasso horseshoe spike_and_slab"
HIER_PRIORS="horseshoe lasso"
HIERARCHY="DENDRO"
RUN_DIR=""
DRY_RUN=0
QUIET=0

# ----------------------------------------------------------------------------- output
if [[ -t 1 ]]; then B=$'\e[1m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; N=$'\e[0m'; else B=; G=; R=; Y=; N=; fi
info() { printf '%s\n' "${B}==>${N} $*"; }
warn() { printf '%s\n' "${Y}warning:${N} $*" >&2; }
die()  { printf '%s\n' "${R}error:${N} $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./run_all.sh [options] [step ...]

Runs the pipeline steps in order (all of them if none are given):

  preprocess  one-hot -> Yeo-Johnson -> standardization -> modelling datasets   seconds
  layer2      extant-fossil comparability: GMM cross-likelihood, EMD, tests    seconds
  features    horseshoe-based Bayesian feature selection (PyMC)                 minutes
  latent      variational Bayesian GPLVM, then BHC vs Dirichlet Process         minutes
  layer1      Bayesian logistic regression (one run per prior) and the
              hierarchical cascade, all with exact leave-one-out + NUTS         hours
  baselines   classical models (RF, SVM, LR) and classical cascades             minutes
  plots       PCA projection of the extant species                              seconds

Options:
  --shipped             Each step reads the exact datasets committed in its src/
                        folder, i.e. the inputs used for the thesis (default).
  --chain               End-to-end rerun: every intermediate dataset is rebuilt
                        from the raw Excel and fed to the next step. Missing
                        upstream steps are added automatically.
  --priors "P ..."      Priors for the flat model (default: "$PRIORS").
                        Valid: $VALID_PRIORS  (lasso = Laplace)
  --hier-priors "P ..." Priors for the hierarchical cascade (default: "$HIER_PRIORS").
  --hierarchy H         Cascade structure: $VALID_HIERARCHIES (default: $HIERARCHY,
                        the data-derived tree used as reference in the thesis).
  --out DIR             Run directory (default: runs/<timestamp>). Pointing it to an
                        existing run resumes it: layer1 continues from its
                        per-fold LOO checkpoints.
  --python PATH         Python interpreter (default: \$PYTHON or python3).
  --quiet               Write script output only to the log files.
  --dry-run             Show what would run, without running it.
  -h, --help            Show this help.

Examples:
  ./run_all.sh                          # everything, with the thesis inputs
  ./run_all.sh layer2 latent            # just Layers 2 and 3 (a few minutes)
  ./run_all.sh --priors horseshoe layer1
  ./run_all.sh --chain                  # rebuild everything from the raw data
  ./run_all.sh --out runs/20260601-1200 layer1   # resume an interrupted run

Outputs: runs/<timestamp>/<stage>/ (same layout as the repository), one log per
step in runs/<timestamp>/logs/, and run_info.txt with the commit, the options
and the exact package versions.
EOF
}

# ----------------------------------------------------------------------------- arguments
STEPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shipped)      MODE="shipped" ;;
    --chain)        MODE="chain" ;;
    --priors)       [[ $# -ge 2 ]] || die "--priors needs a value"; PRIORS="$2"; shift ;;
    --hier-priors)  [[ $# -ge 2 ]] || die "--hier-priors needs a value"; HIER_PRIORS="$2"; shift ;;
    --hierarchy)    [[ $# -ge 2 ]] || die "--hierarchy needs a value"; HIERARCHY="$2"; shift ;;
    --out)          [[ $# -ge 2 ]] || die "--out needs a value"; RUN_DIR="$2"; shift ;;
    --python)       [[ $# -ge 2 ]] || die "--python needs a value"; PY="$2"; shift ;;
    --quiet)        QUIET=1 ;;
    --dry-run)      DRY_RUN=1 ;;
    -h|--help)      usage; exit 0 ;;
    -*)             die "unknown option '$1' (see --help)" ;;
    *)              [[ " ${ALL_STEPS[*]} " == *" $1 "* ]] || die "unknown step '$1'. Steps: ${ALL_STEPS[*]}"
                    STEPS+=("$1") ;;
  esac
  shift
done

for p in $PRIORS $HIER_PRIORS; do
  [[ " $VALID_PRIORS " == *" $p "* ]] || die "unknown prior '$p'. Valid: $VALID_PRIORS"
done
[[ " $VALID_HIERARCHIES " == *" $HIERARCHY "* ]] || die "unknown hierarchy '$HIERARCHY'. Valid: $VALID_HIERARCHIES"
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=("${ALL_STEPS[@]}")

# ----------------------------------------------------------------------------- run directory
[[ -n "$RUN_DIR" ]] || RUN_DIR="$REPO/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$RUN_DIR")"
RUN_DIR="$(cd "$(dirname "$RUN_DIR")" && pwd)/$(basename "$RUN_DIR")"
MARKER="$RUN_DIR/.run_all"
STAMP="$RUN_DIR/.step_started"
LOGS="$RUN_DIR/logs"

step_done() { [[ -f "$MARKER" ]] && grep -qx "done $1" "$MARKER"; }

# In chain mode a step must read what its upstream steps produced in this run.
declare -A DEPS=([layer2]="preprocess" [features]="preprocess" [latent]="preprocess"
                 [layer1]="features latent" [baselines]="latent" [plots]="preprocess")
if [[ "$MODE" == "chain" ]]; then
  needed=" ${STEPS[*]} "
  changed=1
  while (( changed )); do
    changed=0
    for s in $needed; do
      for d in ${DEPS[$s]:-}; do
        if [[ "$needed" != *" $d "* ]] && ! step_done "$d"; then needed+="$d "; changed=1; fi
      done
    done
  done
  STEPS=()
  for s in "${ALL_STEPS[@]}"; do [[ "$needed" == *" $s "* ]] && STEPS+=("$s"); done
else
  ordered=()
  for s in "${ALL_STEPS[@]}"; do [[ " ${STEPS[*]} " == *" $s "* ]] && ordered+=("$s"); done
  STEPS=("${ordered[@]}")
fi

# ----------------------------------------------------------------------------- Python checks
declare -A MODULES=(
  [preprocess]="pandas numpy scipy sklearn openpyxl"
  [layer2]="pandas numpy scipy sklearn matplotlib seaborn"
  [features]="pandas numpy sklearn pymc arviz"
  [latent]="pandas numpy scipy sklearn matplotlib seaborn torch gpytorch"
  [layer1]="pandas numpy scipy sklearn matplotlib seaborn pymc pytensor arviz tqdm"
  [baselines]="pandas numpy sklearn matplotlib seaborn networkx tqdm"
  [plots]="pandas numpy sklearn matplotlib seaborn"
)

check_python() {
  command -v "$PY" >/dev/null 2>&1 || die "Python not found: '$PY' (use --python or set PYTHON)"
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "Python 3.10+ required ($("$PY" --version 2>&1))"
  local mods="" s m missing=""
  for s in "${STEPS[@]}"; do mods+=" ${MODULES[$s]}"; done
  for m in $(tr ' ' '\n' <<<"$mods" | sort -u); do
    "$PY" -c "import $m" >/dev/null 2>&1 || missing+=" $m"
  done
  [[ -z "$missing" ]] || die "missing Python packages:$missing
       Install them with:  $PY -m pip install -r requirements.txt"
}

# ----------------------------------------------------------------------------- helpers
copy_to_run() {   # copy code + inputs (never results) from the repository into the run dir
  info "Preparing run directory $RUN_DIR"
  mkdir -p "$RUN_DIR" "$LOGS"
  (cd "$REPO" && find . \( -path ./runs -o -path ./.git -o -path ./.venv -o -path ./venv \) -prune \
      -o -type f \( -name '*.py' -o -path '*/src/*' \) -print) |
  while IFS= read -r f; do
    mkdir -p "$RUN_DIR/$(dirname "$f")"
    cp "$REPO/$f" "$RUN_DIR/$f"
  done
  cp "$REPO/transformations/one_hot/PALAEO3_microtexture2.xlsx" "$RUN_DIR/transformations/one_hot/"
  # Thesis latent space (d = 5, 75 inducing points): input of the structure discovery
  # and of every model that combines raw variables with latent dimensions.
  mkdir -p "$RUN_DIR/visualization/gplvm/results_ld5_75"
  cp "$REPO/visualization/gplvm/results_ld5_75/espacio_latente_gplvm.csv" "$RUN_DIR/visualization/gplvm/results_ld5_75/"
  printf 'mode %s\n' "$MODE" > "$MARKER"
}

write_run_info() {
  {
    echo "date:       $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "commit:     $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'n/a')$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
    echo "mode:       $MODE"
    echo "steps:      ${STEPS[*]}"
    echo "priors:     flat [$PRIORS] | hierarchical [$HIER_PRIORS] | hierarchy $HIERARCHY"
    echo "python:     $("$PY" --version 2>&1) ($(command -v "$PY"))"
    echo "host:       $(uname -srm)"
    echo; echo "packages:"
    "$PY" -m pip freeze 2>/dev/null | sed 's/^/  /' || echo "  (pip not available)"
  } >> "$RUN_DIR/run_info.txt"
}

run_py() {   # run_py <log> <dir> <script> [args ...]  — honours ENV_EXTRA
  local log="$1" dir="$2"; shift 2
  local cmd=(env MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 ${ENV_EXTRA:-} "$PY" "$@")
  printf '  %s$ %s\n' "$dir" "${ENV_EXTRA:+$ENV_EXTRA }$PY $*"
  (( DRY_RUN )) && return 0
  echo "### $(date '+%H:%M:%S')  $dir\$ ${ENV_EXTRA:+$ENV_EXTRA }$PY $*" >> "$log"
  if (( QUIET )); then
    (cd "$RUN_DIR/$dir" && "${cmd[@]}") >> "$log" 2>&1
  else
    (cd "$RUN_DIR/$dir" && "${cmd[@]}") 2>&1 | tee -a "$log"
  fi
}

expect() {   # expect <dir> <file> ...  — several scripts print errors instead of failing,
             # so check that each output exists and was written during this step
  (( DRY_RUN )) && return 0
  local dir="$1" f missing=(); shift
  for f in "$@"; do
    [[ -s "$RUN_DIR/$dir/$f" && "$RUN_DIR/$dir/$f" -nt "$STAMP" ]] || missing+=("$dir/$f")
  done
  (( ${#missing[@]} == 0 )) || die "expected output not produced: ${missing[*]}"
}

share() {   # share <source file> <destination dir> [new name]  — chain mode only
  [[ "$MODE" == "chain" ]] || return 0
  local src="$1" dst="$2" name="${3:-$(basename "$1")}"
  printf '  -> %s\n' "$dst/$name"
  (( DRY_RUN )) && return 0
  mkdir -p "$RUN_DIR/$dst"
  cp "$RUN_DIR/$src" "$RUN_DIR/$dst/$name"
}

# ----------------------------------------------------------------------------- steps
step_preprocess() {
  local log="$LOGS/preprocess.log"
  run_py "$log" transformations/one_hot one_hot.py
  expect transformations/one_hot dataset_onehot.csv
  if (( ! DRY_RUN )); then
    cp "$RUN_DIR/transformations/one_hot/dataset_onehot.csv" "$RUN_DIR/transformations/normalization/"
    cp "$RUN_DIR/transformations/one_hot/dataset_onehot.csv" "$RUN_DIR/transformations/src/"
  fi
  run_py "$log" transformations/normalization normalization.py
  expect transformations/normalization dataset_normalized.csv
  run_py "$log" transformations/standarization standarization.py
  expect transformations/src dataset_onehot_std.csv
  run_py "$log" transformations build_model_datasets.py
  expect transformations/src dataset_normalized_eda.csv dataset_nbool_std.csv dataset_con_grupos.csv

  share transformations/src/dataset_normalized_eda.csv eda/src dataset_normalized.csv
  local d
  for d in eda/src feature_engineering/src probabilistic_models/src ml_models/src; do
    share transformations/src/dataset_nbool_std.csv "$d"
  done
  for d in ml_models/src probabilistic_models/src visualization/gplvm/src visualization/pca/src; do
    share transformations/src/dataset_con_grupos.csv "$d"
  done
}

step_layer2() {
  local log="$LOGS/layer2.log" v
  for v in nbool:dataset_nbool_std.csv norm:dataset_normalized.csv; do
    run_py "$log" eda run_layer2.py --input "src/${v#*:}" --outdir "layer2/${v%%:*}"
    expect "eda/layer2/${v%%:*}" verosimilitud_gmm_1.csv verosimilitud_gmm_2.csv distancias_emd.csv
  done
}

step_features() {
  local log="$LOGS/features.log"
  run_py "$log" feature_engineering bayesian_feature_selection.py
  expect feature_engineering dataset_seleccionado_extant.csv seleccion_horseshoe_extant.csv
  share feature_engineering/dataset_seleccionado_extant.csv probabilistic_models/src
}

step_latent() {
  local log="$LOGS/latent.log"
  run_py "$log" visualization/gplvm gplvm.py
  expect visualization/gplvm espacio_latente_gplvm.csv
  share visualization/gplvm/espacio_latente_gplvm.csv visualization/gplvm/results_ld5_75
  share visualization/gplvm/espacio_latente_gplvm.csv probabilistic_models/src
  share visualization/gplvm/espacio_latente_gplvm.csv ml_models/src espacio_latente_gplvm_1.csv
  run_py "$log" visualization/gplvm latent_structure_discovery.py
  expect visualization/gplvm bhc_dendrogram_ld5.png clustering_comparison_ld5.png fossil_cluster_assignments_ld5.csv
}

step_layer1() {
  local log="$LOGS/layer1.log" p
  for p in $PRIORS; do
    ENV_EXTRA="TIPO_PRIOR=$p" run_py "$log" probabilistic_models blr.py
    expect probabilistic_models "predicciones_loocv_blr_$p.csv"
  done
  for p in $HIER_PRIORS; do
    ENV_EXTRA="TIPO_PRIOR=$p JERARQUIA=$HIERARCHY" run_py "$log" probabilistic_models blr_hierarchical.py
    expect probabilistic_models "predicciones_loocv_3niveles_$p.csv"
  done
}

step_baselines() {
  local log="$LOGS/baselines.log"
  run_py "$log" ml_models ml_models.py
  expect ml_models predicciones_directas_especies.csv
  run_py "$log" ml_models hierarchical_models.py
  expect ml_models predicciones_arbol_ml.csv
}

step_plots() {
  run_py "$LOGS/plots.log" visualization/pca pca.py
  expect visualization/pca pca_extant_elipses.png
}

# ----------------------------------------------------------------------------- main
check_python

if [[ -f "$MARKER" ]]; then
  prev_mode="$(sed -n 's/^mode //p' "$MARKER")"
  [[ "$prev_mode" == "$MODE" ]] || die "$RUN_DIR was a --$prev_mode run; resume it with --$prev_mode"
  info "Resuming $RUN_DIR"
elif (( DRY_RUN )); then
  info "Would prepare run directory $RUN_DIR"
elif [[ -e "$RUN_DIR" ]] && [[ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
  die "$RUN_DIR exists and is not a run_all.sh run directory"
else
  copy_to_run
fi
(( DRY_RUN )) || write_run_info

info "Mode: $MODE | steps: ${STEPS[*]}"
[[ " ${STEPS[*]} " == *" latent "* ]] && warn "the GPLVM is not seeded: a new latent space differs slightly from the thesis one.$([[ $MODE == shipped ]] && echo ' In --shipped mode the downstream steps keep using the thesis latent space.')"

declare -A STATUS=() TIME=()
current=""
on_exit() {
  local rc=$?
  if (( rc != 0 )) && [[ -n "$current" ]]; then
    printf '\n%s\n' "${R}Step '$current' failed.${N} Log: $LOGS/$current.log" >&2
    printf 'Fix the problem, then resume with:\n  ./run_all.sh --%s --out %q %s\n' "$MODE" "$RUN_DIR" "$current" >&2
  fi
}
trap on_exit EXIT

for s in "${STEPS[@]}"; do
  current="$s"
  info "[$s]"
  (( DRY_RUN )) || { touch "$STAMP"; sleep 1; }   # outputs must be newer than this
  t0=$SECONDS
  "step_$s"
  TIME[$s]=$(( SECONDS - t0 ))
  STATUS[$s]="ok"
  (( DRY_RUN )) || echo "done $s" >> "$MARKER"
done
current=""

if (( ! DRY_RUN )); then
  printf '\n%s\n' "${B}Summary${N}"
  for s in "${STEPS[@]}"; do
    printf '  %-11s %s%-4s%s %6ss\n' "$s" "$G" "${STATUS[$s]}" "$N" "${TIME[$s]}"
  done
  printf '\nResults: %s\nLogs:    %s\n' "$RUN_DIR" "$LOGS"
fi
