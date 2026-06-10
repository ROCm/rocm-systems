#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_waitcheck_gfx950_tensile_e2e.sh [options]

Builds representative gfx950 TensileLite kernels and runs rj_waitcheck over the
final loadable HSACO sidecars produced by Tensile.

Options:
  --waitcheck EXE       rj_waitcheck executable. Default: RJ_WAITCHECK or PATH.
  --work-dir DIR        Output/work directory. Default: WAITCHECK_TENSILE_WORKDIR.
  --tensilelite-root DIR
                        ROCm rocm-libraries/projects/hipblaslt/tensilelite.
                        Default: TENSILELITE_ROOT.
  --rocm-venv DIR       TheRock venv with bin/rocm-sdk. Default: ROCM_VENV.
  --rocm-path DIR       ROCm SDK root. Default: ROCM_PATH, rocm-sdk, or --rocm-venv.
  --python EXE          Python with Tensile deps. Default: PYTHON or python.
  --config FILE         Tensile config, absolute or relative to TensileLite root.
                        May be passed more than once.
  --skip-generate       Scan existing work-dir outputs without rebuilding Tensile.
  --scan-library-co     Also scan TensileLibrary_gfx950.co containers.
  -h, --help            Show this help.

Environment:
  TENSILELITE_ROOT, ROCM_VENV, ROCM_PATH, PYTHON, RJ_WAITCHECK,
  WAITCHECK_TENSILE_WORKDIR, WAITCHECK_TENSILE_CONFIGS

WAITCHECK_TENSILE_CONFIGS is colon-separated and uses the same paths as
--config. The default configs build one GEMM case and one sparse GEMM case.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

abs_dir() {
  local path=$1
  mkdir -p "$path"
  cd "$path" && pwd
}

abs_existing_dir() {
  local path=$1
  [[ -d "$path" ]] || die "not a directory: $path"
  cd "$path" && pwd
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "$script_dir/.." && pwd)"

rj_waitcheck=${RJ_WAITCHECK:-rj_waitcheck}
work_dir=${WAITCHECK_TENSILE_WORKDIR:-"$source_root/build/waitcheck-e2e/tensile-gfx950"}
tensilelite_root=${TENSILELITE_ROOT:-}
rocm_venv=${ROCM_VENV:-}
rocm_path=${ROCM_PATH:-}
python_exe=${PYTHON:-python}
gpu_target=gfx950
skip_generate=0
scan_library_co=0
configs=()

if [[ -n "${WAITCHECK_TENSILE_CONFIGS:-}" ]]; then
  IFS=: read -r -a configs <<<"${WAITCHECK_TENSILE_CONFIGS}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --waitcheck)
      [[ $# -ge 2 ]] || die "--waitcheck requires a value"
      rj_waitcheck=$2
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a value"
      work_dir=$2
      shift 2
      ;;
    --tensilelite-root)
      [[ $# -ge 2 ]] || die "--tensilelite-root requires a value"
      tensilelite_root=$2
      shift 2
      ;;
    --rocm-venv)
      [[ $# -ge 2 ]] || die "--rocm-venv requires a value"
      rocm_venv=$2
      shift 2
      ;;
    --rocm-path)
      [[ $# -ge 2 ]] || die "--rocm-path requires a value"
      rocm_path=$2
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      python_exe=$2
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || die "--config requires a value"
      configs+=("$2")
      shift 2
      ;;
    --skip-generate)
      skip_generate=1
      shift
      ;;
    --scan-library-co)
      scan_library_co=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ ${#configs[@]} -eq 0 ]]; then
  configs=(
    "Tensile/Tests/common/gemm/gfx950/bf16_cvt.yaml"
    "Tensile/Tests/common/sparse/gfx950/spmm_bf16_sb.yaml"
  )
fi

if [[ -z "$tensilelite_root" ]]; then
  die "set TENSILELITE_ROOT or pass --tensilelite-root"
fi
tensilelite_root="$(abs_existing_dir "$tensilelite_root")"
[[ -d "$tensilelite_root/Tensile" ]] || die "not a TensileLite root: $tensilelite_root"

if [[ -n "$rocm_venv" ]]; then
  rocm_venv="$(abs_existing_dir "$rocm_venv")"
fi
if [[ -z "$rocm_path" && -n "$rocm_venv" && -x "$rocm_venv/bin/rocm-sdk" ]]; then
  rocm_path=$("$rocm_venv/bin/rocm-sdk" path --root)
fi
if [[ -z "$rocm_path" ]] && command -v rocm-sdk >/dev/null 2>&1; then
  rocm_path=$(rocm-sdk path --root)
fi
[[ -n "$rocm_path" ]] || die "set ROCM_VENV, ROCM_PATH, or provide rocm-sdk in PATH"
rocm_path="$(abs_existing_dir "$rocm_path")"
[[ -x "$rocm_path/bin/amdclang++" ]] || die "missing amdclang++ under ROCm path: $rocm_path"
[[ -x "$rocm_path/lib/llvm/bin/clang-offload-bundler" ]] ||
  die "missing clang-offload-bundler under ROCm path: $rocm_path"

if [[ -n "$rocm_venv" && -x "$rocm_venv/bin/python" && "$python_exe" == "python" ]]; then
  python_exe="$rocm_venv/bin/python"
fi
if [[ "$python_exe" == */* ]]; then
  [[ -x "$python_exe" ]] || die "python is not executable: $python_exe"
  python_path="$(cd "$(dirname "$python_exe")" && pwd)"
else
  command -v "$python_exe" >/dev/null 2>&1 || die "python not found: $python_exe"
  python_path="$(dirname "$(command -v "$python_exe")")"
fi

if [[ "$rj_waitcheck" == */* ]]; then
  [[ -x "$rj_waitcheck" ]] || die "rj_waitcheck is not executable: $rj_waitcheck"
else
  command -v "$rj_waitcheck" >/dev/null 2>&1 || die "rj_waitcheck not found: $rj_waitcheck"
fi

work_dir="$(abs_dir "$work_dir")"
mkdir -p "$work_dir/logs"

export ROCM_PATH="$rocm_path"
export HIP_PATH="$rocm_path"
export TENSILE_ROCM_PATH="$rocm_path"
export PATH="$python_path:$rocm_path/bin:$rocm_path/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$rocm_path/lib:$rocm_path/lib64:$rocm_path/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$tensilelite_root:${PYTHONPATH:-}"
export CC="$rocm_path/bin/amdclang"
export CXX="$rocm_path/bin/amdclang++"

unset LD_PRELOAD
unset RJ_CONFIG
unset HSA_MODEL_LIB
unset HSA_MODEL_TOPOLOGY
unset HSA_OVERRIDE_GFX_VERSION

echo "tensilelite: $tensilelite_root"
echo "rocm:        $rocm_path"
echo "python:      $python_exe"
echo "waitcheck:   $rj_waitcheck"
echo "work-dir:    $work_dir"

for config in "${configs[@]}"; do
  [[ -n "$config" ]] || continue
  if [[ "$config" == /* ]]; then
    config_path=$config
    label=${config#"$tensilelite_root/"}
  else
    config_path="$tensilelite_root/$config"
    label=$config
  fi
  [[ -r "$config_path" ]] || die "cannot read Tensile config: $config"

  case_name=$(printf '%s' "$label" |
    sed 's#^Tensile/Tests/common/##; s#\.yaml$##; s#[^A-Za-z0-9._-]#_#g')
  case_work="$work_dir/work/$case_name"
  log_file="$work_dir/logs/$case_name.tensile.log"

  echo
  echo "==> $label"
  if (( skip_generate == 0 )); then
    rm -rf "$case_work"
    mkdir -p "$case_work"
    if ! "$python_exe" "$tensilelite_root/Tensile/bin/Tensile" \
      --build-only \
      --gpu-targets "$gpu_target" \
      "$config_path" \
      "$case_work" >"$log_file" 2>&1; then
      tail -n 120 "$log_file" >&2 || true
      die "Tensile build failed for $label; see $log_file"
    fi
    echo "built:       $case_work"
    echo "build-log:   $log_file"
  else
    [[ -d "$case_work" ]] || die "missing case work directory for --skip-generate: $case_work"
  fi

  mapfile -d '' objects < <(
    find "$case_work" -type f -name "Kernels.so-*-${gpu_target}*.hsaco" -print0 | sort -z
  )
  if (( scan_library_co != 0 )); then
    mapfile -d '' library_cos < <(
      find "$case_work" -type f -name "TensileLibrary_${gpu_target}.co" -print0 | sort -z
    )
    objects+=("${library_cos[@]}")
  fi
  [[ ${#objects[@]} -gt 0 ]] || die "no final HSACO sidecars found under $case_work"

  for object in "${objects[@]}"; do
    "$rj_waitcheck" --target "$gpu_target" "$object"
  done
done

echo
echo "gfx950 Tensile waitcheck e2e passed"
