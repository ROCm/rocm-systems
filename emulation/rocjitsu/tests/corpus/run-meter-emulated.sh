#!/usr/bin/env bash
# Run the rocm-meter corpus under rocjitsu, sharded across processes.
#
# One emulator process per kernel family.  The corpus is embarrassingly
# parallel across families and a single serial pass costs tens of minutes,
# which is too slow to iterate a timing model against.  Each shard writes its
# own report; meter_score.py merges them and drops duplicates, so shards whose
# --kernel substrings overlap are harmless.
#
# conv2d NCHW is deliberately not in the default shard list: it reaches MIOpen,
# which does not terminate under emulation within any useful budget.  Pass
# --with-nchw to include it behind the per-shard timeout anyway.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rocjitsu_root=$(cd "$here/../.." && pwd)

config="$rocjitsu_root/configs/gfx950_mi355x_kmd.json"
tuning="$rocjitsu_root/Testing/tuning/gfx950-calibrated.json"
binary="$rocjitsu_root/build/tools/rocjitsu/rocjitsu"
python_bin="${METER_PYTHON:-$HOME/venv-torch7/bin/python}"
outdir=""
tier="standard"
samples=3
warmups=3
timeout_s=1800
jobs=24
with_nchw=0
extra=()

usage() {
    cat <<USAGE
usage: $(basename "$0") --out DIR [options]

  --out DIR          where shard reports are written (required)
  --config PATH      rocjitsu architecture config (default: $config)
  --tuning PATH      calibrated timing overlay to merge in, or "" for none
                     (default: $tuning, skipped when absent)
  --binary PATH      rocjitsu CLI (default: $binary)
  --python PATH      interpreter with a ROCm torch (default: $python_bin)
  --tier TIER        quick|standard|thorough (default: $tier)
  --samples N        timed samples per case (default: $samples)
  --warmups N        warmup calls per case (default: $warmups)
  --timeout S        per-shard wall-clock limit (default: $timeout_s)
  --jobs N           shards to run at once (default: $jobs)
  --with-nchw        include the conv2d NCHW shard
  --                 everything after is passed to rocm-meter.py
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) outdir=$2; shift 2 ;;
        --config) config=$2; shift 2 ;;
        --tuning) tuning=$2; shift 2 ;;
        --binary) binary=$2; shift 2 ;;
        --python) python_bin=$2; shift 2 ;;
        --tier) tier=$2; shift 2 ;;
        --samples) samples=$2; shift 2 ;;
        --warmups) warmups=$2; shift 2 ;;
        --timeout) timeout_s=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        --with-nchw) with_nchw=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n $outdir ]] || { echo "--out is required" >&2; exit 2; }
[[ -x $binary ]] || { echo "no rocjitsu binary at $binary" >&2; exit 2; }
[[ -r $config ]] || { echo "no config at $config" >&2; exit 2; }
[[ -x $python_bin ]] || { echo "no interpreter at $python_bin" >&2; exit 2; }

# The checked-in config carries the machine's shape and no measured value.
# Without the overlay every calibrated parameter falls back to its slow default
# and the run is correct but uncalibrated, which is worth saying out loud rather
# than leaving to be discovered in the scores.
if [[ -n $tuning && -r $tuning ]]; then
    merged=$(mktemp -t rocjitsu-config-XXXXXX.json)
    trap 'rm -f "$merged"' EXIT
    "$python_bin" "$here/merge-tuning.py" --config "$config" --tuning "$tuning" \
        --out "$merged" || exit 2
    config=$merged
else
    echo "note: no timing tuning overlay at ${tuning:-<none>}; running uncalibrated" >&2
fi

mkdir -p "$outdir"

# One entry per shard: "<name>:<filter> [<filter> ...]".  Split by kernel
# family; the big families are split again by dtype so no single shard
# dominates the wall clock.
shards=(
    "launch_f32:--kernel launch --dtype float32"
    "launch_f16:--kernel launch --dtype float16"
    "launch_bf16:--kernel launch --dtype bfloat16"
    "triad_f32:--kernel triad --dtype float32"
    "triad_f16:--kernel triad --dtype float16"
    "triad_bf16:--kernel triad --dtype bfloat16"
    "copy_f32:--kernel copy --dtype float32"
    "copy_f16:--kernel copy --dtype float16"
    "copy_bf16:--kernel copy --dtype bfloat16"
    "atan_f32:--kernel atan --dtype float32"
    "atan_f16:--kernel atan --dtype float16"
    "atan_bf16:--kernel atan --dtype bfloat16"
    "fused_elementwise_f32:--kernel fused_elementwise --dtype float32"
    "fused_elementwise_f16:--kernel fused_elementwise --dtype float16"
    "fused_elementwise_bf16:--kernel fused_elementwise --dtype bfloat16"
    "gemm_f32:--kernel gemm.eager --kernel gemm.compiled --dtype float32"
    "gemm_f16:--kernel gemm.eager --kernel gemm.compiled --dtype float16"
    "gemm_bf16:--kernel gemm.eager --kernel gemm.compiled --dtype bfloat16"
    "concurrent:--kernel concurrent_gemm"
    "quantized:--kernel scaled_int8_mm --kernel scaled_fp8_mm"
    "attention_qk_f32:--kernel attention_qk --dtype float32"
    "attention_qk_f16:--kernel attention_qk --dtype float16"
    "attention_qk_bf16:--kernel attention_qk --dtype bfloat16"
    "attention_pv_f32:--kernel attention_pv --dtype float32"
    "attention_pv_f16:--kernel attention_pv --dtype float16"
    "attention_pv_bf16:--kernel attention_pv --dtype bfloat16"
    "attention_softmax_f32:--kernel attention_softmax --dtype float32"
    "attention_softmax_f16:--kernel attention_softmax --dtype float16"
    "attention_softmax_bf16:--kernel attention_softmax --dtype bfloat16"
    "rms_norm_f32:--kernel rms_norm --dtype float32"
    "rms_norm_f16:--kernel rms_norm --dtype float16"
    "rms_norm_bf16:--kernel rms_norm --dtype bfloat16"
    "fused_add_rms_norm_f32:--kernel fused_add_rms_norm --dtype float32"
    "fused_add_rms_norm_f16:--kernel fused_add_rms_norm --dtype float16"
    "fused_add_rms_norm_bf16:--kernel fused_add_rms_norm --dtype bfloat16"
    "rope_f32:--kernel rope --dtype float32"
    "rope_f16:--kernel rope --dtype float16"
    "rope_bf16:--kernel rope --dtype bfloat16"
    "swiglu:--kernel swiglu"
    "fused_qk:--kernel fused_qk_norm_rope"
    "conv2d_nhwc:--kernel nhwc"
    "conv2d_depthwise:--kernel depthwise"
)
if (( with_nchw )); then
    shards+=("conv2d_nchw:--kernel conv2d.eager.float32.nchw")
fi

run_shard() {
    local spec=$1
    local name=${spec%%:*}
    local filters=${spec#*:}
    local out="$outdir/$name.json"
    local log="$outdir/$name.log"
    local start
    start=$SECONDS
    # shellcheck disable=SC2086
    # Both are diagnostics and both are off unless the caller asked: the
    # per-dispatch trace is what lets a tuning change be evaluated against a
    # recorded corpus instead of by re-running it, and it is joined to cases by
    # the device-clock window the meter records.
    local trace_env=()
    if [[ -n ${METER_TRACE_DIR:-} ]]; then
        mkdir -p "$METER_TRACE_DIR"
        trace_env=(env "ROCJITSU_TIMING_TRACE=$METER_TRACE_DIR/$name.jsonl"
                   "ROCM_METER_DEVICE_WINDOW=1")
    fi
    timeout --kill-after=30 "$timeout_s" \
        "${trace_env[@]}" "$binary" --config "$config" -- \
        "$python_bin" "$here/rocm-meter.py" \
        --tier "$tier" --samples "$samples" --warmups "$warmups" \
        $filters --output "$out" "${extra[@]+"${extra[@]}"}" \
        >"$log" 2>&1
    local status=$?
    printf '%-24s %4ds  exit=%d\n' "$name" "$((SECONDS - start))" "$status"
    return 0
}
export -f run_shard
export outdir timeout_s binary config python_bin here tier samples warmups
export METER_TRACE_DIR
export extra

printf '%s\n' "${shards[@]}" \
    | xargs -P "$jobs" -I{} bash -c 'run_shard "$@"' _ {}

echo
echo "shard reports in $outdir"
ls "$outdir"/*.json 2>/dev/null | wc -l | xargs echo "reports:"
