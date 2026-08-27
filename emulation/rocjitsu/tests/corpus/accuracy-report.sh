#!/usr/bin/env bash
# Run the reference corpus under the emulator, score it, and write the report.
#
# One command, so that the accuracy figure quoted anywhere is one someone else
# can reproduce. Everything it needs is an argument: which rocjitsu, which
# config, which recorded hardware run to score against.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rocjitsu_root=$(cd "$here/../.." && pwd)

config="$rocjitsu_root/configs/gfx950_mi355x_kmd.json"
reference="$rocjitsu_root/Testing/results/meter-real-gfx950.json"
outdir="$rocjitsu_root/Testing/reports"
label="des"
jobs=42
shards=""
tolerance=20

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}"
    cat <<USAGE

  --config PATH      architecture config, timing block included
  --reference PATH   rocm-meter report recorded on hardware
  --out DIR          where the report is written (default: Testing/reports)
  --label NAME       names this run in the report (default: des)
  --shards DIR       reuse shard reports already in DIR instead of running
  --jobs N           parallel emulator processes (default: 42)
  --tolerance PCT    the band the report scores against (default: 20)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --reference) reference=$2; shift 2 ;;
        --out) outdir=$2; shift 2 ;;
        --label) label=$2; shift 2 ;;
        --shards) shards=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        --tolerance) tolerance=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

mkdir -p "$outdir"
if [[ -z $shards ]]; then
    shards="$outdir/shards-$label"
    rm -rf "$shards"
    echo "running the corpus under the emulator ($jobs shards)..."
    "$here/run-meter-emulated.sh" --out "$shards" --config "$config" --jobs "$jobs"
fi

scored="$outdir/scored-$label.json"
python3 "$here/meter_score.py" --real "$reference" --emulated "$shards"/*.json \
    --tolerance-pct "$tolerance" --json "$scored" --worst 20 | tee "$outdir/score-$label.txt"

python3 "$here/meter_report.py" --scored "$label=$scored" --out "$outdir" --format both \
    --title "rocjitsu timing accuracy"

echo
echo "report written to $outdir"
