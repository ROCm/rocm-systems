#!/bin/sh

usage() {
    echo "Usage: $0 <testcase> <gfx_version>"
    echo "  <testcase> must be one of: pmc, sqtt, spm"
    echo "  <gfx_version> is required (e.g., gfx1101)"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

testcase=$1
gfx_version=$2

# Check that both arguments are provided
if [ -z "$testcase" ] || [ -z "$gfx_version" ]; then
    echo "ERROR: Incorrect usage. Missing required arguments."
    usage
fi

baseline_dir=$SCRIPT_DIR/baselines
output_dir=$SCRIPT_DIR/output
app=$SCRIPT_DIR/builder_dump

case "$testcase" in
    pmc)
        mkdir -p "$output_dir/pmc_builder_dump"
        $app pmc $output_dir/pmc_builder_dump ${gfx_version} > $output_dir/pmc_builder_dump/debug_trace.txt 2>&1
        python3 $SCRIPT_DIR/compare_debug_trace.py $baseline_dir/pmc_builder_dump/debug_trace.txt $output_dir/pmc_builder_dump/debug_trace.txt --html $SCRIPT_DIR/pmc_diff_${gfx_version}.html
        ;;
    sqtt)
        mkdir -p "$output_dir/sqtt_builder_dump"
        $app sqtt $output_dir/sqtt_builder_dump ${gfx_version} > $output_dir/sqtt_builder_dump/debug_trace.txt 2>&1
        python3 $SCRIPT_DIR/compare_debug_trace.py $baseline_dir/sqtt_builder_dump/debug_trace.txt $output_dir/sqtt_builder_dump/debug_trace.txt --html $SCRIPT_DIR/sqtt_diff_${gfx_version}.html
        ;;
    spm)
        mkdir -p "$output_dir/spm_builder_dump"
        $app spm $output_dir/spm_builder_dump ${gfx_version} > $output_dir/spm_builder_dump/debug_trace.txt 2>&1
        python3 $SCRIPT_DIR/compare_debug_trace.py $baseline_dir/spm_builder_dump/debug_trace.txt $output_dir/spm_builder_dump/debug_trace.txt --html $SCRIPT_DIR/spm_diff_${gfx_version}.html
        ;;
    *)
        echo "Unknown test case: $testcase"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "$testcase regression : FAILED"
    exit 1
fi

echo "$testcase regression : PASSED"