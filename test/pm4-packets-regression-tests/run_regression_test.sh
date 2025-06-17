#!/bin/sh -x

testcase=$1

baseline_dir=$(dirname $0)/baselines
output_dir=$(dirname $0)/output
gfx_version=gfx1101
app=$(dirname $0)/builder_dump_exe

case "$testcase" in
    pmc)
        $app pmc $output_dir/pmc_builder_dump ${gfx_version} > $output_dir/pmc_builder_dump/debug_trace.txt 2>&1
        python3 $(dirname $0)/compare_debug_trace.py $baseline_dir/pmc_builder_dump/debug_trace.txt $output_dir/pmc_builder_dump/debug_trace.txt --html pmc_diff_gfx1101.html
        ;;
    sqtt)
        $app sqtt $output_dir/sqtt_builder_dump ${gfx_version} > $output_dir/sqtt_builder_dump/debug_trace.txt 2>&1
        python3 $(dirname $0)/compare_debug_trace.py $baseline_dir/sqtt_builder_dump/debug_trace.txt $output_dir/sqtt_builder_dump/debug_trace.txt --html sqtt_diff_gfx1101.html
        ;;
    spm)
        $app spm $output_dir/spm_builder_dump ${gfx_version} > $output_dir/spm_builder_dump/debug_trace.txt 2>&1
        python3 $(dirname $0)/compare_debug_trace.py $baseline_dir/spm_builder_dump/debug_trace.txt $output_dir/spm_builder_dump/debug_trace.txt --html spm_diff_gfx1101.html
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