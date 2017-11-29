#!/bin/sh

test_bin_dflt=./test/ctrl

#export HSA_LIB=/home/evgeny/pkg/compute-psdb-16453/lib
export HSA_LIB=/home/evgeny/git/compute/out/ubuntu-16.04/16.04/lib
export OCL_LIB=/home/evgeny/pkg/opencl_modified/opencl_x86_64/lib
#export OCL_LIB=/home/evgeny/Perforce/eshcherb_opencl/drivers/opencl/dist/linux/debug/lib/x86_64

# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$PWD:$HSA_LIB:$OCL_LIB
# enable error messages logging to '/tmp/rocprofiler_log.txt'
export ROCPROFILER_LOG=1

# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so
# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=test/libtool.so
# enable HSA dispatch intercepting by ROC profiler
export ROCP_HSA_INTERCEPT=1
# ROC profiler metrics config file
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
export ROCP_METRICS=metrics.xml
# input file for the tool library
export ROCP_INPUT=input.xml
# output directory for the tool library, for metrics results file 'results.txt'
# and SQTT trace files 'thread_trace.se<n>.out'
#export ROCP_OUTPUT_DIR=./

if [ -n "$1" ] ; then
  tbin="$*"
else
  tbin=$test_bin_dflt
fi
echo "Run $tbin"
eval $tbin

exit 0
