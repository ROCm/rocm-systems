#!/bin/sh

tbin=./test/ctrl

#export HSA_LIB=/home/evgeny/pkg/compute-psdb-16453/lib
export HSA_LIB=/home/evgeny/git/compute/out/ubuntu-16.04/16.04/lib
export OCL_LIB=/home/evgeny/pkg/opencl_modified/opencl_x86_64/lib
#export OCL_LIB=/home/evgeny/Perforce/eshcherb_opencl/drivers/opencl/dist/linux/debug/lib/x86_64

export LD_LIBRARY_PATH=$PWD:$HSA_LIB:$OCL_LIB
export ROCPROFILER_LOG=1

export HSA_TOOLS_LIB=librocprofiler64.so
export ROCP_TOOL_LIB=test/libtool.so
export ROCP_HSA_INTERCEPT=1
export ROCP_METRICS=metrics.xml
export ROCP_INPUT=input.xml
unset ROCP_PROXY_QUEUE

echo "Run simple profiling test"
if [ -n "$1" ] ; then
  eval "$*"
else
  eval $tbin
fi

exit 0
