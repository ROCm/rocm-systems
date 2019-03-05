#!/bin/sh
BIN_DIR=`dirname $0`
BIN_DIR=`realpath $BIN_DIR`
PKG_DIR=${BIN_DIR%/bin}

# PATH to custom HSA libs
HSA_PATH=$PKG_DIR/lib/hsa

if [ -z "$1" ] ; then
  echo "Usage: $0 <cmd line>"
  exit 1
fi

# profiler plugin library
test_app=$*

# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PKG_DIR/lib:$PKG_DIR/tool:$HSA_PATH
export PATH=.:$PATH

# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so.1
# tool library loaded by ROC profiler
if [ -z "$ROCP_TOOL_LIB" ] ; then
  export ROCP_TOOL_LIB=libintercept_test.so
fi
# enable error messages
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export ROCPROFILER_LOG=1
# ROC profiler metrics config file
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
if [ -z "$ROCP_METRICS" ] ; then
  export ROCP_METRICS=$PKG_DIR/lib/metrics.xml
fi

LD_PRELOAD=$ROCP_TOOL_LIB $test_app
