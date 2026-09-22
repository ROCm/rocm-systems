#!/bin/sh
BIN_DIR=`dirname $0`
BIN_DIR=`realpath $BIN_DIR`
PKG_DIR=${BIN_DIR%/bin}

if [ -z "$1" ] ; then
  echo "Usage: $0 <cmd line>"
  exit 1
fi

# profiler plugin library
test_app=$*

# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so.1

# tool library loaded by ROC profiler
if [ -z "$ROCP_TOOL_LIB" ] ; then
  echo "ROCP_TOOL_LIB is not found"
  exit 1
fi

# ROC profiler metrics config file
if [ -z "$ROCP_METRICS" ] ; then
  echo "ROCP_METRICS is not found"
  exit 1
fi

# enable error messages
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export ROCPROFILER_LOG=1

$test_app
