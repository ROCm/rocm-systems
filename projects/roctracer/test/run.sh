#!/bin/sh

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# cd to build directory
BIN_NAME=`basename $0`
BIN_DIR=`dirname $0`

#To enable symbol lookup in .dynsyn section after llvm-strip
export LOADER_USE_DYNSYM=1

cd $BIN_DIR

if [ -z "$ROCTRACER_LIB_PATH" ] ; then
  if test -f "${BIN_DIR}/../../lib/libroctracer64.so" ; then
    ROCTRACER_LIB_PATH="${BIN_DIR}/../../lib"
  fi
fi

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and other libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
if [ -n "$ROCTRACER_LIB_PATH" ] ; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCTRACER_LIB_PATH
fi
if [ -z "$ROCTRACER_LIB_PATH" ] ; then
  ROCTRACER_LIB_PATH="."
fi
if [ -z "$ROCTRACER_TOOL_PATH" ] ; then
  if test -f "${BIN_DIR}/../../lib/roctracer/libroctracer_tool.so" ; then
    ROCTRACER_TOOL_PATH="${BIN_DIR}/../../lib/roctracer"
  else
    ROCTRACER_TOOL_PATH="."
  fi
fi

# test filter input
test_filter=-1
check_trace_flag=1
if [ -n "$1" ] ; then
  test_filter=$1
  shift
fi
if [ "$2" = "-n" ] ; then
  check_trace_flag=0
fi

# test check routin
test_status=0
test_runnum=0
test_number=0
failed_tests="Failed tests:"

xeval_test() {
  test_number=$test_number
}

ncolors=$(tput colors || echo 0)
if [ -n "$ncolors" ] && [ $ncolors -ge 8 ]; then
  bright="$(tput bold     || echo)"
  red="$(tput setaf 1     || echo)"
  green="$(tput setaf 2   || echo)"
  blue="$(tput setaf 4    || echo)"
  normal="$(tput sgr0     || echo)"
fi

eval_test() {
  label=$1
  cmdline=$2
  test_name=$3

  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "test $test_number: $test_name \"$label\""
    echo "CMD: \"$cmdline\""
    mkdir -p /tmp/test/out
    test_runnum=$((test_runnum + 1))
    eval "$cmdline" 1>/tmp/test/out/$test_name.out 2>/tmp/test/out/$test_name.err
    is_failed=$?
    if [ $is_failed != 0 ] ; then
      echo "--- stdout ---"
      cat /tmp/test/out/$test_name.out
      echo "--- stderr ---"
      cat /tmp/test/out/$test_name.err
    fi
    if [ $is_failed = 0 ] ; then
      python3 ./test/check_trace.py -in $test_name -ck $check_trace_flag
      is_failed=$?
      if [ $is_failed != 0 ] ; then
        echo "Trace checker error:"
        python3 ./test/check_trace.py -v -in $test_name -ck $check_trace_flag
      fi
    fi
    if [ $is_failed = 0 ] ; then
      echo "${bright:-}${blue:-}$test_name: ${green:-}PASSED${normal:-}"
    else
      echo "${bright:-}${blue:-}$test_name: ${red:-}FAILED${normal:-}"
      failed_tests="$failed_tests\n  $test_number: $test_name - \"$label\""
      test_status=$(($test_status + 1))
    fi
  fi

  test_number=$((test_number + 1))
}

# Tests dry run
eval_test "MatrixTranspose dry run" ./test/MatrixTranspose MatrixTranspose_dryrun_trace
eval_test "copy dry run" ./test/copy copy_dryrun_trace

# Standalone test
# ROCtracer is used explicitely by test
eval_test "standalone C test" "./test/MatrixTranspose_ctest" MatrixTranspose_ctest_trace
eval_test "standalone HIP test" "./test/MatrixTranspose_test" MatrixTranspose_test_trace
eval_test "standalone HIP hipaact test" "./test/MatrixTranspose_hipaact_test" MatrixTranspose_hipaact_test_trace
eval_test "standalone HIP MGPU test" "./test/MatrixTranspose_mgpu" MatrixTranspose_mgpu_trace

# Tool test
# ROCtracer/tool is loaded by HSA runtime
export LD_PRELOAD="$ROCTRACER_TOOL_PATH/libroctracer_tool.so"

# ROCTX test
export ROCTRACER_DOMAIN="roctx"
eval_test "roctx test" ./test/roctx_test roctx_test_trace

# SYS test
export ROCTRACER_DOMAIN="sys:roctx"
eval_test "tool SYS test" ./test/MatrixTranspose MatrixTranspose_sys_trace
export ROCTRACER_DOMAIN="sys:hsa:roctx"
eval_test "tool SYS/HSA test" ./test/MatrixTranspose MatrixTranspose_sys_hsa_trace
# Tracing control <delay:length:rate>
export ROCTRACER_DOMAIN="hip"
eval_test "tool period test" "ROCP_CTRL_RATE=10:50000:500000 ./test/MatrixTranspose" MatrixTranspose_hip_period_trace
eval_test "tool flushing test" "ROCP_FLUSH_RATE=100000 ./test/MatrixTranspose" MatrixTranspose_hip_flush_trace

#API records filtering
echo "<trace name=\"HIP\"><parameters api=\"hipFree, hipMalloc, hipMemcpy\"></parameters></trace>" > /tmp/input.xml
export ROCP_INPUT=/tmp/input.xml
eval_test "tool HIP test input" ./test/MatrixTranspose MatrixTranspose_hip_input_trace
unset ROCP_INPUT

# HSA test
export ROCTRACER_DOMAIN="hsa"
# test trace
export ROC_TEST_TRACE=1
# kernels loading iterations
export ROCP_KITER=1
# kernels dispatching iterations per kernel load
# dispatching to the same queue
export ROCP_DITER=1
# GPU agents number
export ROCP_AGENTS=1
# host threads number
# each thread creates a queue pre GPU agent
export ROCP_THRS=1

eval_test "tool HSA test" ./test/copy copy_hsa_trace

echo "<trace name=\"HSA\"><parameters api=\"hsa_agent_get_info, hsa_amd_memory_pool_allocate\"></parameters></trace>" > /tmp/input.xml
export ROCP_INPUT=/tmp/input.xml
eval_test "tool HSA test input" ./test/copy copy_hsa_input_trace
unset ROCP_INPUT

# Check that the tracer tool can be unloaded and then reloaded.
eval_test "Load/Unload/Reload the tracer tool" ./test/load_unload_reload_test load_unload_reload_trace

export LD_PRELOAD=${BIN_DIR}/test/libcodeobj_test.so
eval_test "tool tracer codeobj" ./test/MatrixTranspose code_obj_trace

unset LD_PRELOAD
#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

eval_test "directed TraceBuffer test" ./test/trace_buffer trace_buffer
eval_test "directed MemoryPool test" ./test/memory_pool memory_pool
eval_test "enable/disable callbacks and activities test" ./test/activity_and_callback activity_and_callback_trace
eval_test "use multiple memory pools in HIP activities test" ./test/multi_pool_activities multi_pool_activities_trace
eval_test "Dynamically load the tracer library test" ./test/dlopen dlopen

eval_test "backward compatibility tests" ./test/backward_compat_test backward_compat_test_trace

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
