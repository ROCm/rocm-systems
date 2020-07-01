#!/bin/sh -x

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
cd $BIN_DIR

IS_CI=0
if [ "$BIN_NAME" = "run_ci.sh" ] ; then
  IS_CI=1
fi

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and other libraries
export LD_LIBRARY_PATH=$PWD
if [ $IS_CI = 1 ] ; then
  export LD_LIBRARY_PATH=/opt/rocm/roctracer/lib
fi
if [ -n "$ROCTRACER_LIB_PATH" ] ; then
  export LD_LIBRARY_PATH=$ROCTRACER_LIB_PATH
fi
if [ -z "$ROCTRACER_TOOL_PATH" ] ; then
  ROCTRACER_TOOL_PATH="./test"
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

eval_test() {
  label=$1
  cmdline=$2
  test_name=$3
  test_trace=$test_name.txt

  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "test $test_number: $test_name \"$label\""
    echo "CMD: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$cmdline" >$test_trace 2>&1
    is_failed=$?
    cat $test_trace
    if [ $IS_CI = 1 ] ; then
      is_failed=0;
    else
      if [ $is_failed = 0 ] ; then
        python ./test/check_trace.py -in $test_name -ck $check_trace_flag
        is_failed=$?
        if [ $is_failed != 0 ] ; then
          python ./test/check_trace.py -v -in $test_name -ck $check_trace_flag
        fi
      fi
    fi
    if [ $is_failed = 0 ] ; then
      echo "$test_name: PASSED"
    else
      echo "$test_name: FAILED"
      failed_tests="$failed_tests\n  $test_number: $test_name - \"$label\""
      test_status=$(($test_status + 1))
    fi
  fi

  test_number=$((test_number + 1))
}

# Standalone test
# rocTrecer is used explicitely by test
eval_test "standalone C test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_ctest" MatrixTranspose_ctest_trace
eval_test "standalone HIP test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_test" MatrixTranspose_test_trace
eval_test "standalone HIP hipaact test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_hipaact_test" MatrixTranspose_hipaact_test_trace
eval_test "standalone HIP MGPU test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_mgpu" MatrixTranspose_mgpu_trace

# Tool test
# rocTracer/tool is loaded by HSA runtime
export HSA_TOOLS_LIB="$ROCTRACER_TOOL_PATH/libtracer_tool.so"

# KFD test
export ROCTRACER_DOMAIN="kfd"
eval_test "tool KFD test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose" MatrixTranspose_kfd_trace
# SYS test
export ROCTRACER_DOMAIN="sys:roctx"
eval_test "tool SYS test" ./test/MatrixTranspose MatrixTranspose_sys_trace
export ROCTRACER_DOMAIN="sys:hsa:roctx"
eval_test "tool SYS/HSA test" ./test/MatrixTranspose MatrixTranspose_sys_hsa_trace
# Tracing control <delay:length:rate>
export ROCTRACER_DOMAIN="hip"
eval_test "tool period test" "ROCP_CTRL_RATE=10:100000:1000000 ./test/MatrixTranspose" MatrixTranspose_hip_period_trace
eval_test "tool flushing test" "ROCP_FLUSH_RATE=100000 ./test/MatrixTranspose" MatrixTranspose_hip_flush_trace

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

eval_test "tool HSA test" ./test/hsa/ctrl ctrl_hsa_trace

echo "<trace name=\"HSA\"><parameters api=\"hsa_agent_get_info, hsa_amd_memory_pool_allocate\"></parameters></trace>" > input.xml
export ROCP_INPUT=input.xml
eval_test "tool HSA test input" ./test/hsa/ctrl ctrl_hsa_input_trace

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
