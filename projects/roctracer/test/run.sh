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

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$PWD:/opt/rocm/hip/lib

# test filter input
test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# debugger
debugger=""
if [ -n "$2" ] ; then
  debugger=$2
fi

# test check routin
test_status=0
test_runnum=0
test_number=0
xeval_test() {
  test_number=$test_number
}
eval_test() {
  label=$1
  cmdline=$2
  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "$label: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$debugger $cmdline"
    if [ $? != 0 ] ; then
      echo "$label: FAILED"
      test_status=$(($test_status + 1))
    else
      echo "$label: PASSED"
    fi
  fi
  test_number=$((test_number + 1))
}

# Standalone test
# rocTrecer is used explicitely by test
eval_test "standalone HIP test" "./test/MatrixTranspose_test"

# Tool test
# rocTracer/tool is loaded by HSA runtime
export HSA_TOOLS_LIB="test/libtracer_tool.so"
export ROCTRACER_DOMAIN="hip"

# HIP test
eval_test "tool HIP test" ./test/MatrixTranspose
# with trace sampling control <delay:length:rate>
eval_test "tool HIP period test" "ROCP_CTRL_RATE=10:100000:1000000 ./test/MatrixTranspose"

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

eval_test "tool HSA test" ./test/hsa/ctrl

echo "<trace name=\"HSA\"><parameters api=\"hsa_agent_get_info, hsa_amd_memory_pool_allocate\"></parameters></trace>" > input.xml
export ROCP_INPUT=input.xml
eval_test "tool HSA test input" ./test/hsa/ctrl

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
exit $test_status
