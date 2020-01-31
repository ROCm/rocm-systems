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
export LD_LIBRARY_PATH=$PWD

# test filter input
test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# traces comparison
trace_level=0
# 0 is no trace comparison
# 1 is events count comparison
# 2 is events order comparison 
if [ -n "$2" ] ; then
  trace_level=$2
fi

# debugger
debugger=""
if [ -n "$3" ] ; then
  debugger=$3
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
  trace="${3}.txt"
  rtrace="${3}_r.txt"
  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "$label: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$debugger $cmdline" | tee $rtrace
    if [ $? != 0 ] ; then
      echo "$label: FAILED"
      test_status=$(($test_status + 1))
    else
      case "$trace_level" in
        "0") echo "$label: PASSED"
        ;;
        "1") 
          echo "../script/parse_trace.py -in $rtrace  -cn"
          cnt_r=`../script/parse_trace.py -in $rtrace  -cn`
          cnt=`../script/parse_trace.py -in $trace  -cn`
          if [ "$cnt_r" = "$cnt" ] ; then
            echo "$label: PASSED (trace compare, events count)"
          else
            echo "$label: FAILED (trace compare, events count)"
            test_status=$(($test_status + 1))
          fi
        ;;
        "2")
          echo "../script/parse_trace.py -in $rtrace  -or"
          cnt_r=`../script/parse_trace.py -in $rtrace  -or`
          cnt=`../script/parse_trace.py -in $trace  -or`
          if [ "$cnt_r" = "$cnt" ] ; then
            echo "$label: PASSED (trace compare, events order)"
          else
            echo "$label: FAILED (trace compare, events order)"
            test_status=$(($test_status + 1))
          fi
       ;; 
      esac
    fi
  fi
  test_number=$((test_number + 1))
}

# Standalone test
# rocTrecer is used explicitely by test
eval_test "standalone C test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_ctest" "test/MatrixTranspose_ctest_trace"
eval_test "standalone HIP test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_test" "test/MatrixTranspose_test_trace"
eval_test "standalone HIP MGPU test" "LD_PRELOAD=libkfdwrapper64.so ./test/MatrixTranspose_mgpu" "test/MatrixTranspose_mgpu_trace"

# Tool test
# rocTracer/tool is loaded by HSA runtime
export HSA_TOOLS_LIB="test/libtracer_tool.so"

# SYS test
export ROCTRACER_DOMAIN="sys:roctx"
eval_test "tool SYS test" ./test/MatrixTranspose "test/MatrixTranspose_sys_trace"
export ROCTRACER_DOMAIN="sys:hsa:roctx"
eval_test "tool SYS/HSA test" ./test/MatrixTranspose "test/MatrixTranspose_sys_hsa_trace"
# Tracing control <delay:length:rate>
export ROCTRACER_DOMAIN="hip"
eval_test "tool period test" "ROCP_CTRL_RATE=10:100000:1000000 ./test/MatrixTranspose" "test/MatrixTranspose_hip_trace"

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

eval_test "tool HSA test" ./test/hsa/ctrl "test/ctrl_hsa_trace"

echo "<trace name=\"HSA\"><parameters api=\"hsa_agent_get_info, hsa_amd_memory_pool_allocate\"></parameters></trace>" > input.xml
export ROCP_INPUT=input.xml
eval_test "tool HSA test input" ./test/hsa/ctrl "test/ctrl_hsa_input_trace"

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
exit $test_status
