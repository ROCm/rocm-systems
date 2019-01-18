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
# test check routin
test_status=0
eval_test() {
  label=$1
  cmdline=$2
  echo "$label: \"$cmdline\""
  eval "$cmdline"
  if [ $? != 0 ] ; then
    echo "$label: FAILED"
    test_status=$(($test_status + 1))
  else
    echo "$label: PASSED"
  fi
}

# Standalone test
# rocTrecer is used explicitely by test
eval_test "standalone HIP test" "LD_PRELOAD=$HCC_HOME/lib/libmcwamp_hsa.so ./test/MatrixTranspose_test"

# Tool test
# rocTracer/tool is loaded by HSA runtime
export HSA_TOOLS_LIB="test/libtracer_tool.so libroctracer64.so"

# HIP test
eval_test "tool HIP test" "LD_PRELOAD='$HCC_HOME/lib/libmcwamp_hsa.so $HSA_TOOLS_LIB' ./test/MatrixTranspose"

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

eval_test "tool HSA test" "LD_PRELOAD='$HSA_TOOLS_LIB' ./test/hsa/ctrl"

echo "<trace name=\"HSA\"><parameters list=\"hsa_shut_down, hsa_amd_memory_pool_allocate\"></parameters></trace>" > input.xml
export ROCP_INPUT=input.xml
eval_test "tool HSA test input" "LD_PRELOAD='$HSA_TOOLS_LIB' ./test/hsa/ctrl"

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

if [ $test_status != 0 ] ; then echo "$test_status tests failed"; fi
exit $test_status
