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

test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# test check routin
test_status=0
test_number=0
xeval_test() {
  test_number=$test_number
}
eval_test() {
  label=$1
  cmdline=$2
  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "$label: \"$cmdline\""
    eval "$cmdline"
    if [ $? != 0 ] ; then
      echo "$label: FAILED"
      test_status=$(($test_status + 1))
    else
      echo "$label: PASSED"
    fi
  fi
  test_number=$((test_number + 1))
}

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$PWD
# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so
# enable error messages logging to '/tmp/rocprofiler_log.txt'
export ROCPROFILER_LOG=1
# ROC profiler metrics config file
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
export ROCP_METRICS=metrics.xml
# test trace
export ROC_TEST_TRACE=1

## Intercepting usage model test

# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=./test/libintercept_test.so
export ROCP_KITER=50
export ROCP_DITER=50
export ROCP_AGENTS=1
export ROCP_THRS=1
eval_test "Intercepting usage model test" "../bin/run_tool.sh ./test/ctrl"

## Standalone sampling usage model test

unset ROCP_TOOL_LIB
eval_test "Standalone sampling usage model test" ./test/standalone_test

## Libtool test

# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=libtool.so
# ROC profiler kernels timing
export ROCP_TIMESTAMP_ON=1
# output directory for the tool library, for metrics results file 'results.txt'
# and SQTT trace files 'thread_trace.se<n>.out'
export ROCP_OUTPUT_DIR=./RESULTS

if [ ! -e $ROCP_TOOL_LIB ] ; then
  export ROCP_TOOL_LIB=test/libtool.so
fi

export ROCP_KITER=50
export ROCP_DITER=50
export ROCP_AGENTS=1
export ROCP_THRS=1
export ROCP_INPUT=input.xml
eval_test "'rocprof' libtool test" ./test/ctrl

export ROCP_KITER=10
export ROCP_DITER=10
export ROCP_AGENTS=1
export ROCP_THRS=10
export ROCP_INPUT=input1.xml
eval_test "'rocprof' libtool test n-threads" ./test/ctrl

## SPM test

export ROCP_KITER=1
export ROCP_DITER=1
export ROCP_AGENTS=1
export ROCP_THRS=1
export ROCP_INPUT=spm_input.xml
xeval_test "libtool test, SPM trace test" ./test/ctrl

## Libtool test, counter sets

# Memcopies tracking
export ROCP_MCOPY_TRACKING=1

export ROCP_KITER=1
export ROCP_DITER=4
export ROCP_INPUT=input2.xml
eval_test "libtool test, counter sets" ./test/ctrl

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_status tests failed"
exit $test_status
