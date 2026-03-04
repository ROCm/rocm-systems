#!/bin/sh -x

# MIT License
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# turn on verbose mode
BIN_NAME=`basename $0`
echo $BIN_NAME | grep "_v." >/dev/null 2>&1
if [ $? = 0 ] ; then set -x; fi
BIN_PATH=`realpath $0`
BIN_DIR=`dirname $0`
cd $BIN_DIR

#To enable symbol lookup in .dynsyn section after llvm-strip
export LOADER_USE_DYNSYM=1

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and other libraries
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
# test binary
tbin=./ctrl

# Timeout in seconds
TEST_TIMEOUT=90
test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# test check routin
test_status=0
test_runnum=0
test_number=0
failed_tests="Failed tests:"

xeval_test() {
  test_number=$test_number
}

# Run a command with a timeout, kill it if it exceeds the limit
run_with_timeout() {
  cmdline=$1
  eval "$cmdline" &
  cmd_pid=$!

  # watchdog: kill after TEST_TIMEOUT seconds
  ( sleep $TEST_TIMEOUT; kill $cmd_pid 2>/dev/null ) &
  watchdog_pid=$!

  wait $cmd_pid
  exit_code=$?

  # cancel the watchdog if command finished in time
  kill $watchdog_pid 2>/dev/null
  wait $watchdog_pid 2>/dev/null

  # 143 = killed by SIGTERM, treat as timeout/hang
  if [ $exit_code = 143 ] ; then
    echo "TIMEOUT: test killed after $TEST_TIMEOUT seconds"
    return 1
  fi
  return $exit_code
}

eval_test() {
  label=$1
  cmdline=$2
  test_trace=$test_name.txt

  if [ $test_filter = -1  -o $test_filter = $test_number ] ; then
    echo "test $test_number: $test_name \"$label\""
    test_runnum=$((test_runnum + 1))
    eval "$cmdline"
    is_failed=$?
    if [ $is_failed = 0 ] ; then
      echo "$test_name: PASSED"
    else
      echo "$test_name: FAILED"
      failed_tests="$failed_tests\n  $test_number: \"$label\""
      test_status=$(($test_status + 1))
    fi
  fi

  test_number=$((test_number + 1))
}

cd `dirname $BIN_PATH`

# Simple convolution kernel dry run
unset AQLPROFILE_PMC
unset AQLPROFILE_PMC_PRIV
unset AQLPROFILE_SQTT
unset AQLPROFILE_SDMA
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval_test "simple convolution kernel dry run" $tbin

# Run with PMC
export AQLPROFILE_PMC=1
unset AQLPROFILE_PMC_PRIV
unset AQLPROFILE_SQTT
unset AQLPROFILE_SDMA
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval_test "PMC test" $tbin

# Run with SQTT
unset AQLPROFILE_PMC
unset AQLPROFILE_PMC_PRIV
export AQLPROFILE_SQTT=1
unset AQLPROFILE_SDMA
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval_test "SQTT test" $tbin

# Run with PCSMP
unset AQLPROFILE_PMC
unset AQLPROFILE_PMC_PRIV
unset AQLPROFILE_SQTT
export AQLPROFILE_PCSMP=1
unset AQLPROFILE_SDMA
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval_test "PCSMP test" $tbin

# --- SPM ---
test_name="spm"
unset AQLPROFILE_PMC AQLPROFILE_PMC_PRIV AQLPROFILE_SQTT AQLPROFILE_PCSMP
unset AQLPROFILE_SDMA AQLPROFILE_SCAN
export AQLPROFILE_READ_API=0
export AQLPROFILE_SPM=1
export AQLPROFILE_SPM_KFD_MODE=1
export AQLPROFILE_SPM_SAMPLE_RATE=1600
export ROCP_SPM_KFD_MODE=1
export HSA_ENABLE_SDMA=0
eval_test "SPM test" $tbin
unset AQLPROFILE_SPM AQLPROFILE_SPM_KFD_MODE AQLPROFILE_SPM_SAMPLE_RATE
unset ROCP_SPM_KFD_MODE AQLPROFILE_READ_API HSA_ENABLE_SDMA

# --- Summary ---
echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
