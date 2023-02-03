#!/bin/bash

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

# test filter input
test_filter=-1
if [ -n "$1" ] ; then
  test_filter=$1
fi

# test check routine
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
    echo "$label: \"$cmdline\""
    test_runnum=$((test_runnum + 1))
    eval "$cmdline"  > /dev/null 2>&1
    if [ $? != 0 ] ; then
      echo "${bright:-}${blue:-}$test_name: ${red:-}FAILED${normal:-}"
      failed_tests="$failed_tests\n  $test_number: \"$label\""
      test_status=$(($test_status + 1))
    else
      echo "${bright:-}${blue:-}$test_name: ${green:-}PASSED${normal:-}"
    fi
  fi
  test_number=$((test_number + 1))
}

CURRENT_DIR="$( dirname -- "$0"; )";

## Discrete multi-threaded/multi-gpu api test
eval_test "${bright:-}${green:-}running multi-threaded api test..."${normal:-} ${CURRENT_DIR}/profiler_api_test api_test 

## Discrete multi-process binary test
eval_test "${bright:-}${green:-}running multi-process binary test..."${normal:-} ${CURRENT_DIR}/profiler_multiprocess_test multiprocess_test 

## Discrete multi-threaded binary test
eval_test "${bright:-}${green:-}running multi-threaded binary test..."${normal:-} ${CURRENT_DIR}/profiler_multithreaded_test multithreaded_test 

## Discrete multi-queue binary test
#eval_test "${bright:-}${green:-}running multi-queue binary test..."${normal:-} ${CURRENT_DIR}/profiler_multiqueue_test multiqueue_test 

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
