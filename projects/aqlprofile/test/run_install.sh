#!/bin/sh -x


# turn on verbose mode
BIN_NAME=`basename $0`
echo $BIN_NAME | grep "_v." >/dev/null 2>&1
if [ $? = 0 ] ; then set -x; fi
BIN_PATH=`realpath $0`
BIN_DIR=`dirname $0`
cd $BIN_DIR

# enable tools load failure reporting
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
# paths to ROC profiler and other libraries
export LD_LIBRARY_PATH=$BIN_DIR/../../../lib:$LD_LIBRARY_PATH
# test binary
tbin=./ctrl

# test filter input
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

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

echo "$test_number tests total / $test_runnum tests run / $test_status tests failed"
if [ $test_status != 0 ] ; then
  echo $failed_tests
fi
exit $test_status
