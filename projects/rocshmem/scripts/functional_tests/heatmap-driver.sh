###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

#!/bin/bash
if true || tty -s; then
  PRETTY_FAILED="\033[1;31mFAILED\033[0m"
  PRETTY_PASSED="\033[1;32mPASSED\033[0m"
else
  PRETTY_FAILED="FAILED"
  PRETTY_PASSED="PASSED"
fi

# This names/values should match the TestType enum in rocSHMEM/tests/functional_tests/tester.hpp
# and in driver.sh
declare -A TEST_NUMBERS=(
  ["get"]="0"
  ["getnbi"]="1"
  ["put"]="2"
  ["putnbi"]="3"
  ["amo_fadd"]="4"
  ["amo_finc"]="5"
  ["amo_fetch"]="6"
  ["amo_fcswap"]="7"
  ["amo_add"]="8"
  ["amo_inc"]="9"
  ["amo_cswap"]="10"
  ["init"]="11"
  ["pingpong"]="12"
  ["randomaccess"]="13"
  ["barrierall"]="14"
  ["syncall"]="15"
  ["teamsync"]="16"
  ["collect"]="17"
  ["fcollect"]="18"
  ["alltoall"]="19"
  ["alltoalls"]="20"
  ["shmemptr"]="21"
  ["p"]="22"
  ["g"]="23"
  ["wgget"]="24"
  ["wggetnbi"]="25"
  ["wgput"]="26"
  ["wgputnbi"]="27"
  ["waveget"]="28"
  ["wavegetnbi"]="29"
  ["waveput"]="30"
  ["waveputnbi"]="31"
  ["teambroadcast"]="32"
  ["teamreduction"]="33"
  ["teamctxget"]="34"
  ["teamctxgetnbi"]="35"
  ["teamctxput"]="36"
  ["teamctxputnbi"]="37"
  ["teamctxinfra"]="38"
  ["putnbimr"]="39"
  ["amo_set"]="40"
  ["amo_swap"]="41"
  ["amo_fetchand"]="42"
  ["amo_fetchor"]="43"
  ["amo_fetchxor"]="44"
  ["amo_and"]="45"
  ["amo_or"]="46"
  ["amo_xor"]="47"
  ["pingall"]="48"
  ["putsignal"]="49"
  ["wgputsignal"]="50"
  ["waveputsignal"]="51"
  ["putsignalnbi"]="52"
  ["wgputsignalnbi"]="53"
  ["waveputsignalnbi"]="54"
  ["signalfetch"]="55"
  ["wgsignalfetch"]="56"
  ["wavesignalfetch"]="57"
  ["teamwgbarrier"]="58"
  ["defaultctxget"]="59"
  ["defaultctxgetnbi"]="60"
  ["defaultctxput"]="61"
  ["defaultctxputnbi"]="62"
  ["defaultctxp"]="63"
  ["defaultctxg"]="64"
  ["wavebarrierall"]="65"
  ["wgbarrierall"]="66"
  ["wavesyncall"]="67"
  ["wgsyncall"]="68"
  ["teambarrier"]="69"
  ["teamwavebarrier"]="70"
  ["teamwavesync"]="71"
  ["teamwgsync"]="72"
  ["teamctxsingleinfra"]="73"
  ["teamctxblockinfra"]="74"
  ["teamctxoddeveninfra"]="75"
)

ExecTest() {
  TEST_NAME=$1
  NUM_RANKS=$2
  NUM_WG=$3
  NUM_THREADS=$4
  MAX_MSG_SIZE=$5
  HEAP_SIZE=$((48*1024*1024*1024))

  if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(amd-smi list | grep GPU | wc -l)}
  elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null
  then
    NUM_GPUS=${NUM_GPUS:-$(rocm-smi --showserial | grep GPU | wc -l)}
  fi
  NUM_GPUS=${NUM_GPUS:-0}
  NUM_GPUS=$(($NUM_GPUS > 0? $NUM_GPUS: 8))

  TEST_NUM=${TEST_NUMBERS[$TEST_NAME]}

  if [[ "" == "$TEST_NUM" ]]
  then
    echo "Test $TEST_NAME does not exist" >&2
    DRIVER_RETURN_STATUS=1
    return
  fi

  if [[ "" == "$ROCSHMEM_MAX_NUM_CONTEXTS" ]]
  then
    ROCSHMEM_MAX_NUM_CONTEXTS=$NUM_WG
  fi

  # MPI Parameters
  LAUNCHER=mpirun
  OPTIONS=" -n $NUM_RANKS -mca pml ucx -mca osc ucx"
  OPTIONS+=" -x ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS"
  OPTIONS+=" -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384 -x ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
  OPTIONS+=" --map-by numa "

  if [[ "" != "$HOSTFILE" ]]
  then
    OPTIONS+=" --hostfile $HOSTFILE"
  fi

  # Construct Test Command
  TEST_LOG_NAME="$TEST_NAME"_n"$NUM_RANKS"_w"$NUM_WG"_z"$NUM_THREADS"
  CMD="$LAUNCHER $OPTIONS $APP -u 1 -a $TEST_NUM -w $NUM_WG -z $NUM_THREADS"

  if [[ "" != "$MAX_MSG_SIZE" ]]
  then
    CMD+=" -s $MAX_MSG_SIZE"
    TEST_LOG_NAME+=_"$MAX_MSG_SIZE"B
  fi

  CMD+=" >> $LOG_DIR/$TEST_LOG_NAME.log 2>&1"

  # Run Test
  if [ $NUM_GPUS -ge $NUM_RANKS ] || [[ "" != "$HOSTFILE" ]]; then
    echo $TEST_LOG_NAME
    echo "# $CMD" >"$LOG_DIR/$TEST_LOG_NAME.log"
    eval $CMD
  else
    echo "Skipping test $TEST_LOG_NAME ($NUM_RANKS greater than $NUM_GPUS)"
  fi

  # Validate Test
  if [ $? -ne 0 ]
  then
    echo -e "$PRETTY_FAILED: $TEST_LOG_NAME" >&2
    cat "$LOG_DIR/$TEST_LOG_NAME.log"
    DRIVER_RETURN_STATUS=1
    FAILED_LIST="$FAILED_LIST $TEST_LOG_NAME"
  fi

  unset ROCSHMEM_MAX_NUM_CONTEXTS
}

TestHeatMapRMA() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "get"              2       1            1         $((1*1024*1024))
  ExecTest  "get"              2       32           1024      $((1*1024*1024))
  ExecTest  "waveget"          2       1            64        1073741824
  ExecTest  "waveget"          2       2            64        1073741824
  ExecTest  "waveget"          2       16           1024      1073741824
  ExecTest  "wgget"            2       1            1024      1073741824
  ExecTest  "wgget"            2       16           1024      1073741824
  #ExecTest  "wgget"            2       32           1024      1073741824

  ExecTest  "put"              2       1            1         $((1*1024*1024))
  ExecTest  "put"              2       32           1024      $((1*1024*1024))
  ExecTest  "waveput"          2       1            64        1073741824
  ExecTest  "waveput"          2       2            64        1073741824
  ExecTest  "waveput"          2       16           1024      1073741824
  ExecTest  "wgput"            2       1            1024      1073741824
  ExecTest  "wgput"            2       16           1024      1073741824
  #ExecTest  "wgput"            2       32           1024      1073741824
}

TestHeatMapColl() {
  ExecTest  "alltoall"         2       1            256        1073741824
  ExecTest  "alltoall"         4       1            256        1073741824
  ExecTest  "alltoall"         8       1            256        1073741824
  ExecTest  "alltoall"         16      1            256        1073741824
  ExecTest  "alltoall"         32      1            256        1073741824
  ExecTest  "alltoall"         64      1            256        1073741824
}

TestHeatMap() {
  TestHeatMapRMA
  TestHeatMapColl
}


ValidateInput() {
  INPUT_COUNT=$1
  if [ $INPUT_COUNT -lt 3 ] ; then
    echo "This script must be run with at least 3 arguments."
    echo 'Usage: ${0} argument1 argument2 argument3 [argument4]'
    echo "  argument1 : path to the tester driver"
    echo "  argument2 : test type to run, e.g put"
    echo "  argument3 : directory to put the output logs"
    echo "  argument4 : path to hostfile"
    exit 1
  fi
}

ValidateLogDir() {
  if [ ! -d $1 ]; then
    echo "LOG_DIR=$1 does not exist"
    mkdir -p $1
    echo "Created $1"
  fi
}

APP=$1
TEST=$2
LOG_DIR=$3
HOSTFILE=$4

DRIVER_RETURN_STATUS=0

ValidateInput $#
ValidateLogDir $LOG_DIR

case $TEST in
  *"heatmaprma")
    TestHeatMapRMA
    ;;
  *"heatmapcoll")
    TestHeatMapColl
    ;;
  *"heatmap")
    TestHeatMap
    ;;
  *)
    ##############################################################################
    #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
    ##############################################################################
    ExecTest  $TEST              2       1            1         8
    ;;
esac

EXIT_STATUS=$(($DRIVER_RETURN_STATUS || $?))
if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "TESTS PASSED"
else
  echo -e "TESTS FAILED: $FAILED_LIST"
fi
exit $EXIT_STATUS
