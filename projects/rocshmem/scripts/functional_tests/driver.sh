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
  ["alltoallv"]="20"
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
  ["alltoallmem_on_stream"]="76"
  ["barrier_all_on_stream"]="77"
  ["broadcastmem_on_stream"]="78"
  ["getmem_on_stream"]="79"
  ["putmem_on_stream"]="80"
  ["putmem_signal_on_stream"]="81"
  ["signal_wait_until_on_stream"]="82"
  ["flood_put"]="83"
  ["flood_putnbi"]="84"
  ["flood_p"]="85"
  ["flood_get"]="86"
  ["flood_getnbi"]="87"
  ["flood_g"]="88"
)

ExecTest() {
  TEST_NAME=$1
  NUM_RANKS=$2
  NUM_WG=$3
  NUM_THREADS=$4
  MAX_MSG_SIZE=$5
  if [[ "" == "$NOTIMEOUT" ]]; then
    TIMEOUT=$((5 * 60)) # Timeout in seconds
  fi
  HEAP_SIZE=$((6*1024*1024*1024))

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

  if [[ "" != "$ROCSHMEM_TEST_USE_DEFAULT_STREAM" ]]
  then
    OPTIONS+=" -x ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"
  fi

  if [[ "" != "$HOSTFILE" ]]
  then
    OPTIONS+=" --hostfile $HOSTFILE"
  fi

  # Build command as an array to avoid command injection with eval
  local -a cmd
  cmd=( "$LAUNCHER"
        -n "$NUM_RANKS"
        -mca pml ucx
        -mca osc ucx
        -x "ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS"
        -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"
        -x "ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
        ${ROCSHMEM_TEST_USE_DEFAULT_STREAM:+-x "ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"}
        ${ROCSHMEM_TEST_UUID:+-x "ROCSHMEM_TEST_UUID=$ROCSHMEM_TEST_UUID"}
        ${TIMEOUT:+--timeout "$TIMEOUT"}
        ${HOSTFILE:+--hostfile "$HOSTFILE"}
        --map-by numa
      )
  # Construct Test Command
  TEST_LOG_NAME="$TEST_NAME"_n"$NUM_RANKS"_w"$NUM_WG"_z"$NUM_THREADS"
  cmd+=( "$APP" -a "$TEST_NUM" -w "$NUM_WG" -z "$NUM_THREADS" ${NOVERIF:+-noverif} )
  if [[ "" != "$MAX_MSG_SIZE" ]]
  then
    # Check if in volume mode
    if [[ $MAX_MSG_SIZE == v* ]]; then
      cmd+=( -v "${MAX_MSG_SIZE#v}" )
    else
      cmd+=( -s "$MAX_MSG_SIZE" )
    fi
    TEST_LOG_NAME+=_"$MAX_MSG_SIZE"B
  fi
  # Create a human-readable representation of the command for logging purposes
  CMD="${cmd[@]}"
  # Run Test
  if [ $NUM_GPUS -ge $NUM_RANKS ] || [[ "" != "$HOSTFILE" ]]; then
    echo "Test:   $TEST_LOG_NAME"
    echo "# $CMD >> $LOG_DIR/$TEST_LOG_NAME.log" >"$LOG_DIR/$TEST_LOG_NAME.log"
    "${cmd[@]}" >>"$LOG_DIR/$TEST_LOG_NAME.log" 2>&1
  else
    echo "Skip:   $TEST_LOG_NAME ($NUM_RANKS greater than $NUM_GPUS)"
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

TestRMAPut() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "put"              2       1            1         1048576
  ExecTest  "put"              2       1            1024      512
  ExecTest  "put"              2       8            1         1048576
  ExecTest  "put"              2       16           128       8
  ExecTest  "put"              2       32           256       512
  ExecTest  "put"              2       64           1024      8

  ExecTest  "wgput"            2       1            64        1048576
  ExecTest  "wgput"            2       2            64        1048576
  ExecTest  "wgput"            2       16           64        8

  ExecTest  "waveput"          2       1            64        1048576
  ExecTest  "waveput"          2       2            64        1048576
  ExecTest  "waveput"          2       2            128       1048576
  ExecTest  "waveput"          2       16           128       8

  ExecTest  "defaultctxput"    2       4            128       1024
  ExecTest  "teamctxput"       2       4            128       1024
  ExecTest  "teamctxput"       2       16           256       1024

  ExecTest  "p"                2       1            1         128
  ExecTest  "p"                2       1            1024      2
  ExecTest  "p"                2       8            1         32
  ExecTest  "p"                2       16           128       4

  ExecTest  "shmemptr"         2       1            1         8
  ExecTest  "shmemptr"         2       1            1024      8
  ExecTest  "shmemptr"         2       8            1         8
  ExecTest  "shmemptr"         2       16           128       8

  ExecTest  "putmem_on_stream" 2       1            1         1048576

  export ROCSHMEM_TEST_USE_DEFAULT_STREAM=1
  ExecTest  "putmem_on_stream" 2       1            1         1048576
  unset ROCSHMEM_TEST_USE_DEFAULT_STREAM

  ################################ Non-Blocking ################################

  ExecTest  "putnbi"           2       1            1         1048576
  ExecTest  "putnbi"           2       1            1024      512
  ExecTest  "putnbi"           2       8            1         1048576
  ExecTest  "putnbi"           2       16           128       8
  ExecTest  "putnbi"           2       32           256       512
  ExecTest  "putnbi"           2       64           1024      8

  ExecTest  "wgputnbi"         2       1            64        1048576
  ExecTest  "wgputnbi"         2       2            64        1048576
  ExecTest  "wgputnbi"         2       16           64        8

  ExecTest  "waveputnbi"       2       1            64        1048576
  ExecTest  "waveputnbi"       2       2            64        1048576
  ExecTest  "waveputnbi"       2       2            128       1048576
  ExecTest  "waveputnbi"       2       16           128       8

  ExecTest  "defaultctxputnbi" 2       4            128       1024
  ExecTest  "teamctxputnbi"    2       4            128       1024
  ExecTest  "teamctxputnbi"    2       16           256       1024
}

TestRMAGet() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "get"              2       1            1         1048576
  ExecTest  "get"              2       1            1024      512
  ExecTest  "get"              2       8            1         1048576
  ExecTest  "get"              2       16           128       8
  ExecTest  "get"              2       32           256       512
  ExecTest  "get"              2       64           1024      8

  ExecTest  "wgget"            2       1            64        1048576
  ExecTest  "wgget"            2       2            64        1048576
  ExecTest  "wgget"            2       16           64        8

  ExecTest  "waveget"          2       1            64        1048576
  ExecTest  "waveget"          2       2            64        1048576
  ExecTest  "waveget"          2       2            128       1048576
  ExecTest  "waveget"          2       16           128       8

  ExecTest  "defaultctxget"    2       4            128       1024
  ExecTest  "teamctxget"       2       4            128       1024
  ExecTest  "teamctxget"       2       16           256       1024

  ExecTest  "g"                2       1            1         128
  ExecTest  "g"                2       1            1024      1
  ExecTest  "g"                2       8            1         32
  ExecTest  "g"                2       16           128       4

  ExecTest  "getmem_on_stream" 2       1            1         1048576

  ################################ Non-Blocking ################################

  ExecTest  "getnbi"           2       1            1         1048576
  ExecTest  "getnbi"           2       1            1024      512
  ExecTest  "getnbi"           2       8            1         1048576
  ExecTest  "getnbi"           2       16           128       8
  ExecTest  "getnbi"           2       32           256       512
  ExecTest  "getnbi"           2       64           1024      8

  ExecTest  "wggetnbi"         2       1            64        1048576
  ExecTest  "wggetnbi"         2       2            64        1048576
  ExecTest  "wggetnbi"         2       16           64        8

  ExecTest  "wavegetnbi"       2       1            64        1048576
  ExecTest  "wavegetnbi"       2       2            64        1048576
  ExecTest  "wavegetnbi"       2       2            128       1048576
  ExecTest  "wavegetnbi"       2       16           128       8

  ExecTest  "defaultctxgetnbi" 2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       16           256       1024
}

TestRMA() {
  TestRMAPut
  TestRMAGet
}

TestAMORO() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "amo_fetch"        2       1            1
  ExecTest  "amo_fetch"        2       1            1024
  ExecTest  "amo_fetch"        2       8            1
  ExecTest  "amo_fetch"        2       32           128

  ExecTest  "amo_set"          2       1            1
  ExecTest  "amo_set"          2       8            1
  ExecTest  "amo_set"          2       32           1

  ExecTest  "amo_fcswap"       2       1            1
  ExecTest  "amo_fcswap"       2       32           1
  ExecTest  "amo_fcswap"       2       8            1

  ExecTest  "amo_fetchand"     2       1            1

  ExecTest  "amo_and"          2       1            1

  ExecTest  "amo_xor"          2       1            1
}

TestAMO() {
  TestAMORO

  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "amo_finc"         2       1            1
  ExecTest  "amo_finc"         2       1            1024
  ExecTest  "amo_finc"         2       8            1
  ExecTest  "amo_finc"         2       32           128

  ExecTest  "amo_inc"          2       1            1
  ExecTest  "amo_inc"          2       1            1024
  ExecTest  "amo_inc"          2       8            1
  ExecTest  "amo_inc"          2       32           128

  ExecTest  "amo_fadd"         2       1            1
  ExecTest  "amo_fadd"         2       1            1024
  ExecTest  "amo_fadd"         2       8            1
  ExecTest  "amo_fadd"         2       32           128

  ExecTest  "amo_add"          2       1            1
  ExecTest  "amo_add"          2       1            1024
  ExecTest  "amo_add"          2       8            1
  ExecTest  "amo_add"          2       32           128
}

TestSigOpsRO() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "putsignal"        2       1            1         1048576
  ExecTest  "putsignal"        2       2            32        1048576
  ExecTest  "wgputsignal"      2       2            32        1048576
  ExecTest  "waveputsignal"    2       1            32        1048576
  ExecTest  "waveputsignal"    2       2            64        1048576

  ExecTest  "putsignalnbi"     2       1            1         1048576
  ExecTest  "putsignalnbi"     2       2            32        1048576
  ExecTest  "wgputsignalnbi"   2       2            32        1048576
  ExecTest  "waveputsignalnbi" 2       1            32        1048576
  ExecTest  "waveputsignalnbi" 2       2            64        1048576

  ExecTest  "signalfetch"      2       1            1
  ExecTest  "wgsignalfetch"    2       2            32
  ExecTest  "wavesignalfetch"  2       1            32
  ExecTest  "wavesignalfetch"  2       1            64

  ExecTest  "signal_wait_until_on_stream" 2  1      1
}

TestSigOps() {
  TestSigOpsRO
  ExecTest  "putmem_signal_on_stream" 2  1          1         1048576
}

TestColl() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "barrierall"       2       1            1

  ExecTest  "wavebarrierall"   2       1            1

  ExecTest  "wgbarrierall"     2       1            1

  ExecTest  "teambarrier"      2       1            1
  ExecTest  "teambarrier"      2       16           64
  ExecTest  "teambarrier"      2       32           256
  ExecTest  "teambarrier"      2       39           1024

  ExecTest  "teamwavebarrier"  2       1            1
  ExecTest  "teamwavebarrier"  2       16           64
  ExecTest  "teamwavebarrier"  2       32           256
  ExecTest  "teamwavebarrier"  2       39           1024

  ExecTest  "teamwgbarrier"    2       1            1
  ExecTest  "teamwgbarrier"    2       16           64
  ExecTest  "teamwgbarrier"    2       32           256
  ExecTest  "teamwgbarrier"    2       39           1024

  ExecTest  "teamsync"         2       1            1
  ExecTest  "teamsync"         2       16           64
  ExecTest  "teamsync"         2       32           256
  ExecTest  "teamsync"         2       39           1024

  ExecTest  "teamwavesync"     2       1            1
  ExecTest  "teamwavesync"     2       16           64
  ExecTest  "teamwavesync"     2       32           256
  ExecTest  "teamwavesync"     2       39           1024

  ExecTest  "teamwgsync"       2       1            1
  ExecTest  "teamwgsync"       2       16           64
  ExecTest  "teamwgsync"       2       32           256
  ExecTest  "teamwgsync"       2       39           1024

  ExecTest  "syncall"          2       1            1

  ExecTest  "wavesyncall"      2       1            1

  ExecTest  "wgsyncall"        2       1            1

  ExecTest  "alltoall"         2       1            64        512

  ExecTest  "teambroadcast"    2       1            64        32768

  ExecTest  "fcollect"         2       1            64        32768

  ExecTest  "teamreduction"    2       1            64        32768

  ExecTest  "alltoallmem_on_stream"  2  1           64        1048576
  ExecTest  "broadcastmem_on_stream" 2  1           64        1048576
  ExecTest  "barrier_all_on_stream"  2  1           1
}

TestOther() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "init"             2       1            1

  ExecTest  "pingpong"         2       1            1
  ExecTest  "pingpong"         2       8            1
  ExecTest  "pingpong"         2       32           1

  ExecTest  "pingall"          2       1            1
  ExecTest  "pingall"          2       8            1
  ExecTest  "pingall"          2       32           1

  ExecTest  "flood_put"        2       64           1024
  ExecTest  "flood_get"        2       64           1024

  ExecTest  "flood_put"        8       64           1024
  ExecTest  "flood_putnbi"     8       64           1024
  ExecTest  "flood_p"          8       64           1024
  ExecTest  "flood_get"        8       64           1024
  ExecTest  "flood_getnbi"     8       64           1024
  ExecTest  "flood_g"          8       64           1024

  # This test requires more contexts than workgroups
  export ROCSHMEM_MAX_NUM_CONTEXTS=1024
  ExecTest  "teamctxinfra"        2       1            1
  ExecTest  "teamctxsingleinfra"  2       1            1
  ExecTest  "teamctxblockinfra"   4       1            1
  ExecTest  "teamctxblockinfra"   5       1            1
  ExecTest  "teamctxoddeveninfra" 4       1            1
  ExecTest  "teamctxoddeveninfra" 5       1            1
  unset ROCSHMEM_MAX_NUM_CONTEXTS
}

# TODO: remove when GDA is feature complete
TestGDA() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "put"              2       1            1         1048576
  ExecTest  "put"              2       1            1024      512
  ExecTest  "put"              2       8            1         1048576
  ExecTest  "put"              2       16           128       8
  ExecTest  "put"              2       32           256       512
  ExecTest  "put"              2       64           1024      8

  ExecTest  "wgput"            2       1            64        1048576
  ExecTest  "wgput"            2       2            64        1048576
  ExecTest  "wgput"            2       16           64        8

  ExecTest  "waveput"          2       1            64        1048576
  ExecTest  "waveput"          2       2            64        1048576
  ExecTest  "waveput"          2       2            128       1048576
  ExecTest  "waveput"          2       16           128       8

  ExecTest  "defaultctxput"    2       4            128       1024
  ExecTest  "teamctxput"       2       4            128       1024
  ExecTest  "teamctxput"       2       16           256       1024

  ExecTest  "get"              2       1            1         1048576
  ExecTest  "get"              2       1            1024      512
  ExecTest  "get"              2       8            1         1048576
  ExecTest  "get"              2       16           128       8
  ExecTest  "get"              2       32           256       512
  ExecTest  "get"              2       64           1024      8

  ExecTest  "wgget"            2       1            64        1048576
  ExecTest  "wgget"            2       2            64        1048576
  ExecTest  "wgget"            2       16           64        8

  ExecTest  "waveget"          2       1            64        1048576
  ExecTest  "waveget"          2       2            64        1048576
  ExecTest  "waveget"          2       2            128       1048576
  ExecTest  "waveget"          2       16           128       8

  ExecTest  "defaultctxget"    2       4            128       1024
  ExecTest  "teamctxget"       2       4            128       1024
  ExecTest  "teamctxget"       2       16           256       1024

#  ExecTest  "g"                2       1            1         128
#  ExecTest  "g"                2       1            1024      2
#  ExecTest  "g"                2       8            1         32
#  ExecTest  "g"                2       16           128       4

  ExecTest  "p"                2       1            1         128
  ExecTest  "p"                2       1            1024      2
  ExecTest  "p"                2       8            1         32
  ExecTest  "p"                2       16           128       4

  ################################ Non-Blocking ################################

  ExecTest  "putnbi"           2       1            1         1048576
  ExecTest  "putnbi"           2       1            1024      512
  ExecTest  "putnbi"           2       8            1         1048576
  ExecTest  "putnbi"           2       16           128       8
  ExecTest  "putnbi"           2       32           256       512
  ExecTest  "putnbi"           2       64           1024      8

  ExecTest  "wgputnbi"         2       1            64        1048576
  ExecTest  "wgputnbi"         2       2            64        1048576
  ExecTest  "wgputnbi"         2       16           64        8

  ExecTest  "waveputnbi"       2       1            64        1048576
  ExecTest  "waveputnbi"       2       2            64        1048576
  ExecTest  "waveputnbi"       2       2            128       1048576
  ExecTest  "waveputnbi"       2       16           128       8

  ExecTest  "defaultctxputnbi" 2       4            128       1024
  ExecTest  "teamctxputnbi"    2       4            128       1024
  ExecTest  "teamctxputnbi"    2       16           256       1024

  ExecTest  "getnbi"           2       1            1         1048576
  ExecTest  "getnbi"           2       1            1024      512
  ExecTest  "getnbi"           2       8            1         1048576
  ExecTest  "getnbi"           2       16           128       8
  ExecTest  "getnbi"           2       32           256       512
  ExecTest  "getnbi"           2       64           1024      8

  ExecTest  "wggetnbi"         2       1            64        1048576
  ExecTest  "wggetnbi"         2       2            64        1048576
  ExecTest  "wggetnbi"         2       16           64        8

  ExecTest  "wavegetnbi"       2       1            64        1048576
  ExecTest  "wavegetnbi"       2       2            64        1048576
  ExecTest  "wavegetnbi"       2       2            128       1048576
  ExecTest  "wavegetnbi"       2       16           128       8

  ExecTest  "defaultctxgetnbi" 2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       16           256       1024

#TestAMO() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "amo_fetch"        2       1            1
  ExecTest  "amo_fetch"        2       1            1024
  ExecTest  "amo_fetch"        2       8            1
  ExecTest  "amo_fetch"        2       32           128

  ExecTest  "amo_set"          2       1            1
  ExecTest  "amo_set"          2       8            1
  ExecTest  "amo_set"          2       32           1

  ExecTest  "amo_fcswap"       2       1            1
  ExecTest  "amo_fcswap"       2       32           1
  ExecTest  "amo_fcswap"       2       8            1

  ExecTest  "amo_finc"         2       1            1
  ExecTest  "amo_finc"         2       1            1024
  ExecTest  "amo_finc"         2       8            1
  ExecTest  "amo_finc"         2       32           128

  ExecTest  "amo_inc"          2       1            1
  ExecTest  "amo_inc"          2       1            1024
  ExecTest  "amo_inc"          2       8            1
  ExecTest  "amo_inc"          2       32           128

  ExecTest  "amo_fadd"         2       1            1
  ExecTest  "amo_fadd"         2       1            1024
  ExecTest  "amo_fadd"         2       8            1
  ExecTest  "amo_fadd"         2       32           128

  ExecTest  "amo_add"          2       1            1
  ExecTest  "amo_add"          2       1            1024
  ExecTest  "amo_add"          2       8            1
  ExecTest  "amo_add"          2       32           128

  ExecTest  "amo_fetchand"     2       1            1

  ExecTest  "amo_and"          2       1            1

  ExecTest  "amo_xor"          2       1            1

#TestColl() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "barrierall"       2       1            1
  ExecTest  "teambarrier"      2       1            1

  ExecTest  "teamsync"         2       1            1
  ExecTest  "syncall"          2       1            1

  ExecTest  "alltoall"         2       1            1         512

  ExecTest  "teambroadcast"    2       1            1         32768

  ExecTest  "fcollect"         2       1            1         32768

# deadlock on gda, size 8KB
#  ExecTest  "teamreduction"    2       1            1         32768

#TestOther() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "init"             2       1            1

  ExecTest  "pingpong"         2       1            1
  ExecTest  "pingpong"         2       8            1
  ExecTest  "pingpong"         2       32           1

  ExecTest  "flood_put"        2       64           1024
  ExecTest  "flood_get"        2       64           1024

  ExecTest  "flood_put"        8       64           1024
  ExecTest  "flood_putnbi"     8       64           1024
  ExecTest  "flood_p"          8       64           1024
  ExecTest  "flood_get"        8       64           1024
  ExecTest  "flood_getnbi"     8       64           1024
#  ExecTest  "flood_g"          8       64           1024 # _g not implemented

  # This test requires more contexts than workgroups
  export ROCSHMEM_MAX_NUM_CONTEXTS=1024
  ExecTest  "teamctxinfra"        2       1            1
  ExecTest  "teamctxsingleinfra"  2       1            1
  ExecTest  "teamctxblockinfra"   4       1            1
  ExecTest  "teamctxblockinfra"   5       1            1
  ExecTest  "teamctxoddeveninfra" 4       1            1
  ExecTest  "teamctxoddeveninfra" 5       1            1
  unset ROCSHMEM_MAX_NUM_CONTEXTS
}

TestHeatMapRMA() {
  NOTIMEOUT=1
  NOVERIF=1
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "get"              2       1            1         v1048576
  ExecTest  "get"              2       32           1024      v1073741824
  ExecTest  "waveget"          2       1            64        v1073741824
  ExecTest  "waveget"          2       2            64        v1073741824
  ExecTest  "waveget"          2       16           1024      v1073741824
  ExecTest  "wgget"            2       1            1024      v1073741824
  ExecTest  "wgget"            2       16           1024      v1073741824
  #ExecTest  "wgget"            2       32           1024      v1073741824

  ExecTest  "put"              2       1            1         v1048576
  ExecTest  "put"              2       32           1024      v1073741824
  ExecTest  "waveput"          2       1            64        v1073741824
  ExecTest  "waveput"          2       2            64        v1073741824
  ExecTest  "waveput"          2       16           1024      v1073741824
  ExecTest  "wgput"            2       1            1024      v1073741824
  ExecTest  "wgput"            2       16           1024      v1073741824
  #ExecTest  "wgput"            2       32           1024      v1073741824
}

TestHeatMapColl() {
  NOTIMEOUT=1
  NOVERIF=1
  ExecTest  "alltoall"         2       1            256        v1073741824
  ExecTest  "alltoall"         4       1            256        v1073741824
  ExecTest  "alltoall"         8       1            256        v1073741824
  ExecTest  "alltoall"         16      1            256        v1073741824
  ExecTest  "alltoall"         32      1            256        v1073741824
  ExecTest  "alltoall"         64      1            256        v1073741824
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
    TestHeatMapRMA
    TestHeatMapColl
    ;;
  *"gda")
    TestGDA
    ;;
  *"all")
    TestRMA
    TestAMO
    TestSigOps
    TestColl
    TestOther
    ;;
  *"all-ro")
    TestRMAPut
    TestAMORO
    TestSigOpsRO
    TestColl
    TestOther
    ;;
  *"rma")
    TestRMA
    ;;
  *"put")
    TestRMAPut
    ;;
  *"get")
    TestRMAGet
    ;;
  *"amo")
    TestAMO
    ;;
  *"sigops")
    TestSigOps
    ;;
  *"coll")
    TestColl
    ;;
  *"other")
    TestOther
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
