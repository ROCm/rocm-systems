#!/bin/bash
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
  # 39: putnbimr removed, use putnbi with -b instead
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
  ["hipmodule_init"]="89"
  ["flood_add"]="90"
  ["flood_fadd"]="91"
  ["flood_waitadd"]="92"
  ["device_bitcode"]="93"
  ["library_info"]="94"
  ["teamctxsharedinfra"]="95"
  ["quiet_on_stream"]="96"
  ["sync_all_on_stream"]="97"
  ["teamctxsubsetparentinfra"]="98"
  ["fence_putwavesignal"]="99"
  ["fence_putlargesmall"]="100"
  ["fence_fanout"]="101"
  ["fence_putwavenbichunks"]="102"
  ["tile_put_contiguous"]="103"
  ["tile_put_rowmajor"]="104"
  ["tile_put_colmajor"]="105"
  ["tile_put_arbitrary"]="106"
  ["tile_put_wave_contiguous"]="107"
  ["tile_put_wg_contiguous"]="108"
  ["tile_get_contiguous"]="109"
  ["tile_get_wg_contiguous"]="110"
  ["tile_put_1d"]="111"
  ["tile_get_1d"]="112"
  ["tile_get_wave_contiguous"]="113"
  ["tile_get_rowmajor"]="114"
  ["tile_get_colmajor"]="115"
  ["tile_get_arbitrary"]="116"
  ["reduce_on_stream"]="117"
  ["host_ctx_create"]="118"
)

ExecTest() {
  TEST_NAME=$1
  NUM_RANKS=$2
  NUM_WG=$3
  NUM_THREADS=$4
  MAX_MSG_SIZE=$5
  IS_RETRY=${6:-0}  # Optional 6th parameter to indicate if this is a retry

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
        -mca pml "${OMPI_MCA_pml:-ucx}"
        -mca osc "${OMPI_MCA_osc:-ucx}"
        -x "ROCSHMEM_MAX_NUM_CONTEXTS=$ROCSHMEM_MAX_NUM_CONTEXTS"
        -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"
        -x "ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-$HEAP_SIZE}"
        ${ROCSHMEM_MAX_NUM_HOST_CONTEXTS:+-x "ROCSHMEM_MAX_NUM_HOST_CONTEXTS=$ROCSHMEM_MAX_NUM_HOST_CONTEXTS"}
        ${ROCSHMEM_TEST_USE_DEFAULT_STREAM:+-x "ROCSHMEM_TEST_USE_DEFAULT_STREAM=$ROCSHMEM_TEST_USE_DEFAULT_STREAM"}
        ${ROCSHMEM_TEST_UUID:+-x "ROCSHMEM_TEST_UUID=$ROCSHMEM_TEST_UUID"}
        ${TIMEOUT:+--timeout "$TIMEOUT"}
        ${HOSTFILE:+--hostfile "$HOSTFILE"}
        --map-by numa
      )
  # Construct Test Command
  TEST_LOG_NAME="$TEST_NAME"_n"$NUM_RANKS"_w"$NUM_WG"_z"$NUM_THREADS"
  cmd+=( "$APP" -a "$TEST_NUM" -w "$NUM_WG" -z "$NUM_THREADS" ${NOVERIF:+-noverif} -localbuftype ${LOCALBUFTYPE:-heap} ${ROCSHMEM_TEST_ARGS:-} )
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

  # Determine log file name based on whether this is a retry
  if [ $IS_RETRY -eq 1 ]; then
    LOG_FILE="$LOG_DIR/$TEST_LOG_NAME.retry.log"
    echo "Retry:  $TEST_LOG_NAME"
  else
    LOG_FILE="$LOG_DIR/$TEST_LOG_NAME.log"
    echo "Test:   $TEST_LOG_NAME"
  fi

  # Run Test
  if [ $NUM_GPUS -ge $NUM_RANKS ] || [[ "" != "$HOSTFILE" ]]; then
    echo "# $CMD >> $LOG_FILE" >"$LOG_FILE"
    "${cmd[@]}" >>"$LOG_FILE" 2>&1
  else
    echo "Skip:   $TEST_LOG_NAME ($NUM_RANKS greater than $NUM_GPUS)"
  fi

  # Validate Test
  if [ $? -ne 0 ]
  then
    echo -e "$PRETTY_FAILED: $TEST_LOG_NAME" >&2
    cat "$LOG_FILE"
    DRIVER_RETURN_STATUS=1
    if [ $IS_RETRY -eq 0 ]; then
      # Track failed tests with their parameters for potential retry
      # Capture environment/config state to ensure retry runs under same conditions
      FAILED_LIST="$FAILED_LIST $TEST_LOG_NAME"
      FAILED_TESTS+=("$TEST_NAME|$NUM_RANKS|$NUM_WG|$NUM_THREADS|$MAX_MSG_SIZE|${ROCSHMEM_TEST_USE_DEFAULT_STREAM:-}|${ROCSHMEM_MAX_NUM_CONTEXTS:-}|${NOTIMEOUT:-}|${NOVERIF:-}")
    else
      # Track tests that failed even after retry
      RETRY_FAILED_LIST="$RETRY_FAILED_LIST $TEST_LOG_NAME"
    fi
  else
    # If this was a retry and it passed, remove from failed list
    if [ $IS_RETRY -eq 1 ]; then
      echo -e "$PRETTY_PASSED: $TEST_LOG_NAME (passed on retry)"
      RETRY_PASSED_LIST="$RETRY_PASSED_LIST $TEST_LOG_NAME"
    fi
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

  ExecTest  "defaultctxput"    2       4            128       1024
  ExecTest  "teamctxput"       2       4            128       1024
  ExecTest  "teamctxput"       2       16           256       1024

  ExecTest  "wgput"            2       1            64        1048576
  ExecTest  "wgput"            2       2            64        1048576
  ExecTest  "wgput"            2       16           64        8

  ExecTest  "waveput"          2       1            64        1048576
  ExecTest  "waveput"          2       2            64        1048576
  ExecTest  "waveput"          2       2            128       1048576
  ExecTest  "waveput"          2       16           128       8

  ################################ Non-Blocking ################################
  ExecTest  "p"                2       1            1         128
  ExecTest  "p"                2       1            1024      2
  ExecTest  "p"                2       8            1         32
  ExecTest  "p"                2       16           128       4

  ExecTest  "putnbi"           2       1            1         1048576
  ExecTest  "putnbi"           2       1            1024      512
  ExecTest  "putnbi"           2       8            1         1048576
  ExecTest  "putnbi"           2       16           128       8
  ExecTest  "putnbi"           2       32           256       512
  ExecTest  "putnbi"           2       64           1024      8

  ExecTest  "defaultctxputnbi" 2       4            128       1024
  ExecTest  "teamctxputnbi"    2       4            128       1024
  ExecTest  "teamctxputnbi"    2       16           256       1024

  ExecTest  "wgputnbi"         2       1            64        1048576
  ExecTest  "wgputnbi"         2       2            64        1048576
  ExecTest  "wgputnbi"         2       16           64        8

  ExecTest  "waveputnbi"       2       1            64        1048576
  ExecTest  "waveputnbi"       2       2            64        1048576
  ExecTest  "waveputnbi"       2       2            128       1048576
  ExecTest  "waveputnbi"       2       16           128       8

  ################################ User Buffer Tests ################################
  export LOCALBUFTYPE=host
  ExecTest  "putnbi"           2       32           128       512
  unset LOCALBUFTYPE

  export LOCALBUFTYPE=device
  ExecTest  "putnbi"           2       32           128       512
  unset LOCALBUFTYPE

  export LOCALBUFTYPE=fine
  ExecTest  "putnbi"           2       32           128       512
  unset LOCALBUFTYPE

  export LOCALBUFTYPE=uncached
  ExecTest  "putnbi"           2       32           128       512
  unset LOCALBUFTYPE

  export LOCALBUFTYPE=managed
  ExecTest  "putnbi"           2       32           128       512
  unset LOCALBUFTYPE
}

TestRMAGet() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-120
  ExecTest  "get"              2       1            1         1048576
  ExecTest  "get"              2       1            1024      512
  ExecTest  "get"              2       8            1         1048576
  ExecTest  "get"              2       16           128       8
  ExecTest  "get"              2       32           256       512
  ExecTest  "get"              2       64           1024      8

  ExecTest  "defaultctxget"    2       4            128       1024
  ExecTest  "teamctxget"       2       4            128       1024
  ExecTest  "teamctxget"       2       16           256       1024

  ExecTest  "wgget"            2       1            64        1048576
  ExecTest  "wgget"            2       2            64        1048576
  ExecTest  "wgget"            2       16           64        8

  ExecTest  "waveget"          2       1            64        1048576
  ExecTest  "waveget"          2       2            64        1048576
  ExecTest  "waveget"          2       2            128       1048576
  ExecTest  "waveget"          2       16           128       8

  if [[ $TEST != gda* ]]; then #AIROCSHMEM-162
  ExecTest  "g"                2       1            1         128
  ExecTest  "g"                2       1            1024      1
  ExecTest  "g"                2       8            1         32
  ExecTest  "g"                2       16           128       4
  else echo "Skip:   g_* (AIROCSHMEM-162: GDA _g not implemented)"; fi

  ################################ Non-Blocking ################################
  ExecTest  "getnbi"           2       1            1         1048576
  ExecTest  "getnbi"           2       1            1024      512
  ExecTest  "getnbi"           2       8            1         1048576
  ExecTest  "getnbi"           2       16           128       8
  ExecTest  "getnbi"           2       32           256       512
  ExecTest  "getnbi"           2       64           1024      8

  ExecTest  "defaultctxgetnbi" 2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       4            128       1024
  ExecTest  "teamctxgetnbi"    2       16           256       1024

  ExecTest  "wggetnbi"         2       1            64        1048576
  ExecTest  "wggetnbi"         2       2            64        1048576
  ExecTest  "wggetnbi"         2       16           64        8

  ExecTest  "wavegetnbi"       2       1            64        1048576
  ExecTest  "wavegetnbi"       2       2            64        1048576
  ExecTest  "wavegetnbi"       2       2            128       1048576
  ExecTest  "wavegetnbi"       2       16           128       8
  else echo "Skip:   get_* (AIROCSHMEM-120: RO get tests abort)"; fi

  ################################ User Buffer Tests ################################
  # AIROCSHMEM-120 for RO
  if [[ $TEST != ro* ]]; then
    export LOCALBUFTYPE=host
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=device
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=fine
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=uncached
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE

    export LOCALBUFTYPE=managed
    ExecTest  "getnbi"           2       32           128       512
    unset LOCALBUFTYPE
  fi
}

TestRMA() {
  TestRMAPut
  TestRMAGet
}

TestAMO() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-211
  ExecTest  "amo_add"          2       1            1
  ExecTest  "amo_add"          2       1            1024
  ExecTest  "amo_add"          2       8            1
  ExecTest  "amo_add"          2       32           128

  ExecTest  "amo_fadd"         2       1            1
  ExecTest  "amo_fadd"         2       1            1024
  ExecTest  "amo_fadd"         2       8            1
  ExecTest  "amo_fadd"         2       32           128

  ExecTest  "amo_inc"          2       1            1
  ExecTest  "amo_inc"          2       1            1024
  ExecTest  "amo_inc"          2       8            1
  ExecTest  "amo_inc"          2       32           128

  ExecTest  "amo_finc"         2       1            1
  ExecTest  "amo_finc"         2       1            1024
  ExecTest  "amo_finc"         2       8            1
  ExecTest  "amo_finc"         2       32           128
  else echo "Skip:   amo_add* (AIROCSHMEM-211: ro amo abort)"; fi

  ExecTest  "amo_set"          2       1            1
  ExecTest  "amo_set"          2       8            1
  ExecTest  "amo_set"          2       32           1

  ExecTest  "amo_fetch"        2       1            1
  ExecTest  "amo_fetch"        2       1            1024
  ExecTest  "amo_fetch"        2       8            1
  ExecTest  "amo_fetch"        2       32           128

  ExecTest  "amo_fcswap"       2       1            1
  ExecTest  "amo_fcswap"       2       32           1
  ExecTest  "amo_fcswap"       2       8            1

  ExecTest  "amo_and"          2       1            1

  ExecTest  "amo_fetchand"     2       1            1

  ExecTest  "amo_xor"          2       1            1
}

TestSigOps() {
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
}

TestColl() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "syncall"          2       1            1

  ExecTest  "wavesyncall"      2       1            1

  ExecTest  "wgsyncall"        2       1            1

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

  ExecTest  "alltoall"         2       1            64        512

  ExecTest  "teambroadcast"    2       1            64        32768

  ExecTest  "fcollect"         2       1            64        32768

  ExecTest  "teamreduction"    2       1            64        32768
}

TestOnStream() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "putmem_on_stream" 2       1            1         1048576
  export ROCSHMEM_TEST_USE_DEFAULT_STREAM=1
  ExecTest  "putmem_on_stream" 2       1            1         1048576
  unset ROCSHMEM_TEST_USE_DEFAULT_STREAM

  ExecTest  "getmem_on_stream" 2       1            1         1048576

  ExecTest  "signal_wait_until_on_stream" 2  1      1
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-217
  ExecTest  "putmem_signal_on_stream" 2  1          1         1048576
  else echo "Skip:   putmem_signal_on_stream (AIROCSHMEM-217: RO sometimes abort)"; fi

  ExecTest  "barrier_all_on_stream"  2  1           1
  ExecTest  "quiet_on_stream"        2  1           1
  ExecTest  "sync_all_on_stream"     2  1           1
  ExecTest  "alltoallmem_on_stream"  2  1           64        1048576
  ExecTest  "broadcastmem_on_stream" 2  1           64        1048576
  export ROCSHMEM_MAX_NUM_CONTEXTS=1024
  export ROCSHMEM_MAX_NUM_HOST_CONTEXTS=1024
  ExecTest  "reduce_on_stream"       2  1           64        1048576
  unset ROCSHMEM_MAX_NUM_CONTEXTS
  unset ROCSHMEM_MAX_NUM_HOST_CONTEXTS
}

TestOther() {
  ##############################################################################
  #       | Name             | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################
  ExecTest  "init"             2       1            1
  ExecTest  "library_info"     2       1            1
  ExecTest  "hipmodule_init"   2       1            1
  ExecTest  "device_bitcode"   2       1            1
  ExecTest  "device_bitcode"   2       32           1024
  ExecTest  "device_bitcode"   4       16           256
  ExecTest  "device_bitcode"   8       16           128

  ExecTest  "pingpong"         2       1            1
  ExecTest  "pingpong"         2       8            1
  ExecTest  "pingpong"         2       32           1

  ExecTest  "pingall"          2       1            1
  ExecTest  "pingall"          2       8            1
  ExecTest  "pingall"          2       32           1

  ################################ Flood test ##################################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-324
  ExecTest  "flood_put"        2       64           1024
  ExecTest  "flood_put"        8       64           1024
  ExecTest  "flood_putnbi"     8       64           1024
  ExecTest  "flood_p"          8       64           1024

  ExecTest  "flood_get"        2       64           1024
  ExecTest  "flood_get"        8       64           1024
  ExecTest  "flood_getnbi"     8       64           1024
  if [[ $TEST != gda* ]]; then #AIROCSHMEM-162
  ExecTest  "flood_g"          8       64           1024
  else echo "Skip:   flood_g (AIROCSHMEM-162: GDA _g not implemented)"; fi

  ExecTest  "flood_add"        2       64           1024
  ExecTest  "flood_add"        8       64           1024
  ExecTest  "flood_fadd"       8       64           1024
  ExecTest  "flood_waitadd"    8       64           1024
  else echo "Skip:   flood_* (AIROCSHMEM-324: RO flood tests fail in UCX)"; fi

  # This test requires more contexts than workgroups
  export ROCSHMEM_MAX_NUM_CONTEXTS=1024
  ExecTest  "teamctxinfra"        2       1            1
  ExecTest  "teamctxsingleinfra"  2       1            1
  ExecTest  "teamctxblockinfra"   4       1            1
  ExecTest  "teamctxblockinfra"   5       1            1
  ExecTest  "teamctxoddeveninfra" 4       1            1
  ExecTest  "teamctxoddeveninfra" 5       1            1
  ExecTest  "teamctxsharedinfra"  2       1            1
  ExecTest  "teamctxsharedinfra"  5       1            1
  ExecTest  "teamctxsubsetparentinfra" 4  1            1
  ExecTest  "teamctxsubsetparentinfra" 5  1            1
  export ROCSHMEM_MAX_NUM_HOST_CONTEXTS=1024
  ExecTest  "host_ctx_create"          2       1            1
  unset ROCSHMEM_MAX_NUM_CONTEXTS
  unset ROCSHMEM_MAX_NUM_HOST_CONTEXTS

  ExecTest  "shmemptr"         2       1            1         8
  ExecTest  "shmemptr"         2       1            1024      8
  ExecTest  "shmemptr"         2       8            1         8
  ExecTest  "shmemptr"         2       16           128       8

  ########################### Fence ordering tests #############################
  if [[ $TEST != ro* ]]; then #AIROCSHMEM-418: fence tests not supported on RO
  ExecTest  "fence_putwavesignal"    2       1     64        1048576
  ExecTest  "fence_putwavesignal"    2       8     256       1048576
  ExecTest  "fence_putwavesignal"    2       32    1024      65536
  ExecTest  "fence_putlargesmall"    2       1     64        4096
  ExecTest  "fence_putlargesmall"    2       8     256       65536
  ExecTest  "fence_fanout"           2       1     64        1048576
  ExecTest  "fence_fanout"           4       4     256       65536
  ExecTest  "fence_fanout"           8       8     256       65536
  ExecTest  "fence_putwavenbichunks" 2       1     64        1048576
  ExecTest  "fence_putwavenbichunks" 2       8     256       65536
  else echo "Skip:   fence_* (AIROCSHMEM-418: fence tests not supported on RO)"; fi
}

TestTiles() {
  ##############################################################################
  #       | Name                      | Ranks | Workgroups | Threads | Max Message Size #
  ##############################################################################

  # Detect wavefront size based on GPU architecture
  # gfx1100 and gfx1201 have wavefront size 32, most others have 64
  WAVE_SIZE=64
  if command -v rocminfo >/dev/null 2>&1; then
    if rocminfo | grep -qE "Name:.*(gfx1100|gfx1201)"; then
      WAVE_SIZE=32
    fi
  fi

  ExecTest  "tile_put_contiguous"       2       1            1
  ExecTest  "tile_put_rowmajor"         2       1            1
  ExecTest  "tile_put_colmajor"         2       1            1
  ExecTest  "tile_put_arbitrary"        2       1            1
  ExecTest  "tile_put_wave_contiguous"  2       1            $WAVE_SIZE
  ExecTest  "tile_put_wg_contiguous"    2       1            $((WAVE_SIZE * 16))
  ExecTest  "tile_put_wg_contiguous"    2       4            $((WAVE_SIZE * 16))
  ExecTest  "tile_get_contiguous"       2       1            1
  ExecTest  "tile_get_rowmajor"         2       1            1
  ExecTest  "tile_get_colmajor"         2       1            1
  ExecTest  "tile_get_arbitrary"        2       1            1
  ExecTest  "tile_get_wg_contiguous"    2       1            $((WAVE_SIZE * 16))
  ExecTest  "tile_get_wg_contiguous"    2       4            $((WAVE_SIZE * 16))
  ExecTest  "tile_put_1d"               2       1            1
  ExecTest  "tile_get_1d"               2       1            1
  ExecTest  "tile_get_wave_contiguous"  2       1            $WAVE_SIZE
}

TestHeatMapRMA() {
  NOTIMEOUT=1
  NOVERIF=1
  # Batch rotation allocates volume*batch*2 = 20 GiB; use 22 GiB to leave headroom
  ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-$((22*1024*1024*1024))}
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
    echo "Usage: ${0} <executable> <test_suite | test_name | test_config> <log_dir> [hostfile]"
    echo
    echo "    <executable>  : path to the tester executable"
    echo "    <test_suite>  : test suite to run, e.g. 'all', 'rma', or 'put'"
    echo "    <test_name>   : name of test to run, e.g. 'putnbi' or 'amo_fadd'"
    echo "    <test_config> : quoted test configuration to run, in the format"
    echo '                    "<test_name> <ranks> <workgroups> <threads> [max_msg_size]"'
    echo '                    e.g. "putnbi 2 8 1024 65536" or "amo_fadd 2 1 64"'
    echo "        <ranks>        : number of PEs/ranks to use for test"
    echo "        <workgroups>   : number of workgroups per PE"
    echo "        <threads>      : number of threads per workgroup"
    echo "        [max_msg_size] : maximum message size to test"
    echo "    <log_dir>     : path to output log directory"
    echo "    [hostfile]    : path to hostfile"
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

RerunFailedTests() {
  if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    return
  fi

  echo ""
  echo "========================================================================"
  echo "Rerunning ${#FAILED_TESTS[@]} failed test(s)..."
  echo "========================================================================"
  echo ""

  # Clear the driver return status for retry
  DRIVER_RETURN_STATUS=0
  RETRY_FAILED_LIST=""
  RETRY_PASSED_LIST=""

  # Rerun each failed test with the same environment/config state
  for test_params in "${FAILED_TESTS[@]}"; do
    IFS='|' read -r test_name num_ranks num_wg num_threads max_msg_size use_default_stream max_contexts notimeout noverif <<< "$test_params"

    # Restore environment state from original test run
    if [[ -n "$use_default_stream" ]]; then
      export ROCSHMEM_TEST_USE_DEFAULT_STREAM="$use_default_stream"
    fi
    if [[ -n "$max_contexts" ]]; then
      export ROCSHMEM_MAX_NUM_CONTEXTS="$max_contexts"
    fi
    if [[ -n "$notimeout" ]]; then
      NOTIMEOUT="$notimeout"
    fi
    if [[ -n "$noverif" ]]; then
      NOVERIF="$noverif"
    fi

    ExecTest "$test_name" "$num_ranks" "$num_wg" "$num_threads" "$max_msg_size" 1

    # Clean up environment state after retry
    unset ROCSHMEM_TEST_USE_DEFAULT_STREAM
    unset ROCSHMEM_MAX_NUM_CONTEXTS
    unset NOTIMEOUT
    unset NOVERIF
  done

  echo ""
  echo "========================================================================"
  echo "Retry Summary"
  echo "========================================================================"

  if [[ -n "$RETRY_PASSED_LIST" ]]; then
    echo -e "$PRETTY_PASSED on retry:$RETRY_PASSED_LIST"
  fi

  if [[ -n "$RETRY_FAILED_LIST" ]]; then
    echo -e "$PRETTY_FAILED even after retry:$RETRY_FAILED_LIST"
  fi

  echo ""
}

# Parse optional flags before positional arguments
ARTIFACT_DIR=""
_POSITIONAL=()
for _arg in "$@"; do
  if [[ "$_arg" == --artifact-dir=* ]]; then
    ARTIFACT_DIR="${_arg#--artifact-dir=}"
  elif [[ "$_ARG_NEXT_IS_ARTIFACT_DIR" == "1" ]]; then
    ARTIFACT_DIR="$_arg"
    _ARG_NEXT_IS_ARTIFACT_DIR=0
  elif [[ "$_arg" == "--artifact-dir" ]]; then
    _ARG_NEXT_IS_ARTIFACT_DIR=1
  else
    _POSITIONAL+=("$_arg")
  fi
done

APP=${_POSITIONAL[0]:-}
TEST=${_POSITIONAL[1]:-}
LOG_DIR=${_POSITIONAL[2]:-}
HOSTFILE=${_POSITIONAL[3]:-}

DRIVER_RETURN_STATUS=0
FAILED_TESTS=()  # Array to store failed test parameters
RETRY_THRESHOLD=${RETRY_THRESHOLD:-5}  # Maximum number of failed tests to retry (can be overridden via env var)

ValidateInput ${#_POSITIONAL[@]}
ValidateLogDir $LOG_DIR

# Print build info and environment variables before running tests
ROCSHMEM_INFO="$(dirname "$APP")/rocshmem_info"
if [ ! -x "$ROCSHMEM_INFO" ]; then
  # builddir case
  ROCSHMEM_INFO="$(dirname "$APP")/../../tools/rocshmem_info"
fi
if [ -x "$ROCSHMEM_INFO" ]; then
  "$ROCSHMEM_INFO"
fi

case $TEST in
  "heatmaprma")
    TestHeatMapRMA
    ;;
  "heatmapcoll")
    TestHeatMapColl
    ;;
  "heatmap")
    TestHeatMapRMA
    TestHeatMapColl
    ;;
  "all"|"gda"|"gda-mlx5"|"gda-bnxt"|"gda-ionic"|"ro"|"all-ro")
    TEST=${TEST#all-} #convert all-ro used in CI scripts into simple ro prefix
    TestRMA
    TestAMO
    TestSigOps
    TestColl
    TestOther
    TestOnStream
    # Tile tests are only supported on IPC backend
    if [[ ! "$TEST" =~ ^(gda|ro) ]]; then
      TestTiles
    fi
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
  *"stream")
    TestOnStream
    ;;
  *"other")
    TestOther
    ;;
  *"tiles")
    TestTiles
    ;;
  *)
    #######################################################################################
    #        |   Name   |   Ranks   |   Workgroups   |   Threads   |   Max Message Size   #
    #######################################################################################
    # Allow passing in a test config as "<test_name> <ranks> <workgroups> <threads> [max_msg_size]"
    # e.g. "putnbi 2 8 1024 65536" or "amo_fadd 2 1 64"
    TEST_OPTS=($TEST)
    NAME=${TEST_OPTS[0]}
    if [ ${#TEST_OPTS[@]} -eq 4 ] || [ ${#TEST_OPTS[@]} -eq 5 ]; then
      RANKS=${TEST_OPTS[1]}
      WORKGROUPS=${TEST_OPTS[2]}
      THREADS=${TEST_OPTS[3]}
      MAX_MESSAGE_SIZE=${TEST_OPTS[4]}
    else
      RANKS=2
      WORKGROUPS=1
      THREADS=1
      MAX_MESSAGE_SIZE=8
    fi
    ExecTest  "${NAME}"  "${RANKS}"  "${WORKGROUPS}"  "${THREADS}"  "${MAX_MESSAGE_SIZE}"
    ;;
esac

EXIT_STATUS=$DRIVER_RETURN_STATUS

# If there were failures, try to rerun them once (only if below threshold)
if [ $EXIT_STATUS -ne 0 ] && [ ${#FAILED_TESTS[@]} -gt 0 ]; then
  if [ ${#FAILED_TESTS[@]} -le $RETRY_THRESHOLD ]; then
    RerunFailedTests
    EXIT_STATUS=$DRIVER_RETURN_STATUS
  else
    echo ""
    echo "========================================================================"
    echo "Too many test failures (${#FAILED_TESTS[@]} > $RETRY_THRESHOLD threshold)"
    echo "Skipping retry - this may indicate a systemic issue"
    echo "========================================================================"
    echo ""
  fi
fi

# Final summary
echo ""
echo "========================================================================"
if [ $EXIT_STATUS -eq 0 ]; then
  echo -e "TESTS PASSED"
else
  if [[ -n "$RETRY_FAILED_LIST" ]]; then
    echo -e "TESTS FAILED (even after retry): $RETRY_FAILED_LIST"
  else
    echo -e "TESTS FAILED: $FAILED_LIST"
  fi
fi
echo "========================================================================"

# Generate performance artifact if requested
# --artifact-dir works for any test suite but is most useful with heatmap suites.
# For a single build (no baseline), perf_compare.py is called via run_perf_compare.sh
# using --skip-build; the caller must supply --baseline-dir and --branch-dir via the
# PERF_BASELINE_DIR and PERF_BRANCH_DIR environment variables if a comparison is wanted.
# When only ARTIFACT_DIR is set (no PERF_BASELINE_DIR), per-test plots are still generated.
if [[ -n "$ARTIFACT_DIR" && $EXIT_STATUS -eq 0 ]]; then
  case "$TEST" in
    heatmap|heatmaprma|heatmapcoll)
      echo ""
      echo "========================================================================"
      echo "Generating performance artifact -> $ARTIFACT_DIR"
      echo "========================================================================"
      SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
      RUN_COMPARE="$SCRIPT_DIR/run_perf_compare.sh"

      mkdir -p "$ARTIFACT_DIR"

      if [[ -n "${PERF_BASELINE_DIR:-}" && -n "${PERF_BRANCH_DIR:-}" ]]; then
        # Full comparison: caller supplied pre-built baseline and branch directories
        bash "$RUN_COMPARE" \
          --skip-build \
          --suite "$TEST" \
          --iterations 1 \
          --baseline-dir "$PERF_BASELINE_DIR" \
          --branch-dir   "$PERF_BRANCH_DIR" \
          --outdir "$ARTIFACT_DIR"
        echo ""
        echo "CI ARTIFACT: $ARTIFACT_DIR/heatmap_summary.png"
        echo "CI ARTIFACT: $ARTIFACT_DIR/heatmap_summary.txt"
        echo "CI ARTIFACT: $ARTIFACT_DIR/heatmap_data.csv"
        echo "CI ARTIFACT: $ARTIFACT_DIR/per_test/"
      else
        # Single-build: generate per-test latency plots only (no baseline to compare against)
        COMPARE="$SCRIPT_DIR/perf_compare.py"
        VENV_DIR="$SCRIPT_DIR/.perf-venv"
        # Self-healing: recreate venv if dependencies are missing
        if [[ -d "$VENV_DIR" ]] && \
           ! "$VENV_DIR/bin/python3" -c "import matplotlib, numpy, pandas, seaborn" &>/dev/null; then
          rm -rf "$VENV_DIR"
        fi
        if [[ ! -d "$VENV_DIR" ]]; then
          if python3 -m ensurepip --version &>/dev/null; then
            python3 -m venv "$VENV_DIR" || { VENV_DIR=""; }
          else
            python3 -m venv --system-site-packages --without-pip "$VENV_DIR" || { VENV_DIR=""; }
          fi
          [[ -x "$VENV_DIR/bin/python3" ]] && \
            "$VENV_DIR/bin/python3" -m pip install --quiet matplotlib numpy pandas seaborn || true
        fi
        PYTHON="${VENV_DIR:+$VENV_DIR/bin/}python3"
        # Use the single run log dir as both baseline and variant so per-test plots are produced.
        # The heatmap will show 0% (identical), which is expected for a single-build run.
        "$PYTHON" "$COMPARE" \
          --baseline "$LOG_DIR" \
          --variants "current:$LOG_DIR" \
          --outdir "$ARTIFACT_DIR" || true
        echo ""
        echo "CI ARTIFACT: $ARTIFACT_DIR/per_test/"
        echo "NOTE: Set PERF_BASELINE_DIR and PERF_BRANCH_DIR for a full heatmap comparison"
      fi
      ;;
  esac
elif [[ -n "$ARTIFACT_DIR" && $EXIT_STATUS -ne 0 ]]; then
  echo ""
  echo "WARNING: Skipping artifact generation due to test failures" >&2
fi

exit $EXIT_STATUS
