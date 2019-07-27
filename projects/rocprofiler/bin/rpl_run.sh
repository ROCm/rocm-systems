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

time_stamp=`date +%y%m%d_%H%M%S`
BIN_DIR=$(dirname $(realpath $0))
PKG_DIR=$(dirname $BIN_DIR)
ROOT_DIR=$(dirname $PKG_DIR)
TT_DIR=$ROOT_DIR/roctracer
RUN_DIR=`pwd`
TMP_DIR="/tmp"
DATA_DIR="rpl_data_${time_stamp}_$$"

RPL_PATH=$PKG_DIR/lib
TLIB_PATH=$PKG_DIR/tool
TTLIB_PATH=$TT_DIR/tool

# Default HIP path
if [ -z "$HIP_PATH" ] ; then
  export HIP_PATH=/opt/rocm/hip
fi
# Default HCC path
if [ -z "$HCC_HOME" ] ; then
  export HCC_HOME=/opt/rocm/hcc
fi

# runtime API trace
HSA_TRACE=0
SYS_TRACE=0
HIP_TRACE=0

# Generate stats
GEN_STATS=0

export PATH=.:$PATH

# enable error logging
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export ROCPROFILER_LOG=1
unset ROCPROFILER_SESS

# ROC Profiler environment
# Loading of ROC Profiler by HSA runtime
export HSA_TOOLS_LIB=$RPL_PATH/librocprofiler64.so
# Loading of the test tool by ROC Profiler
export ROCP_TOOL_LIB=$TLIB_PATH/libtool.so
# Enabling HSA dispatches intercepting by ROC PRofiler
export ROCP_HSA_INTERCEPT=1
# Disabling internal ROC Profiler proxy queue (simple version supported for testing purposes)
unset ROCP_PROXY_QUEUE
# ROC Profiler metrics definition
export ROCP_METRICS=$PKG_DIR/lib/metrics.xml
# Disable AQL-profile read API
export AQLPROFILE_READ_API=0
# ROC Profiler package path
export ROCP_PACKAGE_DIR=$PKG_DIR

# error handling
fatal() {
  echo "$0: Error: $1"
  echo ""
  usage
}

error() {
  echo "$0: Error: $1"
  echo ""
  exit 1
}

# usage method
usage() {
  bin_name=`basename $0`
  echo "ROCm Profiling Library (RPL) run script, a part of ROCprofiler library package."
  echo "Full path: $BIN_DIR/$bin_name"
  echo "Metrics definition: $PKG_DIR/lib/metrics.xml"
  echo ""
  echo "Usage:"
  echo "  $bin_name [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>"
  echo ""
  echo "Options:"
  echo "  -h - this help"
  echo "  --verbose - verbose mode, dumping all base counters used in the input metrics"
  echo "  --list-basic - to print the list of basic HW counters"
  echo "  --list-derived - to print the list of derived metrics with formulas"
  echo ""
  echo "  -i <.txt|.xml file> - input file"
  echo "      Input file .txt format, automatically rerun application for every pmc line:"
  echo ""
  echo "        # Perf counters group 1"
  echo "        pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts VALUUtilization FetchSize"
  echo "        # Perf counters group 2"
  echo "        pmc : WriteSize L2CacheHit"
  echo "        # Filter by dispatches range, GPU index and kernel names"
  echo "        # supported range formats: \"3:9\", \"3:\", \"3\""
  echo "        range: 1 : 4"
  echo "        gpu: 0 1 2 3"
  echo "        kernel: simple Pass1 simpleConvolutionPass2"
  echo ""
  echo "      Input file .xml format, for single profiling run:"
  echo ""
  echo "        # Metrics list definition, also the form \"<block-name>:<event-id>\" can be used"
  echo "        # All defined metrics can be found in the 'metrics.xml'"
  echo "        # There are basic metrics for raw HW counters and high-level metrics for derived counters"
  echo "        <metric name=SQ:4,SQ_WAVES,VFetchInsts"
  echo "        ></metric>"
  echo ""
  echo "        # Filter by dispatches range, GPU index and kernel names"
  echo "        <metric"
  echo "          # range formats: \"3:9\", \"3:\", \"3\""
  echo "          range=\"\""
  echo "          # list of gpu indexes \"0,1,2,3\""
  echo "          gpu_index=\"\""
  echo "          # list of matched sub-strings \"Simple1,Conv1,SimpleConvolution\""
  echo "          kernel=\"\""
  echo "        ></metric>"
  echo ""
  echo "  -o <output file> - output CSV file [<input file base>.csv]"
  echo "  -d <data directory> - directory where profiler store profiling data including traces [/tmp]"
  echo "      The data directory is renoving autonatically if the directory is matching the temporary one, which is the default."
  echo "  -t <temporary directory> - to change the temporary directory [/tmp]"
  echo "      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory."
  echo ""
  echo "  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]"
  echo "  --timestamp <on|off> - to turn on/off the kernel disoatches timestamps, dispatch/begin/end/complete [off]"
  echo "  --ctx-wait <on|off> - to wait for outstanding contexts on profiler exit [on]"
  echo "  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]"
  echo "  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]"
  echo ""
  echo "  --stats - generating kernel execution stats, file <output name>.stats.csv"
  echo "  --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible"
  echo "  --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible"
  echo "  --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible"
  echo "    Generated files: <output name>.hsa_stats.txt <output name>.json"
  echo "    Traced API list can be set by input .txt or .xml files."
  echo "    Input .txt:"
  echo "      hsa: hsa_queue_create hsa_amd_memory_pool_allocate"
  echo "    Input .xml:"
  echo "      <trace name=\"HSA\">"
  echo "        <parameters list=\"hsa_queue_create, hsa_amd_memory_pool_allocate\">"
  echo "        </parameters>"
  echo "      </trace>"
  echo ""
  echo "Configuration file:"
  echo "  You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:${HOME}:<package path>"
  echo "  First the configuration file is looking in the current directory, then in your home, and then in the package directory."
  echo "  Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat'."
  echo "  An example of 'rpl_rc.xml':"
  echo "    <defaults"
  echo "      basenames=off"
  echo "      timestamp=off"
  echo "      ctx-limit=0"
  echo "      heartbeat=0"
  echo "    ></defaults>"
  echo ""
  exit 1
}

# profiling run method
OUTPUT_LIST=""
run() {
  export ROCP_INPUT="$1"
  OUTPUT_DIR="$2"
  shift
  shift
  APP_CMD=$*

  if [ "$OUTPUT_DIR" = "-" ] ; then
    input_tag=`echo $ROCP_INPUT | sed "s/\.xml//"`
    export ROCP_OUTPUT_DIR=${input_tag}_results_${time_stamp}
  elif [ "$OUTPUT_DIR" = "--" ] ; then
    unset ROCP_OUTPUT_DIR
  else
    export ROCP_OUTPUT_DIR=$OUTPUT_DIR
  fi
  echo "RPL: result dir '$ROCP_OUTPUT_DIR'"

  if [ ! -e "$ROCP_INPUT" ] ; then
    error "Input file '$ROCP_INPUT' not found"
  fi

  if [ -n "$ROCP_OUTPUT_DIR" ] ; then
    if [ "$OUTPUT_DIR" = "-" ] ; then
      if [ -e "$ROCP_OUTPUT_DIR" ] ; then
        error "generated dir '$ROCP_OUTPUT_DIR' exists"
      fi
    fi
    mkdir -p "$ROCP_OUTPUT_DIR"

    OUTPUT_LIST="$OUTPUT_LIST $ROCP_OUTPUT_DIR/results.txt"
  fi

  API_TRACE=""
  if [ "$HSA_TRACE" = 1 ] ; then
    API_TRACE="hsa"
  fi
  if [ "$SYS_TRACE" = 1 ] ; then
    API_TRACE="sys"
  fi
  if [ "$HIP_TRACE" = 1 ] ; then
    if [ -z "$API_TRACE" ] ; then
      API_TRACE="hip";
    else
      API_TRACE="all"
    fi
  fi
  if [ -n "$API_TRACE" ] ; then
    API_TRACE=$(echo $API_TRACE | sed 's/all//')
    if [ -n "$API_TRACE" ] ; then export ROCTRACER_DOMAIN=$API_TRACE; fi
    if [ "$API_TRACE" = "hip" -o "$API_TRACE" = "sys" ] ; then
      OUTPUT_LIST="$ROCP_OUTPUT_DIR/"
    fi
    export HSA_TOOLS_LIB="$TTLIB_PATH/libtracer_tool.so"
  fi

  redirection_cmd=""
  if [ -n "$ROCP_OUTPUT_DIR" ] ; then
    redirection_cmd="2>&1 | tee $ROCP_OUTPUT_DIR/log.txt"
  fi

  #unset ROCP_OUTPUT_DIR
  CMD_LINE="$APP_CMD $redirection_cmd"
  eval "$CMD_LINE"
}

# main
echo "RPL: on '$time_stamp' from '$PKG_DIR' in '$RUN_DIR'"
# Parsing arguments
if [ -z "$1" ] ; then
  usage
fi

INPUT_FILE=""
DATA_PATH="-"
OUTPUT_DIR="-"
output=""
csv_output=""

ARG_IN=""
while [ 1 ] ; do
  ARG_IN=$1
  ARG_VAL=1
  if [ "$1" = "-h" ] ; then
    usage
  elif [ "$1" = "-i" ] ; then
    INPUT_FILE="$2"
  elif [ "$1" = "-o" ] ; then
    output="$2"
  elif [ "$1" = "-d" ] ; then
    DATA_PATH=$2
  elif [ "$1" = "-t" ] ; then
    TMP_DIR="$2"
    if [ "$OUTPUT_DIR" = "-" ] ; then
      DATA_PATH=$TMP_DIR
    fi
  elif [ "$1" = "--list-basic" ] ; then
    export ROCP_INFO=b
    eval "$PKG_DIR/tool/ctrl"
    exit 1
  elif [ "$1" = "--list-derived" ] ; then
    export ROCP_INFO=d
    eval "$PKG_DIR/tool/ctrl"
    exit 1
  elif [ "$1" = "--basenames" ] ; then
    if [ "$2" = "on" ] ; then
      export ROCP_TRUNCATE_NAMES=1
    else
      export ROCP_TRUNCATE_NAMES=0
    fi
  elif [ "$1" = "--timestamp" ] ; then
    if [ "$2" = "on" ] ; then
      export ROCP_TIMESTAMP_ON=1
    else
      export ROCP_TIMESTAMP_ON=0
    fi
  elif [ "$1" = "--ctx-wait" ] ; then
    if [ "$2" = "on" ] ; then
      export ROCP_OUTSTANDING_WAIT=1
    else
      export ROCP_OUTSTANDING_WAIT=0
    fi
  elif [ "$1" = "--ctx-limit" ] ; then
    export ROCP_OUTSTANDING_MAX="$2"
  elif [ "$1" = "--heartbeat" ] ; then
    export ROCP_OUTSTANDING_MON="$2"
  elif [ "$1" = "--stats" ] ; then
    ARG_VAL=0
    export ROCP_TIMESTAMP_ON=1
    GEN_STATS=1
  elif [ "$1" = "--hsa-trace" ] ; then
    ARG_VAL=0
    export ROCP_TIMESTAMP_ON=1
    GEN_STATS=1
    HSA_TRACE=1
  elif [ "$1" = "--sys-trace" ] ; then
    ARG_VAL=0
    export ROCP_TIMESTAMP_ON=1
    GEN_STATS=1
    SYS_TRACE=1
  elif [ "$1" = "--hip-trace" ] ; then
    ARG_VAL=0
    export ROCP_TIMESTAMP_ON=1
    GEN_STATS=1
    HIP_TRACE=1
  elif [ "$1" = "--verbose" ] ; then
    ARG_VAL=0
    export ROCP_VERBOSE_MODE=1
  else
    break
  fi
  shift
  if [ "$ARG_VAL" = 1 ] ; then shift; fi
done

ARG_CK=`echo $ARG_IN | sed "s/^-.*$/-/"`
if [ "$ARG_CK" = "-" ] ; then
  fatal "Wrong option '$ARG_IN'"
fi

if [ -z "$INPUT_FILE" ] ; then
  input_base="results"
  input_type="none"
else
  input_base=`echo "$INPUT_FILE" | sed "s/^\(.*\)\.\([^\.]*\)$/\1/"`
  input_type=`echo "$INPUT_FILE" | sed "s/^\(.*\)\.\([^\.]*\)$/\2/"`
  if [ -z "${input_base}" -o -z "${input_type}" ] ; then
    fatal "Bad input file '$INPUT_FILE'"
  fi
  input_base=`basename $input_base`
fi

if [ "$DATA_PATH" = "-" ] ; then
  DATA_PATH=$TMP_DIR
fi

if [ -n "$output" ] ; then
  if [ "$output" = "--" ] ; then
    OUTPUT_DIR="--"
  else
    csv_output=$output
  fi
else
  csv_output=$RUN_DIR/${input_base}.csv
fi

APP_CMD=$*

echo "RPL: profiling '$APP_CMD'"
echo "RPL: input file '$INPUT_FILE'"

input_list=""
RES_DIR=""
if [ "$input_type" = "xml" ] ; then
  OUTPUT_DIR=$DATA_PATH
  input_list=$INPUT_FILE
elif [ "$input_type" = "txt" -o "$input_type" = "none" ] ; then
  RES_DIR=$DATA_PATH/$DATA_DIR
  if [ -e $RES_DIR ] ; then
    error "Rundir '$RES_DIR' exists"
  fi
  mkdir -p $RES_DIR
  echo "RPL: output dir '$RES_DIR'"
  if [ "$input_type" = "txt" ] ; then
    $BIN_DIR/txt2xml.sh $INPUT_FILE $RES_DIR
  else
    echo "<metric></metric>" > $RES_DIR/input.xml
  fi
  input_list=`/bin/ls $RES_DIR/input*.xml`
  export ROCPROFILER_SESS=$RES_DIR
else
  fatal "Bad input file type '$INPUT_FILE'"
fi

if [ -n "$csv_output" ] ; then
  rm -f $csv_output
fi

for name in $input_list; do
  run $name $OUTPUT_DIR $APP_CMD
  if [ -n "$ROCPROFILER_SESS" -a -e "$ROCPROFILER_SESS/error" ] ; then
    echo "Error found, profiling aborted."
    csv_output=""
    break
  fi
done

if [ -n "$csv_output" ] ; then
  if [ "$GEN_STATS" = "1" ] ; then
    db_output=$(echo $csv_output | sed "s/\.csv/.db/")
    python $BIN_DIR/tblextr.py $db_output $OUTPUT_LIST
  else
    python $BIN_DIR/tblextr.py $csv_output $OUTPUT_LIST
  fi
  if [ "$?" -eq 0 ] ; then
    echo "RPL: '$csv_output' is generated"
  else
    echo "Data extracting error: $OUTPUT_LIST'"
  fi
fi

if [ "$DATA_PATH" = "$TMP_DIR" ] ; then
  if [ -e "$RES_DIR" ] ; then
    rm -rf $RES_DIR
  fi
fi

exit 0
