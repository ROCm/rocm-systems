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

ROCPROF_ARGS="$*"
time_stamp=`date +%y%m%d_%H%M%S`
BIN_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))
ROOT_DIR=$(dirname $BIN_DIR)
RUN_DIR=`pwd`
TMP_DIR="/tmp"
DATA_DIR="rpl_data_${time_stamp}_$$"

RPL_PATH=$ROOT_DIR/lib
TLIB_PATH=$RPL_PATH/rocprofiler
TTLIB_PATH=$ROOT_DIR/lib/roctracer
ROCM_LIB_PATH=$ROOT_DIR/lib
PROF_BIN_DIR=$ROOT_DIR/libexec/rocprofiler

# Define color code
GREEN='\033[0;32m'
GREY='\033[0;90m'
RED='\033[0;31m'
RESET='\033[0m'

# Function to display deprecation warning message
display_warning() {
  echo -e "\n${RED}WARNING: We are phasing out development and support for roctracer/rocprofiler/rocprof/rocprofv2 in favor of \
rocprofiler-sdk/rocprofv3 in upcoming ROCm releases. Going forward, only critical defect fixes will be addressed for \
older versions of profiling tools and libraries. We encourage all users to upgrade to the latest version, \
rocprofiler-sdk library and rocprofv3 tool, to ensure continued support and access to new features.${RESET}\n"
}

# Display the warning message
display_warning

# check if rocprof is supportd on this gpu arch
V1_SUPPORTED_GPU_ARCHS=("gfx80x","gfx90x","gfx94x")
IS_SUPPORTED="false"
if [ -f $BIN_DIR/rocm_agent_enumerator ]; then
  CURRENT_AGENTS_LIST=$($BIN_DIR/rocm_agent_enumerator)
else
  IS_SUPPORTED="false"
  echo -e "Warning: Missing rocm_agent_enumerator binary"
  exit 0
fi

if [ -z "$ROCP_PYTHON_VERSION" ] ; then
  ROCP_PYTHON_VERSION=python3
fi

# runtime API trace
ROCTX_TRACE=0
HSA_TRACE=0
SYS_TRACE=0
HIP_TRACE=0

# Generate stats
GEN_STATS=0

# Quoting profiled cmd line
CMD_QTS=1

export PATH=.:$PATH

# enable error logging
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export ROCPROFILER_LOG=1
unset ROCPROFILER_SESS

# Profiler environment
# Loading of profiler library by HSA runtime
MY_HSA_TOOLS_LIB="$RPL_PATH/librocprofiler64.so.1"

# Tool internal Preloads
ROCPROFV1_LD_PRELOAD=$MY_HSA_TOOLS_LIB
ROCPROFV1_TOOL_PRELOAD=$MY_HSA_TOOLS_LIB

# Loading of the test tool by ROC Profiler
export ROCP_TOOL_LIB=$TLIB_PATH/librocprof-tool.so
# ROC Profiler metrics definition
export ROCP_METRICS=$TLIB_PATH/metrics.xml

# Enabling HSA dispatches intercepting by ROC PRofiler
export ROCP_HSA_INTERCEPT=1
# Disabling internal ROC Profiler proxy queue (simple version supported for testing purposes)
unset ROCP_PROXY_QUEUE
# Disable AQL-profile read API
export AQLPROFILE_READ_API=0
# ROC Profiler package path
export ROCP_PACKAGE_DIR=$ROOT_DIR
# enabled SPM KFD mode
export ROCP_SPM_KFD_MODE=1

# error handling
fatal() {
  echo "$0: Error: $1"
  echo ""
  usage
  exit 1
}

error() {
  echo "$0: Error: $1"
  echo ""
  exit 1
}

error_message=""
errck() {
  if [ -n "$error_message" ]; then
    fatal "$1 : $error_message"
  fi
}

# usage method
usage() {
  bin_name=`basename $0`
  echo "ROCm Profiling Library (RPL) run script, a part of ROCprofiler library package."
  echo "Full path: $BIN_DIR/$bin_name"
  echo "Metrics definition: $TLIB_PATH/metrics.xml"
  echo ""
  echo "Usage:"
  echo "  $bin_name [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>"
  echo ""
  echo "Options:"
  echo "  -h - this help"
  echo "  --tool-version <1|2> - to use specific version of rocprof tool, by default v1 is used"
  echo "            1 - rocprofiler tool v1"
  echo "            2 - rocprofiler tool v2"
  echo "  --verbose - verbose mode, dumping all base counters used in the input metrics"
  echo "  --list-basic - to print the list of basic HW counters"
  echo "  --list-derived - to print the list of derived metrics with formulas"
  echo "  --cmd-qts <on|off> - quoting profiled cmd line [on]"
  echo ""
  echo "  -i <.txt|.xml file> - input file"
  echo "      Input file .txt format, automatically rerun application for every profiling features line:"
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
  echo "      The data directory is automatically removed if it is matching the default temporary directory."
  echo "  -t <temporary directory> - to change the temporary directory [/tmp]"
  echo "      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory."
  echo "  -m <metric file> - file defining custom metrics to use in-place of defaults."
  echo ""
  echo "  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]"
  echo "  --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete during kernel profiling [off]"
  echo "  --ctx-wait <on|off> - to wait for outstanding contexts on profiler exit [on]"
  echo "  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]"
  echo "  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]"
  echo "  --obj-tracking <on|off> - to turn on/off kernels code objects tracking [on]"
  echo "    To support V3 code object"
  echo ""
  echo "  --stats - generating kernel execution stats, file <output name>.stats.csv"
  echo ""
  echo "  --roctx-trace - to enable rocTX application code annotation trace, \"Markers and Ranges\" JSON trace section."
  echo "  --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible"
  echo "  --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible"
  echo "  --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible"
  echo "    '--hsa-trace' can be used in addition to select activity tracing from HSA (ROCr runtime) level"
  echo "    Generated files: <output name>.<domain>_stats.txt <output name>.json"
  echo "    Traced API list can be set by input .txt or .xml files."
  echo "    Input .txt:"
  echo "      hsa: hsa_queue_create hsa_amd_memory_pool_allocate"
  echo "    Input .xml:"
  echo "      <trace name=\"HSA\">"
  echo "        <parameters list=\"hsa_queue_create, hsa_amd_memory_pool_allocate\">"
  echo "        </parameters>"
  echo "      </trace>"
  echo ""
  echo "  --roctx-rename - to rename kernels with their enclosing rocTX range's message."
  echo ""
  echo "  --trace-start <on|off> - to enable tracing on start [on]"
  echo "  --trace-period <dealy:length:rate> - to enable trace with initial delay, with periodic sample length and rate"
  echo "    Supported time formats: <number(m|s|ms|us)>"
  echo "  --flush-rate <rate> - to enable trace flush rate (time period)"
  echo "    Supported time formats: <number(m|s|ms|us)>"
  echo "  --parallel-kernels - to enable concurrent kernels"
  echo ""
  echo "Configuration file:"
  echo "  You can set your parameters defaults preferences in the configuration file 'rpl_rc.xml'. The search path sequence: .:${HOME}:<installation directory>"
  echo "  First the configuration file is searched in the current directory, then in the current user's home directory, and then in the installation directory."
  echo "  Configurable options: 'basenames', 'timestamp', 'ctx-limit', 'heartbeat', 'obj-tracking'."
  echo "  An example of 'rpl_rc.xml':"
  echo "    <defaults"
  echo "      basenames=off"
  echo "      timestamp=off"
  echo "      ctx-limit=0"
  echo "      heartbeat=0"
  echo "      obj-tracking=off"
  echo "    ></defaults>"
  echo ""
  echo "  --merge-traces - Script for aggregating results from multiple rocprofiler out directries."
  echo "                   Usage: if running with rocprof"
  echo "                   rocprof --merge-traces -o <outputdir> [<inputdir>...]"
  echo ""
  exit 1
}

# checking for availability of rocminfo utility
if !command -v rocminfo > /dev/null 2>&1 ;  then
  error "'rocminfo' utility is not found: please add ROCM bin path to PATH env var.";
fi

# setting ROCM_LIB_PATH
set_rocm_lib_path() {

  for ROCM_LIB_PATH in "$ROOT_DIR/lib" "$ROOT_DIR/lib64" ; do
     if [ -d "$ROCM_LIB_PATH" ]; then
        return 0
     fi
  done

  #error
  return 255
}

# profiling run method
OUTPUT_LIST=""
run() {
  if ! set_rocm_lib_path ; then
     echo " Fatal could not find ROCm lib directory "
     fatal
  fi

  # split the CURRENT_AGENTS_LIST array into individual elements.
  current_gpus=(${CURRENT_AGENTS_LIST[@]})

  for gpu in "${current_gpus[@]}"; do

    # Check first 5 characters of gpu strings.
    if [[ "${V1_SUPPORTED_GPU_ARCHS[@]:0:5}" =~ "${gpu:0:5}" ]]; then
      IS_SUPPORTED="true"
    fi

  done

  if [[ $IS_SUPPORTED == "false" ]]; then
    echo ""
    echo "------------ ------------ ------------"
    echo "WARNING: rocprof(v1) is not supported on this device. Recommended use: rocprofv2"
    echo "Please refer project's README for a list of supported architecures."
    echo "------------ ------------ ------------"
    echo ""
  fi

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
  MY_LD_PRELOAD=""
  if [ "$ROCTX_TRACE" = 1 ] ; then
    API_TRACE=${API_TRACE}":roctx"
    MY_LD_PRELOAD="$ROCM_LIB_PATH/libroctx64.so"
  fi
  if [ "$HIP_TRACE" = 1 ] ; then
    API_TRACE=${API_TRACE}":hip"
  fi
  if [ "$SYS_TRACE" = 1 ] ; then
    API_TRACE=${API_TRACE}":sys"
  fi

  if [ "$HSA_TRACE" = 1 ] ; then
    export ROCTRACER_DOMAIN=$API_TRACE":hsa"
    ROCPROFV1_TOOL_PRELOAD="$MY_LD_PRELOAD $TTLIB_PATH/libroctracer_tool.so"
    ROCPROFV1_LD_PRELOAD="$MY_HSA_TOOLS_LIB:$ROCM_LIB_PATH/libroctracer64.so.4"
  elif [ -n "$API_TRACE" ] ; then
    export ROCTRACER_DOMAIN=$API_TRACE
    OUTPUT_LIST="$ROCP_OUTPUT_DIR/"
    ROCPROFV1_TOOL_PRELOAD="$MY_LD_PRELOAD $TTLIB_PATH/libroctracer_tool.so"
    ROCPROFV1_LD_PRELOAD="$ROCM_LIB_PATH/libroctracer64.so.4"
  fi

  if [ "$ROCP_STATS_OPT" = 1 ] ; then
    if [ "$ROCTRACER_DOMAIN" = ":hip" ] ; then
      ROCPROFV1_LD_PRELOAD="$ROCM_LIB_PATH/libroctracer64.so.4"
      ROCPROFV1_TOOL_PRELOAD="$MY_LD_PRELOAD $TTLIB_PATH/libhip_stats.so"
    else
      error_message="ROCP_STATS_OPT is only available with --hip-trace option"
      echo $error_message
      exit 1
    fi
  fi

  retval=1
  if [ -n "$ROCP_OUTPUT_DIR" ] ; then
    log_file="$ROCP_OUTPUT_DIR/log.txt"
    exit_file="$ROCP_OUTPUT_DIR/exit.txt"
    {
      HSA_TOOLS_LIB="$ROCPROFV1_TOOL_PRELOAD" LD_PRELOAD=$LD_PRELOAD:"$ROCPROFV1_LD_PRELOAD" eval "$APP_CMD"
      retval=$?
      echo "exit($retval)" > $exit_file
    } 2>&1 | tee "$log_file"
    exitval=`cat "$exit_file" | sed -n "s/^.*exit(\([0-9]*\)).*$/\1/p"`
    if [ -n "$exitval" ] ; then retval=$exitval; fi
  else
      HSA_TOOLS_LIB="$ROCPROFV1_TOOL_PRELOAD" LD_PRELOAD=$LD_PRELOAD:"$ROCPROFV1_LD_PRELOAD" eval "$APP_CMD"
    retval=$?
  fi
  return $retval
}

merge_output() {
  while [ -n "$1" ] ; do
    output_dir=$(echo "$1" | sed "s/\/[^\/]*$//")
    for file_name in `ls $output_dir` ; do
      output_name=$(echo $file_name | sed -n "/\.txt$/ s/^[0-9]*_//p")
      if [ -n "$output_name" ] ; then
        trace_file=$output_dir/$file_name
        output_file=$output_dir/$output_name
        touch $output_file
        cat $trace_file >> $output_file
      fi
    done
    shift
  done
}

convert_time_val() {
  local time_maxumim_us=$((0xffffffff))
  local __resultvar=$1
  eval "local val=$"$__resultvar
  val_m=`echo $val | sed -n "s/^\([0-9]*\)m$/\1/p"`
  val_s=`echo $val | sed -n "s/^\([0-9]*\)s$/\1/p"`
  val_ms=`echo $val | sed -n "s/^\([0-9]*\)ms$/\1/p"`
  val_us=`echo $val | sed -n "s/^\([0-9]*\)us$/\1/p"`
  if [ -n "$val_m" ] ; then val_us=$((val_m*60000000))
  elif [ -n "$val_s" ] ; then val_us=$((val_s*1000000))
  elif [ -n "$val_ms" ] ; then val_us=$((val_ms*1000))
  fi

  if [ -z "$val_us" ] ; then
    error_message="invalid time value format ($val)"
  elif [ "$val_us" -gt "$time_maxumim_us" ] ; then
    error_message="time value exceeds maximum supported ($val > ${time_maxumim_us}us)"
  else
    eval $__resultvar="'$val_us'"
  fi
}

unset_v1_envs() {
  # enable error logging
 unset HSA_TOOLS_REPORT_LOAD_FAILURE
 unset HSA_VEN_AMD_AQLPROFILE_LOG
 unset ROCPROFILER_LOG
 unset ROCPROFILER_SESS
 unset MY_HSA_TOOLS_LIB
 unset ROCP_TOOL_LIB
 unset ROCP_METRICS
 unset ROCP_HSA_INTERCEPT
 unset ROCP_PROXY_QUEUE
 unset AQLPROFILE_READ_API
 unset ROCP_PACKAGE_DIR
 unset ROCP_SPM_KFD_MODE
}

################################################################################################
# main
echo "RPL: on '$time_stamp' from '$ROOT_DIR' in '$RUN_DIR'"
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
  if [ "$1" = "--tool-version" ] ; then
    if [ $2 = 1 ] ; then
      :
    elif [ $2 = 2 ] ; then
      # unset all v1 variables
      unset_v1_envs
      eval $BIN_DIR/rocprofv2 $ROCPROF_ARGS
      exit 0
    else
      echo "Wrong option '$1 $2'"
      usage
    fi
  elif [ "$1" = "--version" ]; then
    if [ -f "$BIN_DIR/../libexec/rocprofiler/rocprofiler-version" ]; then
      ROCPROFILER_LIBRARY_VERSION=1 $BIN_DIR/../libexec/rocprofiler/rocprofiler-version
    else
      ROCM_VERSION=$(cat $BIN_DIR/../.info/version)
      echo -e "ROCm version: $ROCM_VERSION"
      echo -e "ROCProfiler version: 2.0"
    fi
    exit 0
  elif [ "$1" = "-h" ] ; then
    usage
  elif [ "$1" = "-i" ] ; then
    INPUT_FILE="$2"
  elif [ "$1" = "-o" ] ; then
    output="$2"
    if [[ $output != *.csv ]]; then
      echo "error: file name must have .CSV extension"
      exit 1
    fi
  elif [ "$1" = "-d" ] ; then
    DATA_PATH=$2
  elif [ "$1" = "-t" ] ; then
    TMP_DIR="$2"
    if [ "$OUTPUT_DIR" = "-" ] ; then
      DATA_PATH=$TMP_DIR
    fi
  elif [ "$1" = "-m" ] ; then
    unset ROCP_METRICS
    export ROCP_METRICS="$2"
  elif [ "$1" = "--list-basic" ] ; then
    export ROCP_INFO=b
    HSA_TOOLS_LIB="$MY_HSA_TOOLS_LIB" eval "$TLIB_PATH/rocprof-ctrl"
    exit 1
  elif [ "$1" = "--list-derived" ] ; then
    export ROCP_INFO=d
    HSA_TOOLS_LIB="$MY_HSA_TOOLS_LIB" eval "$TLIB_PATH/rocprof-ctrl"
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
  elif [ "$1" = "--roctx-trace" ] ; then
    ARG_VAL=0
    GEN_STATS=1
    ROCTX_TRACE=1
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
  elif [ "$1" = "--roctx-rename" ] ; then
    ARG_VAL=0
    export ROCP_RENAME_KERNEL=1
  elif [ "$1" = "--trace-start" ] ; then
    if [ "$2" = "off" ] ; then
      export ROCP_CTRL_RATE="-1:0:0"
    fi
  elif [ "$1" = "--trace-period" ] ; then
    period_expr="^\([^:]*\):\([^:]*\):\([^:]*\)$"
    period_ck=`echo "$2" | sed -n "s/"${period_expr}"/ok/p"`
    if [ -z "$period_ck" ] ; then
      fatal "Wrong option '$1 $2'"
    fi
    period_delay=`echo "$2" | sed -n "s/"${period_expr}"/\1/p"`
    period_len=`echo "$2" | sed -n "s/"${period_expr}"/\2/p"`
    period_rate=`echo "$2" | sed -n "s/"${period_expr}"/\3/p"`
    convert_time_val period_delay
    errck "Option '$ARG_IN', delay value"
    convert_time_val period_len
    errck "Option '$ARG_IN', length value"
    convert_time_val period_rate
    errck "Option '$ARG_IN', rate value"
    export ROCP_CTRL_RATE="$period_delay:$period_len:$period_rate"
  elif [ "$1" = "--flush-rate" ] ; then
    period_rate=$2
    convert_time_val period_rate
    errck "Option '$ARG_IN', rate value"
    export ROCP_FLUSH_RATE="$period_rate"
  elif [ "$1" = "--obj-tracking" ] ; then
    if [ "$2" = "off" ] ; then
      export ROCP_OBJ_TRACKING=0
    fi
  elif [ "$1" = "--parallel-kernels" ] ; then
    ARG_VAL=0
    export ROCP_K_CONCURRENT=1
    export AQLPROFILE_READ_API=1
  elif [ "$1" = "--verbose" ] ; then
    ARG_VAL=0
    export ROCP_VERBOSE_MODE=1
  elif [ "$1" = "--cmd-qts" ] ; then
    if [ "$2" = "off" ] ; then
      CMD_QTS=0
    fi
  elif [ "$1" = "--merge-traces" ] ; then
    shift
    echo "merging traces with $PROF_BIN_DIR/merge_traces.sh"
    $PROF_BIN_DIR/merge_traces.sh $@
    exit 0
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

if [ "$GEN_STATS" = "1" -a "$ROCP_TIMESTAMP_ON" = "0" ] ; then
  fatal "Wrong options, stats enabled with disabled timestamps"
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

# Profiled cmd line string
APP_CMD=$*
if [ "$CMD_QTS" = 1 ] ; then
  APP_CMD=""
  for i in `seq 1 $#`; do
    if [ -n "$APP_CMD" ] ; then
      APP_CMD=$APP_CMD" "
    fi
    eval "arg=\${$i}"
    APP_CMD=$APP_CMD\"$arg\"
  done
fi

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
    $PROF_BIN_DIR/txt2xml.sh $INPUT_FILE $RES_DIR
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

RET=1
export ROCP_MERGE_PIDS=1
for name in $input_list; do
  run $name $OUTPUT_DIR $APP_CMD
  RET=$?
  if [ -n "$ROCPROFILER_SESS" -a -e "$ROCPROFILER_SESS/error" ] ; then
    error_string=`cat $ROCPROFILER_SESS/error`
    echo "Profiling error found: '$error_string'"
    csv_output=""
    RET=1
    break
  fi
done

if [ -n "$csv_output" ] ; then
  merge_output $OUTPUT_LIST
  if [ "$GEN_STATS" = "1" ] ; then
    db_output=$(echo $csv_output | sed "s/\.csv/.db/")
    $ROCP_PYTHON_VERSION $PROF_BIN_DIR/tblextr.py $db_output $OUTPUT_LIST
  else
    $ROCP_PYTHON_VERSION $PROF_BIN_DIR/tblextr.py $csv_output $OUTPUT_LIST
  fi
  if [ "$?" -ne 0 ] ; then
    echo "Profiling data corrupted: '$OUTPUT_LIST'" | tee "$ROCPROFILER_SESS/error"
    RET=1
  fi
fi

if [ "$DATA_PATH" = "$TMP_DIR" ] ; then
  if [ -e "$RES_DIR" ] ; then
    rm -rf $RES_DIR
  fi
fi

exit $RET
