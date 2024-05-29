#!/bin/bash -e

set -eo pipefail

ROCPROFV3_DIR=$(dirname -- "$(realpath "${BASH_SOURCE[0]}")")
ROCM_DIR=$(dirname -- "${ROCPROFV3_DIR}")

: ${ROCPROFILER_LIBRARY_CTOR:=1}
: ${ROCPROF_OUTPUT_PATH:="."}
: ${ROCPROF_OUTPUT_PATH_INTERNAL:="."}
: ${ROCPROF_OUTPUT_FILE_NAME:=""}
: ${ROCPROF_COUNTERS_PATH:=""}
: ${ROCPROF_PRELOAD:=""}
: ${ROCPROF_TOOL_LIBRARY:="${ROCM_DIR}/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"}
: ${ROCPROF_SDK_LIBRARY:="${ROCM_DIR}/lib/librocprofiler-sdk.so"}

export ROCPROFILER_LIBRARY_CTOR

# Define color codes
GREEN='\033[0;32m'
GREY='\033[0;90m'
RESET='\033[0m'

usage() {
  local EC=${1}
  if [ -z "${EC}" ]; then EC=1; fi
  echo -e "${RESET}ROCProfilerV3 Run Script Usage:"
  echo -e "${GREEN}-h   | --help ${RESET}                For showing this message"
  echo -e ""
  echo -e "${GREEN}--hip-trace ${RESET}                  For Collecting HIP Traces (runtime + compiler)"
  echo -e "${GREEN}--hip-runtime-trace ${RESET}          For Collecting HIP Runtime API Traces"
  echo -e "${GREEN}--hip-compiler-trace ${RESET}         For Collecting HIP Compiler generated code Traces"
  echo -e ""
  echo -e "${GREEN}--marker-trace ${RESET}               For Collecting Marker (ROCTx) Traces"
  echo -e "${GREEN}--kernel-trace ${RESET}               For Collecting Kernel Dispatch Traces"
  echo -e "${GREEN}--memory-copy-trace ${RESET}          For Collecting Memory Copy Traces"
  echo -e "${GREEN}--scratch-memory-trace ${RESET}       For Collecting Scratch Memory operations Traces"
  echo -e ""
  echo -e "${GREEN}--hsa-trace ${RESET}                  For Collecting HSA API Traces (core + amd + image + finalizer)"
  echo -e "${GREEN}--hsa-core-trace ${RESET}             For Collecting HSA API Traces (core API)"
  echo -e "${GREEN}--hsa-amd-trace ${RESET}              For Collecting HSA API Traces (AMD-extension API)"
  echo -e "${GREEN}--hsa-image-trace ${RESET}            For Collecting HSA API Traces (Image-extenson API)"
  echo -e "${GREEN}--hsa-finalizer-trace ${RESET}        For Collecting HSA API Traces (Finalizer-extension API)"
  echo -e ""
  echo -e "${GREEN}--sys-trace ${RESET}                  For Collecting HIP, HSA, Marker (ROCTx), Memory copy, Scratch memory, and Kernel dispatch traces\n"
  echo -e "${GREEN}--stats ${RESET}                      For Collecting statistics of enabled tracing types\n"
  echo -e "\t#${GREY} Examples:"
  echo -e "\t#${GREY} (Kernel Dispatch Trace Statistics): rocprofv3 --kernel-trace --stats <executable>"
  echo -e "\t#${GREY} (HSA API Trace Statistics): rocprofv3 --hsa-trace --stats <executable>"
  echo -e "\t#${GREY} (HIP API + Kernel Dispatch Trace Statistics): rocprofv3 --hip-trace --kernel-trace --stats <executable>"
  echo -e "\t#${GREY} (Memory Copy Trace Statistics): rocprofv3 --memory-copy-trace --stats <executable>"
  echo -e ""
  echo -e "${GREEN}-o   | --output-file ${RESET}         For the output file name"
  echo -e "\t#${GREY} usage (with current dir): rocprofv3 --hsa-trace -o <file_name> <executable>"
  echo -e "\t#${GREY} usage (with custom dir):  rocprofv3 --hsa-trace -d <out_dir> -o <file_name> <executable>${RESET}\n"
  echo -e ""
  echo -e "${GREEN}-d   | --output-directory ${RESET}    For adding output path where the output files will be saved"
  echo -e "\t#${GREY} usage (with custom dir):  rocprofv3 --hsa-trace -d <out_dir> <executable>${RESET}"
  echo -e "${GREEN} | --output-format ${RESET}    For adding output format(supported formats: csv, json)"
  echo -e ""
  echo -e "${GREEN}--output-format ${RESET}              For specifying output format. Case-insensitive, comma separated. Options: CSV, JSON, PFTRACE"
  echo -e "\t#${GREY} Examples:"
  echo -e "\t#${GREY} (JSON output):     rocprofv3 --sys-trace --output-format JSON         <executable>"
  echo -e "\t#${GREY} (JSON + CSV):      rocprofv3 --sys-trace --output-format JSON,CSV     <executable>"
  echo -e "\t#${GREY} (JSON + PFTRACE):  rocprofv3 --sys-trace --output-format JSON,PFTRACE <executable>"
  echo -e ""
  echo -e "${GREEN}-M   | --mangled-kernels ${RESET}     Do not demangle the kernel names"
  echo -e "${GREEN}-T   | --truncate-kernels ${RESET}    Truncate the demangled kernel names"
  echo -e ""
  echo -e "${GREEN}-L   | --list-metrics ${RESET}        List metrics for counter collection"
  echo -e "${GREEN}-i   | --input ${RESET}               For counter collection "
  echo -e "\t#${GREY} Input file .txt format, automatically rerun application for every profiling features line"
  echo -e "\t# Perf counters group 1"
  echo -e "\tpmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts"
  echo -e "\t# Perf counters group 2"
  echo -e "\tpmc : WriteSize L2CacheHit ${RESET}"
  echo -e ""
  exit ${EC}
}

if [ -z "$1" ]; then
  usage 1
fi

if [ -n "${ROCPROF_PRELOAD}" ]; then
    ROCPROF_PRELOAD="${ROCPROF_PRELOAD}:${ROCPROF_TOOL_LIBRARY}:${ROCPROF_SDK_LIBRARY}"
else
    ROCPROF_PRELOAD="${ROCPROF_TOOL_LIBRARY}:${ROCPROF_SDK_LIBRARY}"
fi

if [ -n "${ROCP_TOOL_LIBRARIES}" ]; then
    ROCP_TOOL_LIBRARIES="${ROCP_TOOL_LIBRARIES}:${ROCPROF_TOOL_LIBRARY}"
else
    ROCP_TOOL_LIBRARIES="${ROCPROF_TOOL_LIBRARY}"
fi

LD_LIBRARY_PATH=${ROCM_DIR}/lib:${LD_LIBRARY_PATH}

export ROCP_TOOL_LIBRARIES
export LD_LIBRARY_PATH

function check_tracing_enabled() {
    if [[ -n "$ROCPROF_HSA_CORE_API_TRACE" || -n "$ROCPROF_HSA_AMD_EXT_API_TRACE" ||
          -n "$ROCPROF_HSA_IMAGE_EXT_API_TRACE" || -n "$ROCPROF_HSA_FINALIZER_EXT_API_TRACE" ||
          -n "$ROCPROF_HIP_RUNTIME_API_TRACE" || -n "$ROCPROF_HIP_COMPILER_API_TRACE" ||
          -n "$ROCPROF_KERNEL_TRACE" || -n "$ROCPROF_MEMORY_COPY_TRACE" || -n "$ROCPROF_SCRATCH_MEMORY_TRACE"
        ]]; then
        return 0  # Return true if at least one tracing option is set
    else
        return 1  # Return false if none of the tracing options are set
    fi
}

while true; do
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage 0
  elif [[ "$1" == "-M" || "$1" == "--mangled-kernels" ]]; then
    export ROCPROF_DEMANGLE_KERNELS=0
    shift
  elif [[ "$1" == "-T" || "$1" == "--truncate-kernels" ]]; then
    export ROCPROF_TRUNCATE_KERNELS=1
    shift
  elif [[ "$1" == "-i" || "$1" == "--input" ]]; then
    if [ "$2" ] && [ -n "$2" ] && [ -r "$2" ]; then
      export ROCPROF_COUNTERS_PATH=$2
      export ROCPROF_COUNTER_COLLECTION=1
    else
      echo -e "Error: input file \"$2\" doesn't exist!"
      usage 1
    fi
    shift
    shift
  elif [[ "$1" == "-o" || "$1" == "--output-file-name" ]]; then
    if [ "$2" ]; then
      export ROCPROF_OUTPUT_FILE_NAME=$2
      export ROCPROF_OUTPUT_LIST_METRICS_FILE=1
    else
      usage 1
    fi
    shift
    shift
  elif [[ "$1" == "-d" || "$1" == "--output-directory" ]]; then
    if [ "$2" ]; then
      ROCPROF_OUTPUT_PATH_INTERNAL=$2
      export ROCPROF_OUTPUT_PATH=$ROCPROF_OUTPUT_PATH_INTERNAL
      export ROCPROF_OUTPUT_LIST_METRICS_FILE=1
    else
      usage 1
    fi
    shift
    shift
  elif [[ "$1" == "--output-format" ]]; then
    if [ "$2" ]; then
      export ROCPROF_OUTPUT_FORMAT=$2
    else
      usage 1
    fi
    shift
    shift
  elif [ "$1" == "--hsa-trace" ]; then
    export ROCPROF_HSA_CORE_API_TRACE=1
    export ROCPROF_HSA_AMD_EXT_API_TRACE=1
    export ROCPROF_HSA_IMAGE_EXT_API_TRACE=1
    export ROCPROF_HSA_FINALIZER_EXT_API_TRACE=1
    shift
  elif [ "$1" == "--hsa-core-trace" ]; then
    export ROCPROF_HSA_CORE_API_TRACE=1
    shift
  elif [ "$1" == "--hsa-amd-trace" ]; then
    export ROCPROF_HSA_AMD_EXT_API_TRACE=1
    shift
  elif [ "$1" == "--hsa-image-trace" ]; then
    export ROCPROF_HSA_IMAGE_EXT_API_TRACE=1
    shift
  elif [ "$1" == "--hsa-finalizer-trace" ]; then
    export ROCPROF_HSA_FINALIZER_EXT_API_TRACE=1
    shift
  elif [[ "$1" == "-L" || "$1" == "--list-metrics" ]]; then
    export ROCPROF_LIST_METRICS=1
    shift
  elif [ "$1" == "--kernel-trace" ]; then
    export ROCPROF_KERNEL_TRACE=1
    shift
  elif [ "$1" == "--memory-copy-trace" ]; then
    export ROCPROF_MEMORY_COPY_TRACE=1
    shift
  elif [ "$1" == "--scratch-memory-trace" ]; then
    export ROCPROF_SCRATCH_MEMORY_TRACE=1
    shift
  elif [ "$1" == "--marker-trace" ]; then
    export ROCPROF_MARKER_API_TRACE=1
    shift
  elif [ "$1" == "--hip-trace" ]; then
    export ROCPROF_HIP_RUNTIME_API_TRACE=1
    export ROCPROF_HIP_COMPILER_API_TRACE=1
    shift
  elif [ "$1" == "--hip-runtime-trace" ]; then
    export ROCPROF_HIP_RUNTIME_API_TRACE=1
    shift
  elif [ "$1" == "--hip-compiler-trace" ]; then
    export ROCPROF_HIP_COMPILER_API_TRACE=1
    shift
  elif [ "$1" == "--sys-trace" ]; then
    export ROCPROF_HSA_CORE_API_TRACE=1
    export ROCPROF_HSA_AMD_EXT_API_TRACE=1
    export ROCPROF_HSA_IMAGE_EXT_API_TRACE=1
    export ROCPROF_HSA_FINALIZER_EXT_API_TRACE=1
    export ROCPROF_HIP_RUNTIME_API_TRACE=1
    export ROCPROF_HIP_COMPILER_API_TRACE=1
    export ROCPROF_MARKER_API_TRACE=1
    export ROCPROF_KERNEL_TRACE=1
    export ROCPROF_MEMORY_COPY_TRACE=1
    export ROCPROF_SCRATCH_MEMORY_TRACE=1
    shift
  elif [ "$1" == "--stats" ]; then
    export ROCPROF_STATS=1
    shift
  elif [ "$1" == "--" ]; then
    shift
    break
  elif [[ "$1" == "-"* || "$1" == "--"* ]]; then
    echo -e "Wrong option \"$1\", Please use the following options:\n"
    usage 1
  else
    break
  fi
done

# read input counter file
PMC_LINES=()
if [ -n "$ROCPROF_COUNTERS_PATH" ]; then
  input=$ROCPROF_COUNTERS_PATH
  while IFS= read -r line || [[ -n "$line" ]]; do
    #skip empty lines
    if [[ -z "$line" ]]; then
      continue
    fi
    PMC_LINES+=("$line")
  done <"$input"
fi

if [ -n "${PMC_LINES:-}" ]; then
  #for counter collection
  COUNTER=1
  for i in "${!PMC_LINES[@]}"; do
    export ROCPROF_COUNTERS="${PMC_LINES[$i]}"
    if [[ ! ${PMC_LINES[$i]} =~ "pmc" ]]; then
      continue
    fi

    RESULT_PATH="$ROCPROF_OUTPUT_PATH_INTERNAL/pmc_$COUNTER"
    if [ -n "$ROCPROF_OUTPUT_FILE_NAME" ] || [ -n "$ROCPROF_OUTPUT_PATH" ]; then
      export ROCPROF_OUTPUT_PATH=$RESULT_PATH
    fi
    ((COUNTER++))
    LD_PRELOAD="${ROCPROF_PRELOAD}" "${@}"
    if [ -n "$ROCPROF_OUTPUT_PATH" ]; then
      echo -e "\nThe output path for the following counters: $ROCPROF_OUTPUT_PATH"
    fi
  done
elif [ -n "$ROCPROF_LIST_METRICS" ]; then
    LD_PRELOAD="${ROCPROF_PRELOAD}" exec ${ROCM_DIR}/lib/rocprofiler-sdk/rocprofv3-trigger-list-metrics
else
  if ! check_tracing_enabled && [ "$ROCPROF_STATS" == 1 ]; then
    echo -e "Error: Please enable at least one tracing option to collect statistics."
    echo -e "eg: rocprofv3 --stats --kernel-trace <executable>"
    exit 1
   fi
  # for non counter collection. e.g: tracing
  LD_PRELOAD="${ROCPROF_PRELOAD}" exec "${@}"
fi
