#!/bin/bash -e

#
#   This file will export the same environment variables for running sanitizers as run-ci.py
#   This file is useful to set the suppressions files
#
#   Example usage:
#
#       source ./source/scripts/setup-sanitizer-env.sh
#

SUPPR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) &> /dev/null && pwd)

find-symbolizer()
{
    set +e
    SYMBOLIZER=$(which ${1})
    set -e
    if [ -n "${SYMBOLIZER}" ]; then
        : ${EXTERNAL_SYMBOLIZER_PATH:="${SYMBOLIZER}"}
    fi
}

for i in $(seq 20 -1 8)
do
    find-symbolizer llvm-symbolizer-${i}
done

find-symbolizer llvm-symbolizer

if [ -n "${EXTERNAL_SYMBOLIZER_PATH}" ]; then
    EXTERNAL_SYMBOLIZER=" external_symbolizer_path=${EXTERNAL_SYMBOLIZER_PATH}"
fi

: ${ASAN_OPTIONS="detect_leaks=0 use_sigaltstack=0 suppressions=${SUPPR_DIR}/address-sanitizer-suppr.txt"}
: ${LSAN_OPTIONS="suppressions=${SUPPR_DIR}/leak-sanitizer-suppr.txt"}
: ${TSAN_OPTIONS="history_size=5 detect_deadlocks=0 suppressions=${SUPPR_DIR}/thread-sanitizer-suppr.txt${EXTERNAL_SYMBOLIZER}"}
: ${UBSAN_OPTIONS="print_stacktrace=1 suppressions=${SUPPR_DIR}/undef-behavior-sanitizer-suppr.txt${EXTERNAL_SYMBOLIZER}"}

export ASAN_OPTIONS
export LSAN_OPTIONS
export TSAN_OPTIONS
export UBSAN_OPTIONS

echo "ASAN_OPTIONS=\"${ASAN_OPTIONS}\""
echo "LSAN_OPTIONS=\"${LSAN_OPTIONS}\""
echo "TSAN_OPTIONS=\"${TSAN_OPTIONS}\""
echo "UBSAN_OPTIONS=\"${UBSAN_OPTIONS}\""
