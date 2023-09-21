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

for i in $(seq 20 -1 8)
do
    set +e
    SYMBOLIZER=$(which llvm-symbolizer-${i})
    set -e
    if [ -n "${SYMBOLIZER}" ]; then
        : ${EXTERNAL_SYMBOLIZER_PATH:="${SYMBOLIZER}"}
    fi
done

if [ -n "${EXTERNAL_SYMBOLIZER_PATH}" ]; then
    EXTERNAL_SYMBOLIZER=" external_symbolizer_path=${EXTERNAL_SYMBOLIZER_PATH}"
fi

: ${ASAN_OPTIONS="detect_leaks=0 use_sigaltstack=0 suppressions=${SUPPR_DIR}/address-sanitizer-suppr.txt"}
: ${LSAN_OPTIONS="suppressions=${SUPPR_DIR}/leak-sanitizer-suppr.txt"}
: ${TSAN_OPTIONS="history_size=5 second_deadlock_stack=1 suppressions=${SUPPR_DIR}/thread-sanitizer-suppr.txt${EXTERNAL_SYMBOLIZER}"}

export ASAN_OPTIONS
export LSAN_OPTIONS
export TSAN_OPTIONS

echo "ASAN_OPTIONS=\"${ASAN_OPTIONS}\""
echo "LSAN_OPTIONS=\"${LSAN_OPTIONS}\""
echo "TSAN_OPTIONS=\"${TSAN_OPTIONS}\""
