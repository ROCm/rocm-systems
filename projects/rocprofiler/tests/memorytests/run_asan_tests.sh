#!/bin/bash

CURRENT_DIR="$( dirname -- "$0"; )";
export PATH=$rocmtoolsdir:$PATH

echo -e "Running Memory Leaks Check From ${CURRENT_DIR}"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.6 ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=$CURRENT_DIR/suppr.txt ${CURRENT_DIR}/../../rocprofv2 -i $CURRENT_DIR/input.txt $1 > /tmp/rocprofv2-temp 2> $2
echo -e "Log with all detected leaks is available at build/memleaks.log"
eval python3 $CURRENT_DIR/test_mem.py $2
