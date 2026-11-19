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
#
# Shared setup sourced by scripts/build_configs/*. Not meant to be executed
# directly.

if [[ "${ASAN}" == true ]]; then
    export HSA_XNACK=1
fi

generator_args=()
if [[ ! -f CMakeCache.txt ]] && command -v ninja >/dev/null 2>&1; then
    generator_args=(-G Ninja)
fi

build_jobs=$(nproc)
if (( build_jobs > 32 )); then
    build_jobs=32
fi

###############################################################################
# STAGE TIMING
###############################################################################
# Wraps a build stage ("configure", "build", "install", ...) with wall-clock
# timing so the three cmake steps (which the caller's own `time ...` lumps
# together) can be told apart. Appends to build_timing.log in the cwd (the
# build directory). Disable with ROCSHMEM_BUILD_TIMING=0.
rocshmem_timing_log=${ROCSHMEM_BUILD_TIMING_LOG:-build_timing.log}

time_stage() {
    local stage_name="$1"
    shift
    if [[ "${ROCSHMEM_BUILD_TIMING:-1}" == 0 ]]; then
        "$@"
        return
    fi
    local start end elapsed
    start=$(date +%s.%N)
    "$@"
    end=$(date +%s.%N)
    elapsed=$(awk -v s="${start}" -v e="${end}" 'BEGIN{printf "%.2f", e-s}')
    printf '[timing] %-10s %ss\n' "${stage_name}" "${elapsed}" | tee -a "${rocshmem_timing_log}"
}
