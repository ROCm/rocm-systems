# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Locate the ROCm amdclang++ compiler selected by ROCM_PATH.
#
# Some systems expose /usr/bin/amdclang++ through alternatives; using that
# wrapper can derive --rocm-path as /usr and mix headers/device libraries from
# different installs. Keep this lookup constrained to ROCM_PATH.
function(rj_find_amdcxx out_var)
    find_program(
        _rj_amdcxx
        NAMES amdclang++
        PATHS "${ROCM_PATH}"
        PATH_SUFFIXES lib/llvm/bin bin
        NO_DEFAULT_PATH
        NO_CACHE
    )
    set(${out_var} "${_rj_amdcxx}" PARENT_SCOPE)
endfunction()
