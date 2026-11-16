# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_rocm_path)

rj_find_rocm_program(AMDCXX amdclang++)

# Share the capability result with kernel fixtures and host-side HIP tests.
set(RJ_HAS_HIP_KERNEL_TOOLCHAIN FALSE)
if(AMDCXX)
    include(rj_check_hip_kernel_toolchain)
endif()

# Probe ISA parsing before adding a target-specific HIP test. The build of the
# real test still uses the selected ROCM_PATH and its packaged device library.
function(rj_amdcxx_supports_arch out_var compiler arch)
    set(_rj_supports FALSE)
    if(compiler)
        execute_process(
            COMMAND
                "${compiler}" -x hip --offload-arch=${arch}
                --rocm-path=${ROCM_PATH} -nogpulib --cuda-device-only -c
                /dev/null -o /dev/null
            RESULT_VARIABLE _rj_probe_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(_rj_probe_result STREQUAL "0")
            set(_rj_supports TRUE)
        endif()
    endif()
    set(${out_var} "${_rj_supports}" PARENT_SCOPE)
endfunction()
