# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# -------------------------------------------------------------------------------------- #
#
# openmp tests
#
# -------------------------------------------------------------------------------------- #

if(ROCmVersion_DIR)
    set(_rocm_root "${ROCmVersion_DIR}")
elseif(DEFINED ENV{ROCM_PATH})
    set(_rocm_root "$ENV{ROCM_PATH}")
else()
    set(_rocm_root "/opt/rocm")
endif()

set(_rocm_llvm_lib "${_rocm_root}/lib/llvm/lib")

set(_rocm_ld_env
    "LD_PRELOAD=libomptarget.so"
    "LD_LIBRARY_PATH=${_rocm_llvm_lib}:$ENV{LD_LIBRARY_PATH}"
)

if(NOT EXISTS "${_rocm_llvm_lib}/libomptarget.so" AND ROCPROFSYS_USE_ROCM)
    message(
        FATAL_ERROR
        "libomptarget.so not found in ${_rocm_llvm_lib}. "
        "Verify that ROCm is installed correctly and that _rocm_root "
        "(${_rocm_root}) points at the right location."
    )
endif()

if(ROCPROFSYS_OPENMP_USING_LIBOMP_LIBRARY AND ROCPROFSYS_USE_OMPT)
    set(_OMPT_PASS_REGEX "\\|_ompt_")
else()
    set(_OMPT_PASS_REGEX "")
endif()

rocprofiler_systems_add_test(
    NAME openmp-cg
    TARGET openmp-cg
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT
        "${_ompt_environment};ROCPROFSYS_USE_SAMPLING=OFF;ROCPROFSYS_COUT_OUTPUT=ON"
    REWRITE_RUN_PASS_REGEX "${_OMPT_PASS_REGEX}"
    RUNTIME_PASS_REGEX "${_OMPT_PASS_REGEX}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure"
)

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME openmp-lu
    TARGET openmp-lu
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args -E ^GOMP
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT
        "${_ompt_environment};ROCPROFSYS_USE_SAMPLING=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_COUT_OUTPUT=ON"
    REWRITE_RUN_PASS_REGEX "${_OMPT_PASS_REGEX}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure"
)

rocprofiler_systems_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME openmp-target
    TARGET openmp-target
    GPU ON
    LABELS "openmp;openmp-target"
    ENVIRONMENT
        "${_ompt_environment};${_rocm_ld_env};ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"
)

rocprofiler_systems_add_validation_test(
    NAME openmp-target-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "openmp;openmp-target"
    ENVIRONMENT "${_rocm_ld_env}"
    ARGS --label-substrings
         Z4vmulIiEvPT_S1_S1_i_l51.kd
         Z4vmulIfEvPT_S1_S1_i_l51.kd
         Z4vmulIdEvPT_S1_S1_i_l51.kd
         -c
         4
         4
         4
         -d
         0
         0
         0
         -p
)

set(_ompt_sampling_environ
    "${_ompt_environment}"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_USE_OMPT=OFF"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_SAMPLING_FREQ=100"
    "ROCPROFSYS_SAMPLING_DELAY=0.1"
    "ROCPROFSYS_SAMPLING_DURATION=0.25"
    "ROCPROFSYS_SAMPLING_CPUTIME=ON"
    "ROCPROFSYS_SAMPLING_REALTIME=ON"
    "ROCPROFSYS_SAMPLING_CPUTIME_FREQ=1000"
    "ROCPROFSYS_SAMPLING_REALTIME_FREQ=500"
    "ROCPROFSYS_MONOCHROME=ON"
)

set(_ompt_sample_no_tmpfiles_environ
    "${_ompt_environment}"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_USE_OMPT=OFF"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_USE_PROCESS_SAMPLING=OFF"
    "ROCPROFSYS_SAMPLING_CPUTIME=ON"
    "ROCPROFSYS_SAMPLING_REALTIME=OFF"
    "ROCPROFSYS_SAMPLING_CPUTIME_FREQ=700"
    "ROCPROFSYS_USE_TEMPORARY_FILES=OFF"
    "ROCPROFSYS_MONOCHROME=ON"
)

set(_ompt_sampling_samp_regex
    "Sampler for thread 0 will be triggered 1000.0x per second of CPU-time(.*)Sampler for thread 0 will be triggered 500.0x per second of wall-time(.*)Sampling will be disabled after 0.250000 seconds(.*)Sampling duration of 0.250000 seconds has elapsed. Shutting down sampling"
)
set(_ompt_sampling_file_regex
    "sampling-duration-sampling/sampling_percent.(json|txt)(.*)sampling-duration-sampling/sampling_cpu_clock.(json|txt)(.*)sampling-duration-sampling/sampling_wall_clock.(json|txt)"
)
set(_notmp_sampling_file_regex
    "sampling-no-tmp-files-sampling/sampling_percent.(json|txt)(.*)sampling-no-tmp-files-sampling/sampling_cpu_clock.(json|txt)(.*)sampling-no-tmp-files-sampling/sampling_wall_clock.(json|txt)"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME openmp-cg-sampling-duration
    TARGET openmp-cg
    LABELS "openmp;sampling-duration"
    ENVIRONMENT "${_ompt_sampling_environ}"
    SAMPLING_PASS_REGEX "${_ompt_sampling_samp_regex}(.*)${_ompt_sampling_file_regex}"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME openmp-lu-sampling-duration
    TARGET openmp-lu
    LABELS "openmp;sampling-duration"
    ENVIRONMENT "${_ompt_sampling_environ}"
    SAMPLING_PASS_REGEX "${_ompt_sampling_samp_regex}(.*)${_ompt_sampling_file_regex}"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE
    NAME openmp-cg-sampling-no-tmp-files
    TARGET openmp-cg
    LABELS "openmp;no-tmp-files"
    ENVIRONMENT "${_ompt_sample_no_tmpfiles_environ}"
    SAMPLING_PASS_REGEX "${_notmp_sampling_file_regex}"
)
