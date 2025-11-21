# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# ----------------------------------------------------------------------------- #
#
# openmp tests
#
# ----------------------------------------------------------------------------- #

if(NOT EXISTS "${ROCM_LLVM_LIB_PATH}/libomptarget.so" AND ROCPROFSYS_USE_ROCM)
    message(
        FATAL_ERROR
        "libomptarget.so not found in \"${ROCM_LLVM_LIB_PATH}\". "
        "Verify that ROCm is installed correctly and that ROCM_PATH "
        "(${ROCM_PATH}) points at the right location."
    )
endif()

set(_ompt_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_TIME_OUTPUT=OFF"
    "ROCPROFSYS_USE_OMPT=ON"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,trip_count,peak_rss"
    "${_test_openmp_env}"
    "${_test_library_path}"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _ompt_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

if(ROCPROFSYS_OPENMP_USING_LIBOMP_LIBRARY AND ROCPROFSYS_USE_OMPT)
    set(_OMPT_PASS_REGEX "\\|_omp_")
    set(_OMPVV_TARGET_PASS_REGEX "_+omp_offloading")
else()
    set(_OMPT_PASS_REGEX "")
    set(_OMPVV_OFFLOAD_PASS_REGEX "")
endif()

rocprofiler_systems_add_test(
    SKIP_RUNTIME
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
      "${_ompt_environment};ROCPROFSYS_ROCM_DOMAINS=hip_api,hsa_api,kernel_dispatch"
)

rocprofiler_systems_add_validation_test(
    NAME openmp-target-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "openmp;openmp-target"
    ARGS
      --label-substrings
      Z4vmulIiEvPT_S1_S1_i_l51.kd
      Z4vmulIfEvPT_S1_S1_i_l51.kd
      Z4vmulIdEvPT_S1_S1_i_l51.kd
      -c 4 4 4
      -d 0 0 0
      -p
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST openmp-target-sampling)
    set_property(TEST openmp-target-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME openmp-target-sampling
        ROCPD_FILE "rocpd.db"
        LABELS "openmp;openmp-target;rocpd"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/openmp-target/kernel-rules.json"
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/openmp-target/sdk-metrics-rules.json"
    )
endif()

# OpenMP tests generated using OMPVV binaries
if(ROCPROFSYS_OMPVV_HOST_TESTS)
    foreach(HOST_TEST_NAME ${ROCPROFSYS_OMPVV_HOST_TESTS})
        rocprofiler_systems_add_test(
            SKIP_RUNTIME
            NAME ${HOST_TEST_NAME}
            TARGET ${HOST_TEST_NAME}-exec
            LABELS "openmp;ompvv"
            REWRITE_ARGS
              -e -v 2 --instrument-loops
            RUNTIME_ARGS
              -e -v 1 --label return args -E ^GOMP
            SAMPLING_TIMEOUT 300
            REWRITE_TIMEOUT 300
            ENVIRONMENT
              "${_ompt_environment};ROCPROFSYS_USE_SAMPLING=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_COUT_OUTPUT=ON"
            REWRITE_RUN_PASS_REGEX "${_OMPT_PASS_REGEX}"
            REWRITE_FAIL_REGEX "0 instrumented loops in procedure"
        )
    endforeach()

    set(_ompvv_offload_environment
        "${_ompt_environment}"
        "ROCPROFSYS_USE_SAMPLING=ON"
        "ROCPROFSYS_SAMPLING_FREQ=50"
        "ROCPROFSYS_COUT_OUTPUT=ON"
        "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory,hsa_api"
    )

    foreach(OFFLOAD_TEST_NAME ${ROCPROFSYS_OMPVV_OFFLOAD_TESTS})
        rocprofiler_systems_add_test(
            SKIP_RUNTIME
            NAME ${OFFLOAD_TEST_NAME}
            TARGET ${OFFLOAD_TEST_NAME}-exec
            GPU ON
            LABELS "openmp;ompvv;openmp-target"
            REWRITE_ARGS -e -v 2
            SAMPLING_TIMEOUT 300
            REWRITE_TIMEOUT 300
            ENVIRONMENT
              "${_ompvv_offload_environment}"
            REWRITE_RUN_PASS_REGEX
              "${_OMPVV_OFFLOAD_PASS_REGEX}"
        )
    endforeach()
endif()

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
