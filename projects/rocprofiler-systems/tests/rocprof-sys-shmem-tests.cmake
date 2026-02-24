# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# SHMEM tests (OpenSHMEM examples: shmem_hello, shmem_pingpong)
#
# -------------------------------------------------------------------------------------- #

# Determine if SHMEM tests should be disabled
set(_SHMEM_TESTS_DISABLED FALSE)
set(_SHMEM_DISABLED_REASON "")

if(NOT ROCPROFSYS_BUILD_EXAMPLES)
    set(_SHMEM_TESTS_DISABLED TRUE)
    set(_SHMEM_DISABLED_REASON "ROCPROFSYS_BUILD_EXAMPLES is OFF")
elseif(NOT TARGET shmem_hello)
    set(_SHMEM_TESTS_DISABLED TRUE)
    set(_SHMEM_DISABLED_REASON "shmem_hello target not available (oshcc not found)")
elseif(NOT _VALID_GPU)
    set(_SHMEM_TESTS_DISABLED TRUE)
    set(_SHMEM_DISABLED_REASON "No valid GPU detected")
else()
    find_program(OSHRUN_EXECUTABLE NAMES oshrun)
    if(NOT OSHRUN_EXECUTABLE)
        set(_SHMEM_TESTS_DISABLED TRUE)
        set(_SHMEM_DISABLED_REASON "oshrun not found")
    endif()
endif()

if(_SHMEM_TESTS_DISABLED)
    rocprofiler_systems_message(
        AUTHOR_WARNING "SHMEM tests will be DISABLED: ${_SHMEM_DISABLED_REASON}"
    )
endif()

# Common environment for all SHMEM tests
set(SHMEM_OUTPUT_DIR "${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/shmem-pingpong")

set(_shmem_environment
    "${_base_environment}"
    "ROCPROFSYS_USE_PID=OFF"
    "ROCPROFSYS_USE_SHMEM=ON"
    "ROCPROFSYS_OUTPUT_PATH=${SHMEM_OUTPUT_DIR}"
    "OMPI_MCA_memheap_base_max_segments=64"
)

# Enable ROCPD for SHMEM tests only when GPU is available and ROCPD is enabled
# Note: _VALID_GPU is checked in _SHMEM_TESTS_DISABLED logic above
if(NOT _SHMEM_TESTS_DISABLED AND ${ENABLE_ROCPD_TEST})
    list(APPEND _shmem_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

set(_SHMEM_NP 2)
set(_SHMEM_PINGPONG_ARGS "-n" "1000" "-s" "8")

# Marker file: only present when validation passed. A second test checks it and sets
# the fixture so CTest skips dependent tests when validation was skipped (exit 77).
set(_SHMEM_VALIDATION_MARKER "${PROJECT_BINARY_DIR}/.shmem_validation_passed")

# Paths to executables (set by examples/shmem/CMakeLists.txt when built)
set(_SHMEM_HELLO_EXE "${ROCPROFSYS_SHMEM_HELLO_EXE}")
set(_SHMEM_PINGPONG_EXE "${ROCPROFSYS_SHMEM_PINGPONG_EXE}")

# Validation script: run oshrun -np 2 shmem_hello and check output. Exit 77 = skip.
# On success, creates _SHMEM_VALIDATION_MARKER so shmem-validation-passed can set the fixture.
# Run via bash so the script does not require execute permission (CI-safe).
add_test(
    NAME shmem-validation-check
    COMMAND
        bash ${CMAKE_CURRENT_LIST_DIR}/shmem_validation_check.sh ${OSHRUN_EXECUTABLE}
        ${_SHMEM_NP} ${_SHMEM_HELLO_EXE} ${_SHMEM_VALIDATION_MARKER}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    shmem-validation-check
    PROPERTIES
        LABELS "shmem;validation"
        SKIP_RETURN_CODE 77
        TIMEOUT 60
        ENVIRONMENT "${_shmem_environment}"
        DISABLED ${_SHMEM_TESTS_DISABLED}
)

# Only set the fixture when validation actually passed (marker exists). CTest sets
# fixtures only when this test passes; when it is skipped (no marker), dependent
# tests are skipped instead of running and failing.
add_test(
    NAME shmem-validation-passed
    COMMAND bash -c "test -f '${_SHMEM_VALIDATION_MARKER}' && exit 0 || exit 77"
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    shmem-validation-passed
    PROPERTIES
        LABELS "shmem;validation"
        DEPENDS shmem-validation-check
        FIXTURES_SETUP shmem_available
        SKIP_RETURN_CODE 77
        DISABLED ${_SHMEM_TESTS_DISABLED}
)

# Wrapper: run command only if validation passed; exit 77 otherwise (so CTest skips).
set(_SHMEM_RUN_IF_OK
    bash
    ${CMAKE_CURRENT_LIST_DIR}/run_if_shmem_ok.sh
    ${_SHMEM_VALIDATION_MARKER}
)

# ---- shmem_pingpong: baseline (run only) ----
add_test(
    NAME shmem-pingpong-baseline
    COMMAND
        ${_SHMEM_RUN_IF_OK} ${OSHRUN_EXECUTABLE} -n ${_SHMEM_NP} "${_SHMEM_PINGPONG_EXE}"
        ${_SHMEM_PINGPONG_ARGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    shmem-pingpong-baseline
    PROPERTIES
        LABELS "shmem;pingpong;baseline"
        FIXTURES_REQUIRED "shmem_available;rocprofsys-global-tmp-files"
        RUN_SERIAL ON
        TIMEOUT 120
        SKIP_RETURN_CODE 77
        ENVIRONMENT "${_shmem_environment}"
        DISABLED ${_SHMEM_TESTS_DISABLED}
)

# ---- shmem_pingpong: sampling ----
add_test(
    NAME shmem-pingpong-sampling
    COMMAND
        ${_SHMEM_RUN_IF_OK} ${OSHRUN_EXECUTABLE} -n ${_SHMEM_NP}
        $<TARGET_FILE:rocprofiler-systems-sample> -- "${_SHMEM_PINGPONG_EXE}"
        ${_SHMEM_PINGPONG_ARGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    shmem-pingpong-sampling
    PROPERTIES
        LABELS "shmem;pingpong;sampling"
        FIXTURES_REQUIRED "shmem_available;rocprofsys-global-tmp-files"
        RUN_SERIAL ON
        TIMEOUT 120
        SKIP_RETURN_CODE 77
        ENVIRONMENT "${_shmem_environment}"
        DISABLED ${_SHMEM_TESTS_DISABLED}
)

# ---- shmem_pingpong: sys-run (trace with rocprofiler-systems-run) ----
add_test(
    NAME shmem-pingpong-sys-run
    COMMAND
        ${_SHMEM_RUN_IF_OK} ${OSHRUN_EXECUTABLE} -n ${_SHMEM_NP}
        $<TARGET_FILE:rocprofiler-systems-run> -- "${_SHMEM_PINGPONG_EXE}"
        ${_SHMEM_PINGPONG_ARGS}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_tests_properties(
    shmem-pingpong-sys-run
    PROPERTIES
        LABELS "shmem;pingpong"
        FIXTURES_REQUIRED "shmem_available;rocprofsys-global-tmp-files"
        RUN_SERIAL ON
        TIMEOUT 300
        SKIP_RETURN_CODE 77
        ENVIRONMENT "${_shmem_environment}"
        DISABLED ${_SHMEM_TESTS_DISABLED}
)

# ---- Perfetto validation for shmem-pingpong-sys-run (wrapped so it skips when validation skipped) ----
if(ROCPROFSYS_VALIDATION_PYTHON AND ROCPROFSYS_VALIDATION_PYTHON_PERFETTO EQUAL 0)
    set(_SHMEM_PERFETTO_LABELS shmem_pingpong start_pes)
    set(_SHMEM_PERFETTO_COUNTS 1 1)
    set(_SHMEM_PERFETTO_DEPTHS 0 1)

    add_test(
        NAME validate-shmem-pingpong-sys-run-perfetto
        COMMAND
            ${_SHMEM_RUN_IF_OK} ${ROCPROFSYS_VALIDATION_PYTHON}
            ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py -p -l
            ${_SHMEM_PERFETTO_LABELS} -c ${_SHMEM_PERFETTO_COUNTS} -d
            ${_SHMEM_PERFETTO_DEPTHS} -i ${SHMEM_OUTPUT_DIR}/perfetto-trace.proto -t
            /opt/trace_processor/bin/trace_processor_shell
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    set_tests_properties(
        validate-shmem-pingpong-sys-run-perfetto
        PROPERTIES
            DEPENDS shmem-pingpong-sys-run
            SKIP_RETURN_CODE 77
            TIMEOUT 30
            LABELS "shmem;perfetto;validate"
            FIXTURES_REQUIRED "rocprofsys-global-tmp-files"
            DISABLED ${_SHMEM_TESTS_DISABLED}
    )
endif()

# ---- ROCPD validation for shmem-pingpong-sys-run (wrapped so it skips when validation skipped) ----
if(NOT _SHMEM_TESTS_DISABLED AND ENABLE_ROCPD_TEST AND ROCPROFSYS_VALIDATION_PYTHON)
    set(_shmem_rocpd_validation_rules
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/shmem/validation-rules.json"
    )
    set_property(TEST shmem-pingpong-sys-run APPEND PROPERTY LABELS rocpd)

    add_test(
        NAME validate-shmem-pingpong-sys-run-rocpd
        COMMAND
            ${_SHMEM_RUN_IF_OK} ${ROCPROFSYS_VALIDATION_PYTHON}
            ${CMAKE_CURRENT_LIST_DIR}/validate-rocpd.py -db ${SHMEM_OUTPUT_DIR}/rocpd.db
            --validation-rules ${_shmem_rocpd_validation_rules}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    set_tests_properties(
        validate-shmem-pingpong-sys-run-rocpd
        PROPERTIES
            DEPENDS shmem-pingpong-sys-run
            SKIP_RETURN_CODE 77
            TIMEOUT 30
            LABELS "shmem;rocpd;validate"
            FIXTURES_REQUIRED "rocprofsys-global-tmp-files"
            DISABLED ${_SHMEM_TESTS_DISABLED}
    )
endif()
