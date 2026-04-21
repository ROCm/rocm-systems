# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# Unified memory profiling tests
#
# -------------------------------------------------------------------------------------- #

set(_unified_memory_environment
    "${_base_environment}"
    "HSA_XNACK=1"
    "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _unified_memory_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME unified-memory-basic-output
    TARGET unified-memory
    GPU ON
    REWRITE_RUN_ARGS
        -e
        ROCPROFSYS_USE_ROCPD=ON
        ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON
        --
    ENVIRONMENT "${_unified_memory_environment}"
    LABELS "unified-memory" "e2e"
    TIMEOUT 300
)

# -------------------------------------------------------------------------------------- #
#
# Unified memory output validation
#
# -------------------------------------------------------------------------------------- #

if(TEST unified-memory-basic-output-sampling)
    add_test(
        NAME unified-memory-validate-structure
        COMMAND
            ${ROCPROFSYS_VALIDATION_PYTHON}
            ${CMAKE_CURRENT_LIST_DIR}/validate-unified-memory.py --output-dir
            ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/unified-memory-basic-output-sampling
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    set_tests_properties(
        unified-memory-validate-structure
        PROPERTIES
            DEPENDS unified-memory-basic-output-sampling
            LABELS "unified-memory;e2e;validation"
            TIMEOUT 60
    )
endif()

# -------------------------------------------------------------------------------------- #
#
# ROCpd validation tests (XNACK-capable GPUs only — KFD events require XNACK)
#
# -------------------------------------------------------------------------------------- #

# KFD page migration/fault events are only generated on XNACK-capable GPUs
check_rocminfo("xnack" _XNACK_SUPPORTED)

if(
    _XNACK_SUPPORTED
    AND ${ENABLE_ROCPD_TEST}
    AND ${_VALID_GPU}
    AND TEST unified-memory-basic-output-sampling
)
    set_property(TEST unified-memory-basic-output-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME unified-memory-basic-output-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/unified-memory/rules.json"
        LABELS "unified-memory;rocpd"
    )
endif()
