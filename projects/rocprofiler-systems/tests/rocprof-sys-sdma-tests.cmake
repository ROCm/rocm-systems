# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# SDMA tests (sdma-test example) - validate SDMA usage metrics in Perfetto and ROCPD
# SDMA usage requires AMD-SMI >= 26.3 (see source/lib/core/amd_smi.hpp).
#
# -------------------------------------------------------------------------------------- #

if(NOT _VALID_GPU)
    message(
        STATUS
        "SDMA tests require a GPU and no valid GPUs were found; skipping SDMA tests"
    )
    return()
endif()

if(NOT TARGET sdma-test)
    message(WARNING "sdma-test target not available; skipping SDMA tests")
    return()
endif()

set(_sdma_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api"
    "ROCPROFSYS_AMD_SMI_METRICS=sdma_usage"
)

set(_sdma_rocpd_validation_rules
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/sdma/validation-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/sdma/amd-smi-rules.json"
)

if(ENABLE_ROCPD_TEST)
    list(APPEND _sdma_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# Check AMD SMI version - sdma_usage metric requires version >= 26.3.0
find_package(amd_smi QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})
if(NOT amd_smi_FOUND OR amd_smi_VERSION VERSION_LESS "26.3.0")
    if(amd_smi_FOUND)
        message(
            WARNING
            "SDMA tests require AMD-SMI >= 26.3 (found: ${amd_smi_VERSION}); skipping SDMA tests"
        )
    else()
        message(WARNING "amd_smi not found; skipping SDMA tests")
    endif()
    return()
endif()

# Short run: 2 iterations, 64 MB to keep test time reasonable
set(_sdma_run_args "-n" "2" "-s" "64")

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME sdma-test
    TARGET sdma-test
    GPU ON
    ENVIRONMENT "${_base_environment};${_sdma_environment}"
    LABELS "sdma;amd-smi"
    RUN_ARGS ${_sdma_run_args}
    SYS_RUN_TIMEOUT 120
)

# Perfetto validation: require "SDMA Usage" counter track (GPU [N] SDMA Usage (S))
rocprofiler_systems_add_validation_test(
    NAME sdma-test-sys-run
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "sdma;perfetto"
    ARGS --counter-names "SDMA Usage" -p
)
if(TEST validate-sdma-test-sys-run-perfetto)
    set_property(
        TEST validate-sdma-test-sys-run-perfetto
        APPEND
        PROPERTY FIXTURES_REQUIRED rocprofsys-global-tmp-files
    )
endif()

# ROCPD validation: device_sdma_usage in rocpd_info_pmc and rocpd_pmc_event
if(ENABLE_ROCPD_TEST AND TEST sdma-test-sys-run)
    set_property(TEST sdma-test-sys-run APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME sdma-test-sys-run
        ROCPD_FILE "rocpd.db"
        LABELS "sdma;rocpd"
        ARGS --validation-rules ${_sdma_rocpd_validation_rules}
    )

    if(TEST validate-sdma-test-sys-run-rocpd)
        set_property(
            TEST validate-sdma-test-sys-run-rocpd
            APPEND
            PROPERTY FIXTURES_REQUIRED rocprofsys-global-tmp-files
        )
    endif()
endif()
