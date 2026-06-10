#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-avail
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)
# Disable for all sanitizers - Python tests cannot load sanitizer libs dynamically due to
# static TLS block allocation limitations
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR ROCPROFILER_MEMCHECK)
    set(IS_DISABLED ON)
endif()

# Hardware-counter enumeration (info --pmc) does NOT depend on PC sampling support and
# works on any arch whose counters can be queried; only disable it where counters
# literally cannot be queried (gfx906, memcheck).
set(IS_HW_COUNTERS_DISABLED OFF)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$" OR ROCPROFILER_MEMCHECK)
    set(IS_HW_COUNTERS_DISABLED ON)
endif()

# The pmc-check tests below validate a specific counter list that includes GL2C_HIT. GL2C
# is an RDNA (gfx10/11/12) cache block; on CDNA (e.g. gfx9xx / MI300) the equivalent L2
# counter is TCC_HIT, so this exact list cannot be resolved there and pmc-check fails.
# Restrict these checks to the architectures where the listed counters actually exist
# (RDNA, e.g. gfx1150 on WSL2).
#
# Additionally require /dev/kfd: these tests target device 0 (`-d 0` / `device=0`), but on
# WSL the only GPU is enumerated at index 1 (a synthetic CPU agent occupies index 0), so
# `-d 0` resolves to the CPU and pmc-check aborts with "Invalid device id : 0". Gating on
# /dev/kfd skips them on the WSL/dxg path (where hardware counters cannot be collected
# anyway) while leaving them enabled on native Linux with a real KFD device.
set(IS_RDNA_COUNTER_CHECK_DISABLED ON)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   AND NOT ROCPROFILER_MEMCHECK
   AND EXISTS "/dev/kfd")
    set(IS_RDNA_COUNTER_CHECK_DISABLED OFF)
endif()

# PC sampling is not available on navi2/3/4 (gfx10/11/12), gfx906, under memcheck, or when
# arch info is missing (e.g. WSL synth-agent path).
set(IS_PC_SAMPLING_DISABLED OFF)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR "${pc-sampling-gpu-0-gfx-info}" STREQUAL ""
   OR "${pc-sampling-gpu-0-gfx-info}" STREQUAL "pc-sampling-gpu-0-gfx-info-NOTFOUND"
   OR ROCPROFILER_MEMCHECK)
    set(IS_PC_SAMPLING_DISABLED ON)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-hw-counters
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
            --pmc
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;Counter_Name:\\t[a-zA-Z_]*\\n;Description:\\t(.*)\\n*;Expression:\\t(.)*\\n*;Block:\\t[a-zA-Z]*\\n*;Dimensions:\\t([A-Z_]*)\\t([[0-9]*:[0-9]*\\])*\\n*"
    DISABLED "${IS_HW_COUNTERS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-pc-sample-config
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
            --pc-sampling
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests;pc-sampling"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=on"
    PASS_REGULAR_EXPRESSION
        "GPU[:0-9 ]+\\n*;Name[:a-zA-Z_ ]+\\n;configs[: ]+;Method:(.*)\\n*;Unit:(.*)\\n*;Min_Interval:[0-9]*\\n*;Max_Interval:[0-9]*\\n*;Flags:(.*)\\n*"
    DISABLED "${IS_PC_SAMPLING_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-check-hw-counters
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> -d 0
            pmc-check SQ_WAVES GRBM_COUNT GL2C_HIT
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_RDNA_COUNTER_CHECK_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-check-hw-counters_2
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
            pmc-check SQ_WAVES GRBM_COUNT GL2C_HIT:device=0
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_RDNA_COUNTER_CHECK_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-avail-test
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --rocm-path ${rocprofiler-sdk_ROOT_DIR}
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    LABELS "integration-tests"
    DISABLED "${IS_DISABLED}")
