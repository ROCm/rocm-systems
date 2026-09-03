#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-avail
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

# Absence of /dev/kfd indicates a WSL2/DXG (or no-KFD) environment; several tests below
# gate on it. See rocprofiler_sdk_kfd_available in the rocprofiler-sdk-utilities module.
rocprofiler_sdk_kfd_available(_KFD_AVAILABLE)

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# disable when GPU-0 is navi2, navi3, and navi4
list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)
# Disable for all sanitizers - Python tests cannot load sanitizer libs dynamically due to
# static TLS block allocation limitations On WSL (no /dev/kfd) gfx-info[0] is a synthetic
# CPU agent, so the arch regex above misses and these tests would run and fail; disable
# them there so WSL matches the real gfx11-class + KFD behavior (where the arch match
# already disables them).
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR ROCPROFILER_MEMCHECK
   OR NOT _KFD_AVAILABLE)
    set(IS_DISABLED ON)
endif()

# Hardware-counter enumeration (info --pmc) does NOT depend on PC sampling support and
# works on any arch whose counters can be queried; only disable it where counters
# literally cannot be queried (gfx906, memcheck).
set(IS_HW_COUNTERS_DISABLED OFF)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$" OR ROCPROFILER_MEMCHECK)
    set(IS_HW_COUNTERS_DISABLED ON)
endif()

# The pmc-check tests below validate a concrete counter list, and its L2-cache counter is
# spelled per-family: CDNA/GCN (gfx9xx) defines TCC_HIT and RDNA (gfx10/11/12) defines
# GL2C_HIT. See source/share/rocprofiler-sdk/config.yaml, which is the file the SDK loads
# at run time. Emitting the wrong one makes the list unresolvable and pmc-check fails, so
# select the name that GPU-0 actually defines.
set(COUNTER_CHECK_L2_HIT "")
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx9[0-9a-f][0-9a-f]$"
   AND NOT "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$")
    # gfx906 is excluded for the same reason as IS_HW_COUNTERS_DISABLED above: its
    # counters cannot be queried.
    set(COUNTER_CHECK_L2_HIT TCC_HIT)
elseif("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(COUNTER_CHECK_L2_HIT GL2C_HIT)
endif()

set(IS_COUNTER_CHECK_DISABLED ON)
if(COUNTER_CHECK_L2_HIT AND NOT ROCPROFILER_MEMCHECK)
    set(IS_COUNTER_CHECK_DISABLED OFF)
else()
    # Nothing selected the counter, so the tests below are disabled; still name one, so a
    # disabled test is registered with a well-formed command rather than a bare
    # ":device=0".
    set(COUNTER_CHECK_L2_HIT TCC_HIT)
endif()

# PC sampling is not available on navi2/3/4 (gfx10/11/12), gfx906, under memcheck, or when
# arch info is missing (e.g. WSL synth-agent path). It also fundamentally requires
# /dev/kfd, so disable it when that device is absent (e.g. WSL/dxg), otherwise it fails
# with "Cannot open /dev/kfd for pc sampling".
set(IS_PC_SAMPLING_DISABLED OFF)
if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$"
   OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx906$"
   OR "${pc-sampling-gpu-0-gfx-info}" STREQUAL ""
   OR "${pc-sampling-gpu-0-gfx-info}" STREQUAL "pc-sampling-gpu-0-gfx-info-NOTFOUND"
   OR ROCPROFILER_MEMCHECK
   OR NOT _KFD_AVAILABLE)
    set(IS_PC_SAMPLING_DISABLED ON)
endif()

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
if(${IS_DISABLED})
    set(IS_SPM_DISABLED on)
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
    rocprofv3-avail-spm-test-hw-counters
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
            --spm
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION
        "GPU:[0-9]*\\n*;Name:\\t[a-zA-Z_]*\\n;Counter_Name:\\t[a-zA-Z_]*\\n;Description:\\t(.*)\\n*;Expression:\\t(.)*\\n*;Block:\\t[a-zA-Z]*\\n*;Dimensions:\\t([A-Z_]*)\\t([[0-9]*:[0-9]*\\])*\\n*"
    DISABLED "${IS_SPM_DISABLED}")

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
            pmc-check SQ_WAVES GRBM_COUNT ${COUNTER_CHECK_L2_HIT}
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_COUNTER_CHECK_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-check-hw-counters_2
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail>
            pmc-check SQ_WAVES GRBM_COUNT ${COUNTER_CHECK_L2_HIT}:device=0
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${IS_COUNTER_CHECK_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-info
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> info
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "GPU:[0-9]*\\n*"
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_execute_test(
    rocprofv3-avail-test-list
    COMMAND ${Python3_EXECUTABLE} $<TARGET_FILE:rocprofiler-sdk::rocprofv3-avail> list
    DEPENDS rocprofiler-sdk::rocprofv3-avail
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    # `list` output is column-padded ("GPU                           : 0"), so tolerate
    # whitespace around the colon (unlike `info`, which prints a compact "GPU:0").
    PASS_REGULAR_EXPRESSION "GPU[ \t]*:[ \t]*[0-9]*\\n*"
    DISABLED "${IS_DISABLED}")

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
