#
# Integration test: rocprofiler-sdk OMPT keep-or-defer role decision.
#
# Covers the three either/or outcomes of the SDK's ompt_start_tool() logic with
# a minimal SDK client (coexist-client, whose OMPT appetite is toggled by
# COEXIST_OMPT) and a mock "second" OMPT tool (coexist-mock):
#
#   keep       - a client requests OMPT -> the SDK keeps the OMPT tool role and
#                the client collects OMPT records.
#   defer      - no client requests OMPT -> the SDK defers; the mock OMPT tool
#                (via OMP_TOOL_LIBRARIES) becomes the runtime's OMPT tool while
#                the SDK keeps doing its own kernel-dispatch tracing.
#   defer-back - the mock is bound first and returns rocprofiler_ompt_start_tool()
#                to hand the role back to the SDK, which arms OMPT for the client.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-openmp-tools-coexist
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# disable when GPU-0 is navi2, navi3, and navi4 (no target offload there)
list(GET rocprofiler-sdk-tests-gfx-info 0 openmp-tools-coexist-gpu-0-gfx-info)
if("${openmp-tools-coexist-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

# the minimal SDK client whose OMPT appetite drives the keep-vs-defer decision
add_library(openmp-tools-coexist-client SHARED coexist_client.cpp)
target_link_libraries(
    openmp-tools-coexist-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                        rocprofiler-sdk::tests-build-flags)
target_compile_features(openmp-tools-coexist-client PRIVATE cxx_std_17)
set_target_properties(
    openmp-tools-coexist-client PROPERTIES POSITION_INDEPENDENT_CODE ON
                                           INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

# the mock "second" OMPT tool (TAU/Score-P stand-in)
add_library(openmp-tools-coexist-mock SHARED coexist_ompt_tool.cpp)
target_include_directories(
    openmp-tools-coexist-mock SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:rocprofiler-sdk::rocprofiler-sdk,INTERFACE_INCLUDE_DIRECTORIES>)
# links dl for dlsym(3); CMAKE_DL_LIBS is empty where libdl is folded into libc
target_link_libraries(openmp-tools-coexist-mock PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(openmp-tools-coexist-mock PRIVATE cxx_std_17)
set_target_properties(openmp-tools-coexist-mock PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                           CXX_VISIBILITY_PRESET hidden)

set(_common-env
    "OMP_NUM_THREADS=2"
    "OMP_DISPLAY_ENV=1"
    "OMP_TARGET_OFFLOAD=mandatory"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# ---- case 1: keep (a client requests OMPT) ---------------------------------
set(keep-env ${_common-env} "COEXIST_OMPT=1"
             "COEXIST_CLIENT_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/coexist-keep-client.json")

rocprofiler_add_integration_execute_test(
    openmp-tools-coexist-keep
    TARGET openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    PRELOAD "$<TARGET_FILE:openmp-tools-coexist-client>" "${PRELOAD_ENV}"
    ENVIRONMENT "${keep-env}"
    DISABLED "${IS_DISABLED}"
    FIXTURES_SETUP openmp-tools-coexist-keep)

rocprofiler_add_integration_validate_test(
    openmp-tools-coexist-keep
    TEST_PATHS validate_keep.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;openmp-target;ompt-coexist"
    TIMEOUT 45
    TARGET_DEPENDS openmp-tools-coexist-client
    FIXTURES_REQUIRED openmp-tools-coexist-keep
    DISABLED "${IS_DISABLED}"
    ARGS --client-input ${CMAKE_CURRENT_BINARY_DIR}/coexist-keep-client.json)

# ---- case 2: defer (no client requests OMPT; mock becomes the tool) --------
set(defer-env
    ${_common-env}
    "COEXIST_MOCK_MODE=own"
    "OMP_TOOL_LIBRARIES=$<TARGET_FILE:openmp-tools-coexist-mock>"
    "COEXIST_MOCK_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/coexist-defer-mock.json"
    "COEXIST_CLIENT_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/coexist-defer-client.json")

rocprofiler_add_integration_execute_test(
    openmp-tools-coexist-defer
    TARGET openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    PRELOAD "$<TARGET_FILE:openmp-tools-coexist-client>" "${PRELOAD_ENV}"
    ENVIRONMENT "${defer-env}"
    DISABLED "${IS_DISABLED}"
    FIXTURES_SETUP openmp-tools-coexist-defer)

rocprofiler_add_integration_validate_test(
    openmp-tools-coexist-defer
    TEST_PATHS validate_defer.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;openmp-target;ompt-coexist"
    TIMEOUT 45
    TARGET_DEPENDS openmp-tools-coexist-client openmp-tools-coexist-mock
    FIXTURES_REQUIRED openmp-tools-coexist-defer
    DISABLED "${IS_DISABLED}"
    ARGS --client-input ${CMAKE_CURRENT_BINARY_DIR}/coexist-defer-client.json
         --mock-input ${CMAKE_CURRENT_BINARY_DIR}/coexist-defer-mock.json)

# ---- case 3: defer-back (mock bound first hands the role back to the SDK) ---
# The mock is preloaded BEFORE the client so the OpenMP runtime binds the mock's
# ompt_start_tool; in mode=defer it returns rocprofiler_ompt_start_tool().
set(deferback-env
    ${_common-env} "COEXIST_OMPT=1" "COEXIST_MOCK_MODE=defer"
    "COEXIST_MOCK_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/coexist-deferback-mock.json"
    "COEXIST_CLIENT_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/coexist-deferback-client.json")

rocprofiler_add_integration_execute_test(
    openmp-tools-coexist-deferback
    TARGET openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    PRELOAD "$<TARGET_FILE:openmp-tools-coexist-mock>"
            "$<TARGET_FILE:openmp-tools-coexist-client>" "${PRELOAD_ENV}"
    ENVIRONMENT "${deferback-env}"
    DISABLED "${IS_DISABLED}"
    FIXTURES_SETUP openmp-tools-coexist-deferback)

rocprofiler_add_integration_validate_test(
    openmp-tools-coexist-deferback
    TEST_PATHS validate_deferback.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests;openmp-target;ompt-coexist"
    TIMEOUT 45
    TARGET_DEPENDS openmp-tools-coexist-client openmp-tools-coexist-mock
    FIXTURES_REQUIRED openmp-tools-coexist-deferback
    DISABLED "${IS_DISABLED}"
    ARGS --client-input ${CMAKE_CURRENT_BINARY_DIR}/coexist-deferback-client.json
         --mock-input ${CMAKE_CURRENT_BINARY_DIR}/coexist-deferback-mock.json)
