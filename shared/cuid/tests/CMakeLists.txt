# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                     Cmake amdcuid test                            ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

# Find required packages
find_package(GTest CONFIG QUIET)
# fallback to CMake module if GTest is not found via config
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()
# If GTest is still not found, fetch it from source.
# FetchContent defines GTest::gtest / GTest::gtest_main; find_package CONFIG
# mode does the same. find_package MODULE mode defines GTest::GTest / GTest::Main,
# so create aliases to normalize all three paths to a single target name.
if(NOT GTest_FOUND)
    message("GTest not found on system. Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.16.0)
    FetchContent_MakeAvailable(googletest)
elseif(TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    # MODULE mode found GTest::GTest — create the canonical aliases expected below.
    add_library(GTest::gtest ALIAS GTest::GTest)
endif()

set(AMDCUID_TST_EXECUTABLE amdcuid_test)

# Sources listed explicitly so adding a file re-runs configure.
set(TEST_SOURCES_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cc
)

set(TEST_SOURCES_UNIT
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/acpi_parser_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/concurrency_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/cpu_unit_id_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/cuid_gpu_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/device_type_table_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/driver_cuid_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gim_util_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/id_string_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/key_store_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lookup_policy_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/partition_cuid_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/pci_util_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/serial_precedence_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/sriov_vf_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/sha256_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/status_string_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/utilities_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/vectors_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/version_read_test.cc
)

set(TEST_SOURCES_FUNCTIONAL
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_handles_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_query_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/hmac_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/reverse_lookup_test.cc
)

# Headers are listed so they show up in IDE project trees, as lib/ does.
set(TEST_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_base.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/acpi_parser_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/concurrency_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/cuid_gpu_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/driver_cuid_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gim_util_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/id_string_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/pci_util_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/sha256_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/status_string_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/utilities_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/vectors_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/version_read_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_handles_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_query_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/hmac_test.h
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/reverse_lookup_test.h
)

set(TEST_SOURCES ${TEST_SOURCES_ROOT} ${TEST_SOURCES_UNIT} ${TEST_SOURCES_FUNCTIONAL} ${TEST_HEADERS})

# Enable testing
enable_testing()

# Test executable
add_executable(${AMDCUID_TST_EXECUTABLE} ${TEST_SOURCES})

target_include_directories(
    ${AMDCUID_TST_EXECUTABLE}
    PRIVATE
        # Allows #include "unit/version_read_test.h" etc. from main.cc and test_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Allows #include "include/amd_cuid.h" and #include "src/cuid_util.h" etc.
        ${CMAKE_SOURCE_DIR}/lib
)

# concurrency_test.cc spawns std::thread, so the suite needs a threading
# implementation of its own.
find_package(Threads REQUIRED)

# rocm-sha256 explicitly: sha256_test.cc tests that component directly, and
# libamdcuid folds its objects in privately rather than re-exporting it, so the
# header is not reachable through amdcuid_static. The symbols are already in the
# archive; this is for the include path.
target_link_libraries(${AMDCUID_TST_EXECUTABLE} amdcuid_static rocm::sha256 GTest::gtest Threads::Threads)

# The conformance vectors are read at run time rather than compiled in, so the
# generator stays the single source of truth. This is the last of the three
# paths vectors_test.cc tries, after $AMDCUID_VECTORS_PATH and the copy
# installed beside the binary, and the only one that names the build machine's
# source tree.
target_compile_definitions(
    ${AMDCUID_TST_EXECUTABLE}
    PRIVATE AMDCUID_VECTORS_FILE="${CMAKE_CURRENT_SOURCE_DIR}/vectors/cuid_vectors.txt"
)

option(AMDCUID_BUILD_LIFECYCLE_TESTS "Build isolated Linux lifecycle fixtures" OFF)

if(AMDCUID_BUILD_LIFECYCLE_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
endif()
if(Python3_Interpreter_FOUND)
    add_custom_target(
        amdcuid_vectors_check
        ALL
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/vectors/cuid_vectors.py --check
        COMMENT "Checking CUID conformance vectors are up to date"
        VERBATIM
    )
    add_dependencies(${AMDCUID_TST_EXECUTABLE} amdcuid_vectors_check)
endif()

# These binaries replace discovery at link time. They are never installed and
# no override is compiled into production objects.
if(AMDCUID_BUILD_LIFECYCLE_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    get_target_property(_cuid_sources amdcuid_objs SOURCES)
    add_library(cuid_lifecycle_lib STATIC ${_cuid_sources})
    set(_fixture "${CMAKE_CURRENT_BINARY_DIR}/lifecycle-state")
    target_include_directories(cuid_lifecycle_lib PUBLIC ../lib)
    target_link_libraries(cuid_lifecycle_lib PUBLIC rocm::sha256 Threads::Threads)

    # These tests mount a private synthetic /sys, /dev and /run before running
    # the real GPU discovery code. The shim refuses the host namespace.
    foreach(_path_target IN ITEMS cuid_gpu_path_probe cuid_gpu_path_baseline)
        if(_path_target STREQUAL "cuid_gpu_path_probe")
            set(_path_sources lifecycle/gpu_path_probe.cc)
        else()
            set(_path_sources ${TEST_SOURCES})
        endif()
        add_executable(${_path_target} ${_path_sources} lifecycle/gpu_path_shim.cc)
        target_include_directories(${_path_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(
            ${_path_target}
            PRIVATE
                CUID_TEST_ROOT="${_fixture}"
                AMDCUID_VECTORS_FILE="${CMAKE_CURRENT_SOURCE_DIR}/vectors/cuid_vectors.txt"
        )
        target_link_libraries(${_path_target} PRIVATE cuid_lifecycle_lib GTest::gtest)
        target_link_options(
            ${_path_target}
            PRIVATE
                -Wl,--wrap=stat
                -Wl,--wrap=fstat
                -Wl,--wrap=write
                -Wl,--wrap=_ZN7CuidCpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE
                -Wl,--wrap=_ZN7CuidNic8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE
                -Wl,--wrap=_ZN7CuidNpu8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE
                -Wl,--wrap=_ZN12CuidPlatform8discoverERSt6vectorISt10shared_ptrI10CuidDeviceESaIS3_EE
        )
    endforeach()
    add_test(
        NAME cuid_gpu_paths
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lifecycle/test_gpu_paths.py ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        cuid_gpu_paths
        PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180 RESOURCE_LOCK cuid_lifecycle_state
    )
endif()

# Register the privilege suites and the CPU/partition/VF regression cases.
add_test(NAME cuid_unprivileged_tests COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstUnprivileged.*)
add_test(NAME cuid_privileged_tests COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstPrivileged.*)
add_test(NAME cuid_regression_tests COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtst.*)

# Platform-specific config directory
if(WIN32)
    set(TESTS_INSTALL_DIR "TESTS") # Relative to prefix
else()
    set(TESTS_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/${AMDCUID}/tests")
endif()

# Install the test binary
install(TARGETS ${AMDCUID_TST_EXECUTABLE} DESTINATION ${TESTS_INSTALL_DIR} COMPONENT ${TESTS_COMPONENT})

# ... and the conformance vectors beside it. The test resolves them relative to
# its own executable, so an installed binary needs no source tree.
install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vectors/cuid_vectors.txt
    DESTINATION ${TESTS_INSTALL_DIR}
    COMPONENT ${TESTS_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake amdcuid_test                    ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
