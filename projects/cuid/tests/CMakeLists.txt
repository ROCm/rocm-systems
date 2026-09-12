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

# Sources listed explicitly, as lib/CMakeLists.txt does. aux_source_directory()
# is evaluated at configure time, so adding a file did not regenerate the build
# and failed at link until cmake was re-run by hand.
set(TEST_SOURCES_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_base.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cc
)

set(TEST_SOURCES_UNIT
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/acpi_parser_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/concurrency_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/cuid_gpu_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/device_type_table_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/driver_cuid_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/file_lock_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/gim_util_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/id_string_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/key_info_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/pci_util_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/record_generation_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/serial_precedence_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/sha256_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/status_string_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/utilities_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/vectors_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/version_read_test.cc
)

set(TEST_SOURCES_FUNCTIONAL
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_handles_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_query_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_refresh_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/hmac_test.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/rekey_test.cc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/file_lock_test.h
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
    ${CMAKE_CURRENT_SOURCE_DIR}/functional/device_refresh_test.h
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

# Fail the build if cuid_vectors.txt has drifted from its generator, the same
# way check_sha256_drift.py guards the SHA-256 sources.
find_package(Python3 COMPONENTS Interpreter QUIET)
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

# Register two CTest targets split by privilege level.
add_test(NAME cuid_unprivileged_tests COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstUnprivileged.*)
add_test(NAME cuid_privileged_tests COMMAND ${AMDCUID_TST_EXECUTABLE} --gtest_filter=cuidtstPrivileged.*)

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
