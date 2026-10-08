# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(amdsmi_mock_backend LANGUAGES C CXX)

# Configure this directory directly: no ROCm, libdrm, or GPU driver is required.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
set(AMDSMI_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
file(STRINGS "${AMDSMI_INCLUDE}/amd_smi/amdsmi.h" VERSION_LINE REGEX "^#define AMDSMI_LIB_VERSION_MAJOR [0-9]+")
string(REGEX MATCH "[0-9]+$" ABI_MAJOR "${VERSION_LINE}")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/unsupported.cc"
    COMMAND
        "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_stubs.py" --compiler "${CMAKE_C_COMPILER}"
        --header "${AMDSMI_INCLUDE}/amd_smi/amdsmi.h" --implementation "${CMAKE_CURRENT_SOURCE_DIR}/mock.cc" --output
        "${CMAKE_CURRENT_BINARY_DIR}/unsupported.cc"
    DEPENDS generate_stubs.py mock.cc "${AMDSMI_INCLUDE}/amd_smi/amdsmi.h"
    VERBATIM
)
add_library(amdsmi_mock SHARED mock.cc "${CMAKE_CURRENT_BINARY_DIR}/unsupported.cc")
target_include_directories(amdsmi_mock PUBLIC "${AMDSMI_INCLUDE}")
target_compile_features(amdsmi_mock PUBLIC cxx_std_17)
target_compile_options(amdsmi_mock PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)
target_link_libraries(amdsmi_mock PRIVATE Threads::Threads)
# Deliberately no install rule: never replace a production library by accident.
set_target_properties(amdsmi_mock PROPERTIES OUTPUT_NAME amd_smi SOVERSION "${ABI_MAJOR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(amdsmi_mock PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.map")
    set_property(TARGET amdsmi_mock APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/version.map")
endif()

add_executable(mock_backend_probe probe.c)
target_link_libraries(mock_backend_probe PRIVATE amdsmi_mock)
target_compile_options(mock_backend_probe PRIVATE -Wall -Wextra -Werror)

include(CTest)
if(BUILD_TESTING)
    add_executable(mock_backend_test mock_backend_test.cc)
    target_link_libraries(mock_backend_test PRIVATE amdsmi_mock Threads::Threads)
    target_compile_options(mock_backend_test PRIVATE -Wall -Wextra -Werror)
    add_test(NAME mock_backend_contract COMMAND mock_backend_test)
    add_test(
        NAME mock_backend_control
        COMMAND
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_control.py" "$<TARGET_FILE:amdsmi_mock>"
            "$<TARGET_FILE:mock_backend_probe>"
    )
endif()
