# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Check for PyYAML
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import yaml"
    RESULT_VARIABLE PY_YAML_RESULT
    ERROR_QUIET
)
if(NOT PY_YAML_RESULT EQUAL 0)
    message(FATAL_ERROR "PyYAML is required in order to read the configuration file. Please install it")
endif()

if(WIN32)
    set(CONFIG_OS "windows")
    set(CONFIG_ARCH "windows")
else()
    set(CONFIG_OS "linux")

    # TODO multiple arch support
    foreach(ARCH ${HIP_GPU_ARCH_LIST})
        set(CONFIG_ARCH ${ARCH})
    endforeach()
endif()

set(HIP_TESTS_CONFIGS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/configs)
set(CONFIG_COMMON_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/common.py)
set(CONFIG_PARSER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/parse_config.py)
set(CONFIG_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_config.py)
set(SOURCES_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_sources.py)
set(HIP_TESTS_CONFIG_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_tests_config.hh)
set(HIP_TEST_PARAMETERS_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_test_parameters.hh)

file(GLOB HIP_TESTS_GROUP_CONFIGS ${HIP_TESTS_CONFIGS_DIR}/*.yaml)
file(GLOB HIP_TESTS_UNIT_CONFIGS ${HIP_TESTS_CONFIGS_DIR}/unit/*.yaml)

add_custom_target(check_hip_tests_config ALL
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_CHECK_SCRIPT}" "${HIP_TESTS_CONFIGS_DIR}"
    COMMENT "Validating hip-tests YAML configurations"
)

add_custom_target(check_hip_tests_sources ALL
    COMMAND "${Python3_EXECUTABLE}" "${SOURCES_CHECK_SCRIPT}" "${HIP_TESTS_CONFIGS_DIR}" "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking hip-tests sources for missing YAML configurations"
)

# Generate both test definitions and parameter headers from YAML
# - hip_tests_config.hh: TEST_CASE macros with tags
# - hip_test_parameters.hh: Compile-time constants for level parameters
add_custom_command(
    OUTPUT ${HIP_TESTS_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER}
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_PARSER_SCRIPT}"
            "${HIP_TESTS_CONFIGS_DIR}"
            "${HIP_PLATFORM}"
            "${CONFIG_OS}"
            "${CONFIG_ARCH}"
            "${HIP_TESTS_CONFIG_HEADER}"
            "${HIP_TEST_PARAMETERS_HEADER}"
    DEPENDS ${CONFIG_COMMON_MODULE} ${CONFIG_PARSER_SCRIPT} ${HIP_TESTS_UNIT_CONFIGS} ${HIP_TESTS_GROUP_CONFIGS}
    COMMENT "Generating hip_tests_config.hh and hip_test_parameters.hh"
)

# Ensure the generated header files are built before the main target
add_custom_target(hip_tests_config DEPENDS ${HIP_TESTS_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER})
add_dependencies(hip_tests_config check_hip_tests_config check_hip_tests_sources)
set_source_files_properties(${HIP_TESTS_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER} PROPERTIES GENERATED TRUE)
