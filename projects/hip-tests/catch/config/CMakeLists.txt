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

    if (HIP_PLATFORM MATCHES "amd")
        # TODO multiple arch support
        foreach(ARCH ${HIP_GPU_ARCH_LIST})
            set(CONFIG_ARCH ${ARCH})
        endforeach()
    else()
        set(CONFIG_ARCH "linux")
    endif()
endif()

set(HIP_TESTS_CONFIGS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/configs)
set(CONFIG_COMMON_MODULE ${CMAKE_CURRENT_SOURCE_DIR}/common.py)
set(CONFIG_PARSER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/parse_config.py)
set(CONFIG_INJECT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/inject_category_tags.py)
set(CONFIG_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_config.py)
set(SOURCES_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_sources.py)
set(HIP_TEST_CONFIG_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_test_config.hh)
set(HIP_TEST_PARAMETERS_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_test_parameters.hh)
# test_categories.yaml drives tier and GPU-exclusion tag injection.
# Listed in DEPENDS so any change to it triggers a full header rebuild.
set(HIP_TEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories.yaml")

file(GLOB HIP_TESTS_GROUP_CONFIGS ${HIP_TESTS_CONFIGS_DIR}/*.yaml)
file(GLOB HIP_TESTS_UNIT_CONFIGS ${HIP_TESTS_CONFIGS_DIR}/unit/*.yaml)

# When building with Address Sanitizer, pass --asan so the parser skips test
# cases that list 'asan' in their 'disabled' field.
set(CONFIG_ASAN_FLAG)
if(ENABLE_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    set(CONFIG_ASAN_FLAG "--asan")
endif()

if(ENABLE_YAML_TAGS)
    add_custom_target(check_hip_tests_config ALL
        COMMAND "${Python3_EXECUTABLE}" "${CONFIG_CHECK_SCRIPT}" "${HIP_TESTS_CONFIGS_DIR}"
        COMMENT "Validating hip-tests YAML configurations"
    )

    add_custom_target(check_hip_tests_sources ALL
        COMMAND "${Python3_EXECUTABLE}" "${SOURCES_CHECK_SCRIPT}" "${HIP_TESTS_CONFIGS_DIR}" "${CMAKE_SOURCE_DIR}"
        COMMENT "Checking hip-tests sources for missing YAML configurations"
    )
endif()

# Step 1: parse_config.py generates hip_test_config.hh (basic group / level /
#         per-test tags) and hip_test_parameters.hh.  Untouched from upstream.
# Step 2: inject_category_tags.py post-processes hip_test_config.hh in-place,
#         appending tier (quick / standard / comprehensive / full) and
#         GPU-exclusion (ex_gpu_*) tags derived from test_categories.yaml.
#         Any change to test_categories.yaml triggers a full rebuild via DEPENDS.
add_custom_command(
    OUTPUT ${HIP_TEST_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER}
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_PARSER_SCRIPT}"
            "${HIP_TESTS_CONFIGS_DIR}"
            "${HIP_PLATFORM}"
            "${CONFIG_OS}"
            "${CONFIG_ARCH}"
            "${HIP_TEST_CONFIG_HEADER}"
            "${HIP_TEST_PARAMETERS_HEADER}"
            ${CONFIG_ASAN_FLAG}
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_INJECT_SCRIPT}"
            "${HIP_TEST_CONFIG_HEADER}"
            "${HIP_TEST_CATEGORIES_YAML}"
    DEPENDS
        ${CONFIG_COMMON_MODULE}
        ${CONFIG_PARSER_SCRIPT}
        ${CONFIG_INJECT_SCRIPT}
        ${HIP_TESTS_UNIT_CONFIGS}
        ${HIP_TESTS_GROUP_CONFIGS}
        ${HIP_TEST_CATEGORIES_YAML}
    COMMENT "Generating and categorising hip_test_config.hh"
)

# Ensure the generated header files are built before the main target
add_custom_target(hip_tests_config DEPENDS ${HIP_TEST_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER})
if(ENABLE_YAML_TAGS)
    add_dependencies(hip_tests_config check_hip_tests_config check_hip_tests_sources)
endif()
set_source_files_properties(${HIP_TEST_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER} PROPERTIES GENERATED TRUE)
