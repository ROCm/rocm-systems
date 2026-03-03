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

set(HIP_TESTS_CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CONFIG_COMMON_MODULE ${HIP_TESTS_CONFIG_DIR}/common.py)
set(CONFIG_PARSER_SCRIPT ${HIP_TESTS_CONFIG_DIR}/parse_config.py)
set(CONFIG_CHECK_SCRIPT ${HIP_TESTS_CONFIG_DIR}/check_config.py)
set(SOURCES_CHECK_SCRIPT ${HIP_TESTS_CONFIG_DIR}/check_sources.py)
set(HIP_TESTS_CONFIG_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_tests_config.hh)

file(GLOB HIP_TESTS_UNIT_CONFIGS ${HIP_TESTS_CONFIG_DIR}/configs/unit/*.yaml)
file(GLOB HIP_TESTS_GROUP_CONFIGS ${HIP_TESTS_CONFIG_DIR}/configs/*.yaml)
file(GLOB_RECURSE HIP_TESTS_SOURCES ${CMAKE_SOURCE_DIR}/unit/*.cc
                                     ${CMAKE_SOURCE_DIR}/ABM/*.cc
                                     ${CMAKE_SOURCE_DIR}/multiproc/*.cc
                                     ${CMAKE_SOURCE_DIR}/performance/*.cc
                                     ${CMAKE_SOURCE_DIR}/perftests/*.cc
                                     ${CMAKE_SOURCE_DIR}/stress/*.cc
                                     ${CMAKE_SOURCE_DIR}/TypeQualifiers/*.cc)

add_custom_target(check_hip_tests_config ALL
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_CHECK_SCRIPT}" "${HIP_TESTS_CONFIG_DIR}"
    COMMENT "Validating hip-tests YAML configurations"
)

add_custom_target(check_hip_tests_sources ALL
    COMMAND "${Python3_EXECUTABLE}" "${SOURCES_CHECK_SCRIPT}" "${HIP_TESTS_CONFIG_DIR}"
    COMMENT "Checking hip-tests sources for missing YAML configurations"
)

add_custom_command(
    OUTPUT ${HIP_TESTS_CONFIG_HEADER}
    COMMAND "${Python3_EXECUTABLE}" "${CONFIG_PARSER_SCRIPT}" "${HIP_TESTS_CONFIG_DIR}" "${HIP_PLATFORM}" "${CONFIG_OS}" "${CONFIG_ARCH}" "${HIP_TESTS_CONFIG_HEADER}"
    DEPENDS ${CONFIG_COMMON_MODULE} ${CONFIG_PARSER_SCRIPT} ${HIP_TESTS_UNIT_CONFIGS} ${HIP_TESTS_GROUP_CONFIGS}
    COMMENT "Generating hip_tests_config.hh"
)

# Ensure the generated header file is built before the main target
add_custom_target(hip_tests_config DEPENDS ${HIP_TESTS_CONFIG_HEADER})
add_dependencies(hip_tests_config check_hip_tests_config check_hip_tests_sources)
set_source_files_properties(${HIP_TESTS_CONFIG_HEADER} PROPERTIES GENERATED TRUE)
