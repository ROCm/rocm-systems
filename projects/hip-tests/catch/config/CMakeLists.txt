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
    message(
        FATAL_ERROR
        "PyYAML is required in order to read the configuration file. Please install it"
    )
endif()

if(WIN32)
    set(CONFIG_OS "windows")
    set(CONFIG_ARCH "windows")
else()
    set(CONFIG_OS "linux")

    if(HIP_PLATFORM MATCHES "amd")
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
set(CONFIG_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_config.py)
set(SOURCES_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/check_sources.py)
set(HIP_TEST_CONFIG_HEADER ${CATCH_INCLUDE_BINARY_DIR}/hip_test_config.hh)
set(HIP_TEST_PARAMETERS_HEADER
    ${CATCH_INCLUDE_BINARY_DIR}/hip_test_parameters.hh
)
set(HIP_TESTS_CATEGORIES_YAML ${HIP_TESTS_CONFIGS_DIR}/test_categories.yaml)

file(
    GLOB HIP_TESTS_GROUP_CONFIGS
    CONFIGURE_DEPENDS
    ${HIP_TESTS_CONFIGS_DIR}/*.yaml
)
file(
    GLOB HIP_TESTS_UNIT_CONFIGS
    CONFIGURE_DEPENDS
    ${HIP_TESTS_CONFIGS_DIR}/unit/*.yaml
)

# When building with Address Sanitizer, pass --asan so the parser skips test
# cases that list 'asan' in their 'disabled' field.
set(CONFIG_ASAN_FLAG)
if(NOT ENABLE_SANITIZER STREQUAL "OFF")
    set(CONFIG_ASAN_FLAG "--asan")
endif()

if(ENABLE_YAML_TAGS)
    add_custom_target(
        check_hip_tests_config
        ALL
        COMMAND
            "${Python3_EXECUTABLE}" "${CONFIG_CHECK_SCRIPT}"
            "${HIP_TESTS_CONFIGS_DIR}"
        COMMENT "Validating hip-tests YAML configurations"
    )

    add_custom_target(
        check_hip_tests_sources
        ALL
        COMMAND
            "${Python3_EXECUTABLE}" "${SOURCES_CHECK_SCRIPT}"
            "${HIP_TESTS_CONFIGS_DIR}" "${CMAKE_SOURCE_DIR}"
        COMMENT "Checking hip-tests sources for missing YAML configurations"
    )
endif()

# Compile-time switch: register disabled tests as enabled in ctest.
# OFF (default): disabled tests carry [disabled] tag and are DISABLED TRUE in ctest.
# ON: ctest registers them as normal enabled tests; binary tags are unchanged.
option(
    HIP_TESTS_RUN_DISABLED
    "Register disabled tests as enabled in ctest so plain ctest runs them."
    OFF
)
if(HIP_TESTS_RUN_DISABLED)
    message(
        STATUS
        "HIP_TESTS_RUN_DISABLED=ON: disabled tests will run in plain ctest"
    )
endif()

# Tailor the test tags to the GPU(s) of the build machine: build and run the
# hip_tests_sysinfo probe before generating the test definitions, and disable
# test cases whose 'requires' list names a device feature the probe reports as
# unavailable. Only devices matching the offload targets the tests are built for
# are probed. The resulting binaries are specific to the build machine.
# The probe runs once per build tree; delete hip_tests_features.yaml from the
# config build directory to re-probe.
option(
    HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE
    "Probe the GPU(s) of the build machine and disable tests whose required features are unavailable."
    OFF
)
set(CONFIG_FEATURES_ARGS)
set(CONFIG_FEATURES_DEPENDS)
if(HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE)
    set(HIP_TESTS_FEATURES_YAML
        ${CMAKE_CURRENT_BINARY_DIR}/hip_tests_features.yaml
    )
    message(
        STATUS
        "HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE=ON: tests requiring device features "
        "missing on this machine will be disabled (report: ${HIP_TESTS_FEATURES_YAML})"
    )

    set_source_files_properties(
        sysinfo/hip_tests_sysinfo.cc
        PROPERTIES LANGUAGE ${GPGPU_LANGUAGE}
    )
    add_executable(hip_tests_sysinfo sysinfo/hip_tests_sysinfo.cc)
    # The probe runs from the build tree and is never installed, so it needs a
    # build RPATH that finds the HIP runtime even when the project is configured
    # with CMAKE_BUILD_WITH_INSTALL_RPATH (as TheRock does). The runtime
    # directory is added explicitly because CMake drops it from the RPATH when
    # it is also an implicit link directory of the HIP compiler.
    set_target_properties(
        hip_tests_sysinfo
        PROPERTIES
            LINKER_LANGUAGE ${GPGPU_LANGUAGE}
            BUILD_WITH_INSTALL_RPATH OFF
    )
    if(TARGET hip::amdhip64)
        set_target_properties(
            hip_tests_sysinfo
            PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:hip::amdhip64>"
        )
    endif()
    target_link_libraries(hip_tests_sysinfo PRIVATE ${GPGPU_LINKER_LIBRARIES})

    # Target IDs (e.g. gfx942:xnack+) the tests are built for, taken from the
    # --offload-arch flags. Devices that match none of them are not probed.
    # SPIR-V code is not tied to a device, so it cannot be tailored to one:
    # a SPIR-V target is ignored next to device targets and rejected on its own.
    set(_sysinfo_targets)
    foreach(_offload_arch ${OFFLOAD_ARCH_LIST})
        string(REPLACE "--offload-arch=" "" _target_id "${_offload_arch}")
        list(APPEND _sysinfo_targets "${_target_id}")
    endforeach()
    if(_sysinfo_targets STREQUAL "amdgcnspirv")
        message(
            FATAL_ERROR
            "HIP_TESTS_TAILOR_TAGS_TO_CURRENT_MACHINE is incompatible with "
            "building the tests only for SPIR-V (amdgcnspirv): tests cannot be "
            "tailored to a device they are not built for. Add a device target "
            "or turn the option off."
        )
    endif()
    list(REMOVE_ITEM _sysinfo_targets "amdgcnspirv")
    list(JOIN _sysinfo_targets "," _sysinfo_targets)
    message(
        STATUS
        "hip_tests_sysinfo probes devices matching: ${_sysinfo_targets}"
    )

    add_custom_command(
        OUTPUT ${HIP_TESTS_FEATURES_YAML}
        COMMAND
            hip_tests_sysinfo --targets "${_sysinfo_targets}" --output
            ${HIP_TESTS_FEATURES_YAML}
        DEPENDS hip_tests_sysinfo
        COMMENT "Probing HIP device features of the build machine"
    )
    set(CONFIG_FEATURES_ARGS --features-file ${HIP_TESTS_FEATURES_YAML})
    set(CONFIG_FEATURES_DEPENDS ${HIP_TESTS_FEATURES_YAML})
endif()

# Generate both test definitions and parameter headers from YAML
# - hip_test_config.hh: TEST_CASE macros with tags (levels and test categories)
# - hip_test_parameters.hh: Compile-time constants for level parameters, plus
#   the test category -> levels mapping from test_categories.yaml
add_custom_command(
    OUTPUT ${HIP_TEST_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER}
    COMMAND
        "${Python3_EXECUTABLE}" "${CONFIG_PARSER_SCRIPT}"
        "${HIP_TESTS_CONFIGS_DIR}" "${HIP_PLATFORM}" "${CONFIG_OS}"
        "${CONFIG_ARCH}" "${HIP_TEST_CONFIG_HEADER}"
        "${HIP_TEST_PARAMETERS_HEADER}" --categories
        "${HIP_TESTS_CATEGORIES_YAML}" ${CONFIG_ASAN_FLAG}
        ${CONFIG_FEATURES_ARGS}
    DEPENDS
        ${CONFIG_COMMON_MODULE}
        ${CONFIG_PARSER_SCRIPT}
        ${HIP_TESTS_UNIT_CONFIGS}
        ${HIP_TESTS_GROUP_CONFIGS}
        ${HIP_TESTS_CATEGORIES_YAML}
        ${CONFIG_FEATURES_DEPENDS}
    COMMENT "Generating hip_test_config.hh and hip_test_parameters.hh"
)

# Ensure the generated header files are built before the main target
add_custom_target(
    hip_tests_config
    DEPENDS ${HIP_TEST_CONFIG_HEADER} ${HIP_TEST_PARAMETERS_HEADER}
)
if(ENABLE_YAML_TAGS)
    add_dependencies(
        hip_tests_config
        check_hip_tests_config
        check_hip_tests_sources
    )
endif()
set_source_files_properties(
    ${HIP_TEST_CONFIG_HEADER}
    ${HIP_TEST_PARAMETERS_HEADER}
    PROPERTIES GENERATED TRUE
)
