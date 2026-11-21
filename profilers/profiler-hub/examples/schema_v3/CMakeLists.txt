# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(PROFILER_HUB_SCHEMA_V3_EXAMPLES
    memory_alloc_writer
    kernel_dispatch_writer
    memory_copy_writer
    pmc_event_writer
    region_writer
)
set(PROFILER_HUB_SCHEMA_V3_EXAMPLE_TARGETS)

foreach(_example IN LISTS PROFILER_HUB_SCHEMA_V3_EXAMPLES)
    set(_target profiler-hub_${_example})
    list(APPEND PROFILER_HUB_SCHEMA_V3_EXAMPLE_TARGETS ${_target})

    add_executable(${_target} ${_example}.cpp)

    target_link_libraries(${_target} PRIVATE ${PROFILER_HUB_EXAMPLE_LIBRARY})
    target_compile_features(${_target} PRIVATE cxx_std_20)

    # Keep the executables under the profiler-hub build tree at
    # <build>/bin/examples.
    set_target_properties(
        ${_target}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROFILER_HUB_EXAMPLE_OUTPUT_DIR}"
    )

    # Locate the shared library relative to the binary
    set_target_properties(${_target} PROPERTIES BUILD_RPATH "$ORIGIN/../../lib")

    if(PROFILER_HUB_INSTALL_EXAMPLES)
        install(
            TARGETS ${_target}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/profiler-hub/examples
            COMPONENT profiler-hub-examples
        )
    endif()
endforeach()

# schema_v3/ -> examples/ -> profiler-hub/, whether this is the top-level
# project or a subdirectory of a larger build.
get_filename_component(_PHUB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(_PHUB_INTEGRATION_DIR "${_PHUB_ROOT}/tests/integration")

if(PROFILER_HUB_BUILD_TESTS AND EXISTS "${_PHUB_INTEGRATION_DIR}")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE _phub_has_pytest
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _phub_has_pytest EQUAL 0)
        message(
            STATUS
            "[profiler-hub] pytest not found; integration CTests disabled"
        )
    else()
        file(
            GLOB _PHUB_INTEGRATION_PYTEST_FILES
            CONFIGURE_DEPENDS
            "${_PHUB_INTEGRATION_DIR}/*.py"
        )
        # Same layout as rocprofiler-systems: pytest writes CTestTestfile.cmake,
        # but the custom-command OUTPUT is a stamp. That keeps CMake from
        # treating the test file as a build output it owns. The stamp is
        # removed at configure so the next build regenerates the tests.
        # The file lives under tests/integration, not this directory, so the
        # schema_v3 CTestTestfile.cmake and tests/unit stay untouched.
        set(_PHUB_PYTEST_CTEST_DIR
            "${profiler-hub_BINARY_DIR}/tests/integration"
        )
        set(_PHUB_PYTEST_CTEST_FILE
            "${_PHUB_PYTEST_CTEST_DIR}/CTestTestfile.cmake"
        )
        set(_PHUB_PYTEST_CTEST_STAMP
            "${_PHUB_PYTEST_CTEST_DIR}/.pytest_ctest_stamp"
        )
        file(REMOVE "${_PHUB_PYTEST_CTEST_STAMP}")

        add_custom_command(
            OUTPUT "${_PHUB_PYTEST_CTEST_STAMP}"
            COMMAND
                ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
                "PHUB_CTEST_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
                "PHUB_CTEST_SOURCE_DIR=${_PHUB_ROOT}"
                "PHUB_CTEST_EXAMPLE_BIN_DIR=${PROFILER_HUB_EXAMPLE_OUTPUT_DIR}"
                ${Python3_EXECUTABLE} -m pytest ${_PHUB_INTEGRATION_DIR}
                --ctest-mode=generate --ctest-output-path
                "${_PHUB_PYTEST_CTEST_FILE}" -q -p no:cacheprovider
            COMMAND ${CMAKE_COMMAND} -E touch "${_PHUB_PYTEST_CTEST_STAMP}"
            DEPENDS ${_PHUB_INTEGRATION_PYTEST_FILES}
            WORKING_DIRECTORY ${_PHUB_ROOT}
            COMMENT "Generating CTest definitions from pytest suite"
            VERBATIM
        )

        add_custom_target(
            generate-profiler-hub-pytest-ctests
            ALL
            DEPENDS "${_PHUB_PYTEST_CTEST_STAMP}"
        )
        add_dependencies(
            generate-profiler-hub-pytest-ctests
            ${PROFILER_HUB_SCHEMA_V3_EXAMPLE_TARGETS}
        )
    endif()
endif()
