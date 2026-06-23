# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(PROFILER_HUB_SCHEMA_V3_EXAMPLES
    memory_alloc_writer_reader
    kernel_dispatch_writer_reader
    memory_copy_writer_reader
    region_writer_reader
)
set(PROFILER_HUB_SCHEMA_V3_EXAMPLE_TARGETS)

foreach(_example IN LISTS PROFILER_HUB_SCHEMA_V3_EXAMPLES)
    set(_target profiler-hub_${_example})
    list(APPEND PROFILER_HUB_SCHEMA_V3_EXAMPLE_TARGETS ${_target})

    add_executable(${_target} ${_example}.cpp)

    target_link_libraries(${_target} PRIVATE ${PROFILER_HUB_EXAMPLE_LIBRARY})
    target_compile_features(${_target} PRIVATE cxx_std_17)

    # Scratch directory for the rocpd database created during the round-trip.
    target_compile_definitions(
        ${_target}
        PRIVATE PHUB_INTEGRATION_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )

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

if(PROFILER_HUB_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/integration")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    file(
        GLOB _PHUB_INTEGRATION_PYTEST_FILES
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tests/integration/*.py"
    )
    set(_PHUB_PYTEST_CTEST_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake"
    )
    set(_PHUB_PYTEST_CTEST_STAMP
        "${CMAKE_CURRENT_BINARY_DIR}/.pytest_ctest_stamp"
    )
    file(REMOVE "${_PHUB_PYTEST_CTEST_STAMP}")

    add_custom_command(
        OUTPUT "${_PHUB_PYTEST_CTEST_STAMP}"
        COMMAND
            ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
            "PHUB_CTEST_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
            "PHUB_CTEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
            "PHUB_CTEST_EXAMPLE_BIN_DIR=${PROFILER_HUB_EXAMPLE_OUTPUT_DIR}"
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/integration/generate_ctest.py
            ${CMAKE_SOURCE_DIR}/tests/integration "${_PHUB_PYTEST_CTEST_FILE}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_PHUB_PYTEST_CTEST_STAMP}"
        DEPENDS ${_PHUB_INTEGRATION_PYTEST_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating CTest definitions from profiler-hub pytest suite"
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
