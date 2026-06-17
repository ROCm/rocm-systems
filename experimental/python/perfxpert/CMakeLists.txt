###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _PERFXPERT_PYPROJECT)
string(
    REGEX MATCH "(^|\n)[ \t]*version[ \t]*=[ \t]*\"([^\"]+)\""
    _PERFXPERT_VERSION_MATCH
    "${_PERFXPERT_PYPROJECT}"
)
if(NOT CMAKE_MATCH_2)
    message(FATAL_ERROR "Unable to read perfxpert version from pyproject.toml")
endif()
set(PERFXPERT_VERSION "${CMAKE_MATCH_2}")

project(
    perfxpert
    VERSION ${PERFXPERT_VERSION}
    DESCRIPTION "AI-powered GPU trace analysis for rocprofiler-sdk databases"
    LANGUAGES NONE
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PERFXPERT_INSTALL_PYTHON "Install perfxpert Python package" ON)
option(PERFXPERT_BUILD_TESTS    "Build/configure perfxpert tests"  ON)

# ---------------------------------------------------------------------------
# Python package installation
# ---------------------------------------------------------------------------
if(PERFXPERT_INSTALL_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Determine site-packages install prefix
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install prefix" FORCE)
    endif()

    # Install the perfxpert Python package into the standard site-packages
    install(
        CODE "execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip install
                --prefix \"\${CMAKE_INSTALL_PREFIX}\"
                --no-deps
                --no-build-isolation
                \"${CMAKE_CURRENT_SOURCE_DIR}\"
            RESULT_VARIABLE _pip_result
        )
        if(NOT _pip_result EQUAL 0)
            message(FATAL_ERROR \"pip install failed: \${_pip_result}\")
        endif()"
    )

    # Also expose the entry-point script
    install(
        PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/perfxpert"
        DESTINATION bin
        OPTIONAL
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PERFXPERT_BUILD_TESTS AND BUILD_TESTING)
    enable_testing()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(
        NAME perfxpert_unit_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            --noconftest
            -v
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set_tests_properties(perfxpert_unit_tests PROPERTIES
        LABELS "perfxpert;python;unit"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()
