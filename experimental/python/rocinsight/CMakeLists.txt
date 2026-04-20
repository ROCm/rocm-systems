###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(
    rocinsight
    VERSION 0.1.0
    DESCRIPTION "AI-powered GPU trace analysis for rocprofiler-sdk databases"
    LANGUAGES NONE
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ROCINSIGHT_INSTALL_PYTHON "Install rocinsight Python package" ON)
option(ROCINSIGHT_BUILD_TESTS    "Build/configure rocinsight tests"  ON)

# ---------------------------------------------------------------------------
# Python package installation
# ---------------------------------------------------------------------------
if(ROCINSIGHT_INSTALL_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Determine site-packages install prefix
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install prefix" FORCE)
    endif()

    # Install the rocinsight Python package into the standard site-packages
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
        PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/rocinsight"
        DESTINATION bin
        OPTIONAL
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ROCINSIGHT_BUILD_TESTS AND BUILD_TESTING)
    enable_testing()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(
        NAME rocinsight_unit_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest
            "${CMAKE_CURRENT_SOURCE_DIR}/rocinsight/ai_analysis/tests"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            --noconftest
            -v
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    set_tests_properties(rocinsight_unit_tests PROPERTIES
        LABELS "rocinsight;python;unit"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()
