# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Integration test for simdojo find_package() support.
# Builds and installs simdojo as both STATIC and SHARED, then configures
# an out-of-tree consumer project against each installed package.
#
# Invoked via:  cmake -DSIMDOJO_DIR=... -DTEST_DIR=... -DWORK_DIR=... -P integration_test.cmake

cmake_minimum_required(VERSION 3.22)

foreach(_var SIMDOJO_DIR TEST_DIR WORK_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "${_var} must be set (-D${_var}=...)")
    endif()
endforeach()

set(pass 0)
set(fail 0)

macro(run_or_fail step)
    if(NOT rc EQUAL 0)
        message(STATUS "  FAIL (${variant} - ${step})")
        math(EXPR fail "${fail} + 1")
        set(_skip_variant TRUE)
    endif()
endmacro()

foreach(variant static shared)
    if(variant STREQUAL "static")
        set(shared_flag OFF)
    else()
        set(shared_flag ON)
    endif()

    set(install_dir "${WORK_DIR}/install-${variant}")
    set(build_lib   "${WORK_DIR}/build-simdojo-${variant}")
    set(build_test  "${WORK_DIR}/build-test-${variant}")
    set(_skip_variant FALSE)

    message(STATUS "==== ${variant} ====")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${SIMDOJO_DIR}" -B "${build_lib}"
            -G Ninja -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=${shared_flag}
            -DCMAKE_INSTALL_PREFIX=${install_dir}
        RESULT_VARIABLE rc
        OUTPUT_QUIET
    )
    run_or_fail("configure simdojo")
    if(_skip_variant)
        continue()
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${build_lib}" --target install
        RESULT_VARIABLE rc
        OUTPUT_QUIET
    )
    run_or_fail("install simdojo")
    if(_skip_variant)
        continue()
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -S "${TEST_DIR}" -B "${build_test}"
            -G Ninja -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_PREFIX_PATH=${install_dir}
        RESULT_VARIABLE rc
        OUTPUT_QUIET
    )
    run_or_fail("configure consumer")
    if(_skip_variant)
        continue()
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${build_test}"
        RESULT_VARIABLE rc
        OUTPUT_QUIET
    )
    run_or_fail("build consumer")
    if(_skip_variant)
        continue()
    endif()

    execute_process(
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${build_test}"
            --output-on-failure
        RESULT_VARIABLE rc
    )
    if(rc EQUAL 0)
        message(STATUS "  PASS (${variant})")
        math(EXPR pass "${pass} + 1")
    else()
        message(STATUS "  FAIL (${variant})")
        math(EXPR fail "${fail} + 1")
    endif()
endforeach()

message(STATUS "==== Results: ${pass} passed, ${fail} failed ====")
if(NOT fail EQUAL 0)
    message(FATAL_ERROR "${fail} integration test(s) failed")
endif()
