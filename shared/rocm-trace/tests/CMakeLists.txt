# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

enable_language(C)
enable_language(CXX)

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
endif()
# FetchContent and CONFIG mode define GTest::gtest_main; MODULE mode defines GTest::Main.
if(NOT GTest_FOUND)
    message(STATUS "rocm-trace: GTest not found, fetching it")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.16.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
elseif(TARGET GTest::Main AND NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS GTest::Main)
endif()
include(GoogleTest)

if(MSVC)
    set(ROCM_TRACE_TEST_WARNINGS /W4 /WX)
else()
    set(ROCM_TRACE_TEST_WARNINGS -Wall -Wextra -Wpedantic -Werror)
endif()

# values.h: the C++ checks plus a strict C99 translation unit.
add_executable(rocm-trace-tests values_test.cpp values_c.c)
target_link_libraries(
    rocm-trace-tests
    PRIVATE rocm::trace_values GTest::gtest_main
)
target_compile_features(rocm-trace-tests PRIVATE cxx_std_17)
set_target_properties(
    rocm-trace-tests
    PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
)
target_compile_options(rocm-trace-tests PRIVATE ${ROCM_TRACE_TEST_WARNINGS})
gtest_add_tests(TARGET rocm-trace-tests)

# trace.h needs a TraceLogging implementation: the Windows SDK, or on Linux the LTTng
# flavour of microsoft/tracelogging and the liblttng-ust headers. Compile-only check, in
# C and C++. TraceLogging's LTTng macros use GNU statement expressions, so no -Wpedantic.
if(NOT WIN32)
    find_path(
        ROCM_TRACE_TRACELOGGING_INCLUDE_DIR
        tracelogging/TraceLoggingProvider.h
    )
endif()
if(WIN32 OR ROCM_TRACE_TRACELOGGING_INCLUDE_DIR)
    add_library(rocm-trace-header-check OBJECT trace_check.c trace_check.cpp)
    target_link_libraries(rocm-trace-header-check PRIVATE rocm::trace)
    if(ROCM_TRACE_TRACELOGGING_INCLUDE_DIR)
        target_include_directories(
            rocm-trace-header-check
            PRIVATE ${ROCM_TRACE_TRACELOGGING_INCLUDE_DIR}
        )
    endif()
    target_compile_features(rocm-trace-header-check PRIVATE cxx_std_17)
    # trace.h needs C11 (static_assert); values.h is checked as C99 above.
    set_target_properties(
        rocm-trace-header-check
        PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON
    )
    if(NOT MSVC)
        target_compile_options(rocm-trace-header-check PRIVATE -Wall -Werror)
    endif()
else()
    message(
        STATUS
        "rocm-trace: no TraceLogging header (ROCM_TRACE_TRACELOGGING_INCLUDE_DIR), skipping the trace.h check"
    )
endif()

# The probe generator in scripts/: its pytest suite, when Python has the generator's
# requirements (scripts/requirements.txt).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import clang.cindex, yaml, pytest"
        RESULT_VARIABLE ROCM_TRACE_GEN_DEPS
        OUTPUT_QUIET
        ERROR_QUIET
    )
endif()
if(Python3_FOUND AND ROCM_TRACE_GEN_DEPS EQUAL 0)
    add_test(
        NAME rocm-trace-gen
        COMMAND
            ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
            ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/tests
    )
    set(ROCM_TRACE_GEN_TEST_ENV PYTHONDONTWRITEBYTECODE=1)
    if(ROCM_TRACE_TRACELOGGING_INCLUDE_DIR)
        list(
            APPEND ROCM_TRACE_GEN_TEST_ENV
            ROCM_TRACE_GEN_TRACELOGGING_INCLUDE=${ROCM_TRACE_TRACELOGGING_INCLUDE_DIR}
        )
    endif()
    set_tests_properties(
        rocm-trace-gen
        PROPERTIES ENVIRONMENT "${ROCM_TRACE_GEN_TEST_ENV}"
    )
else()
    message(
        STATUS
        "rocm-trace: Python with libclang, PyYAML and pytest not found, skipping the generator tests"
    )
endif()
