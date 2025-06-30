#
#   rocprofiler_options.cmake
#
#   Configure miscellaneous settings
#
include_guard(GLOBAL)

# export compile commands of the project. Many IDEs want the compile_commands.json in root
# directory so run ln -s <build>/compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# C settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_VISIBILITY_PRESET "hidden")
# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
# general settings affecting build
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_UNITY_BUILD OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

rocprofiler_add_feature(CMAKE_BUILD_TYPE "Build type")
rocprofiler_add_feature(CMAKE_INSTALL_PREFIX "Install prefix")

# standard cmake options
rocprofiler_add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
# rocprofiler_add_option(BUILD_STATIC_LIBS "Build static libraries" OFF)

rocprofiler_add_option(
    ROCPROFILER_BUILD_CI "Enable continuous integration default values for options" OFF
    ADVANCED)

rocprofiler_add_option(ROCPROFILER_BUILD_TESTS "Enable building the tests"
                       ${ROCPROFILER_BUILD_CI})
rocprofiler_add_option(ROCPROFILER_BUILD_SAMPLES "Enable building the code samples"
                       ${ROCPROFILER_BUILD_CI})
rocprofiler_add_option(ROCPROFILER_BUILD_BENCHMARK "Enable building the benchmarks" OFF)
rocprofiler_add_option(
    ROCPROFILER_BUILD_CI_STRICT_TIMESTAMPS
    "Disable adjusting for clock skew b/t CPU and GPU timestamps" OFF ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_CODECOV
                       "Enable building for code coverage analysis" OFF)
rocprofiler_add_option(ROCPROFILER_BUILD_DOCS
                       "Enable build + install + packaging documentation" OFF)
rocprofiler_add_option(
    ROCPROFILER_INTERNAL_RCCL_API_TRACE
    "Use (internal) <rocprofiler-sdk/rccl/details/api_trace.h> instead of RCCL-provided <rccl/amd_detail/api_trace.h>. Note: this should never be used in production"
    OFF
    ADVANCED)

rocprofiler_add_option(
    ROCPROFILER_BUILD_GHC_FS
    "Enable building with ghc::filesystem library (via submodule) instead of the C++ filesystem library"
    ON)
rocprofiler_add_option(ROCPROFILER_BUILD_FMT "Enable building fmt library internally" ON)
rocprofiler_add_option(ROCPROFILER_BUILD_GLOG
                       "Enable building glog (Google logging) library internally" ON)
rocprofiler_add_option(ROCPROFILER_BUILD_SQLITE3
                       "Enable building sqlite3 library internally" OFF)
rocprofiler_add_option(ROCPROFILER_BUILD_PYBIND11
                       "Enable building pybind11 library internally" ON)
rocprofiler_add_option(ROCPROFILER_BUILD_GOTCHA
                       "Enable building gotcha library internally" ON)
if(ROCPROFILER_BUILD_TESTS)
    rocprofiler_add_option(
        ROCPROFILER_BUILD_GTEST
        "Enable building gtest (Google testing) library internally" ON ADVANCED)
endif()

rocprofiler_add_option(ROCPROFILER_ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF
                       ADVANCED)

rocprofiler_add_option(
    ROCPROFILER_BUILD_DEVELOPER "Extra build flags for development like -Werror"
    ${ROCPROFILER_BUILD_CI} ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_WERROR "Any compiler warnings are errors"
                       ${ROCPROFILER_BUILD_CI} ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_RELEASE "Build with minimal debug info" OFF
                       ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_DEBUG "Build with extra debug info" OFF ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_STATIC_LIBGCC
                       "Build with -static-libgcc if possible" OFF ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_STATIC_LIBSTDCXX
                       "Build with -static-libstdc++ if possible" OFF ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_STACK_PROTECTOR "Build with -fstack-protector"
                       ON ADVANCED)
rocprofiler_add_option(ROCPROFILER_UNSAFE_NO_VERSION_CHECK
                       "Disable HSA version checking (for development only)" OFF ADVANCED)
rocprofiler_add_option(
    ROCPROFILER_REGENERATE_COUNTERS_PARSER
    "Regenerate the counter parser (requires bison and flex)" OFF ADVANCED)
rocprofiler_add_option(
    ROCPROFILER_BUILD_EXPERIMENTAL_WARNINGS
    "Enable warnings for experimental features but hide with -Wno-deprecated-declarations (this ensures that experimental warning message does not break macros)"
    OFF
    ADVANCED)
rocprofiler_add_option(ROCPROFILER_BUILD_DEPRECATED_WARNINGS
                       "Enable warnings for use of deprecated features" OFF ADVANCED)

# In the future, we will do this even with clang-tidy enabled
foreach(_OPT ROCPROFILER_BUILD_DEVELOPER ROCPROFILER_BUILD_WERROR)
    if(ROCPROFILER_BUILD_CI AND NOT ${_OPT})
        message(AUTHOR_WARNING "Forcing ${_OPT}=ON because ROCPROFILER_BUILD_CI=ON")
        set(${_OPT}
            ON
            CACHE BOOL "forced due ROCPROFILER_BUILD_CI=ON" FORCE)
    endif()
endforeach()

set(ROCPROFILER_BUILD_TYPES "Release" "RelWithDebInfo" "Debug" "MinSizeRel" "Coverage")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE IN_LIST ROCPROFILER_BUILD_TYPES)
    message(
        FATAL_ERROR
            "Unsupported build type '${CMAKE_BUILD_TYPE}'. Options: ${ROCPROFILER_BUILD_TYPES}"
        )
endif()

if(ROCPROFILER_BUILD_CI)
    foreach(_BUILD_TYPE ${ROCPROFILER_BUILD_TYPES})
        string(TOUPPER "${_BUILD_TYPE}" _BUILD_TYPE)

        # remove NDEBUG preprocessor def so that asserts are triggered
        string(REGEX REPLACE ".DNDEBUG" "" CMAKE_C_FLAGS_${_BUILD_TYPE}
                             "${CMAKE_C_FLAGS_${_BUILD_TYPE}}")
        string(REGEX REPLACE ".DNDEBUG" "" CMAKE_CXX_FLAGS_${_BUILD_TYPE}
                             "${CMAKE_CXX_FLAGS_${_BUILD_TYPE}}")
    endforeach()
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ROCPROFILER_BUILD_TYPES}")
endif()

rocprofiler_add_cache_option(ROCPROFILER_MEMCHECK "" STRING "Memory checker type"
                             ADVANCED)

# ASAN is defined by testing team on Jenkins
if(ASAN)
    set(ROCPROFILER_MEMCHECK
        "AddressSanitizer"
        CACHE STRING "Memory checker type (forced by ASAN defined)" FORCE)
endif()

include(rocprofiler_memcheck)

# default FAIL_REGULAR_EXPRESSION for tests
set(ROCPROFILER_DEFAULT_FAIL_REGEX
    "threw an exception|Permission denied|Could not create logging file|failed with error code|Subprocess aborted"
    CACHE INTERNAL "Default FAIL_REGULAR_EXPRESSION for tests" FORCE)

# this should be defaulted to OFF by ROCm 7.0.1 or 7.1 this should only used to disable
# sample tests in extreme circumstances
option(ROCPROFILER_DISABLE_UNSTABLE_CTESTS "Disable unstable tests" ON)
