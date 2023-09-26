#
#   rocprofiler_register_options.cmake
#
#   Configure miscellaneous settings
#
# standard cmake options
rocprofiler_register_add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
rocprofiler_register_add_option(BUILD_STATIC_LIBS "Build static libraries" OFF)
rocprofiler_register_add_option(CMAKE_POSITION_INDEPENDENT_CODE
                                "Build position independent code" ON)

# export compile commands in the project
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

rocprofiler_register_add_option(ROCPROFILER_REGISTER_BUILD_TESTS
                                "Enable building the tests" OFF)
rocprofiler_register_add_option(ROCPROFILER_REGISTER_BUILD_SAMPLES
                                "Enable building the code samples" OFF)
rocprofiler_register_add_option(ROCPROFILER_REGISTER_BUILD_CI
                                "Enable continuous integration additions" OFF ADVANCED)
rocprofiler_register_add_option(ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY
                                "Enable clang-tidy checks" OFF ADVANCED)
rocprofiler_register_add_option(
    ROCPROFILER_REGISTER_BUILD_DEVELOPER "Extra build flags for development like -Werror"
    ${ROCPROFILER_REGISTER_BUILD_CI} ADVANCED)
rocprofiler_register_add_option(ROCPROFILER_REGISTER_BUILD_GLOG "Build GLOG" ON)
rocprofiler_register_add_option(ROCPROFILER_REGISTER_BUILD_FMT "Build FMT" ON)

# In the future, we will do this even with clang-tidy enabled
if(ROCPROFILER_REGISTER_BUILD_CI AND NOT ROCPROFILER_REGISTER_ENABLE_CLANG_TIDY)
    message(
        STATUS
            "Forcing ROCPROFILER_REGISTER_BUILD_DEVELOPER=ON because ROCPROFILER_REGISTER_BUILD_CI=ON"
        )
    set(ROCPROFILER_REGISTER_BUILD_DEVELOPER
        ON
        CACHE
            BOOL
            "Any compiler warnings are errors (forced due ROCPROFILER_REGISTER_BUILD_CI=ON)"
            FORCE)
endif()

set(ROCPROFILER_REGISTER_BUILD_TYPES "Release" "RelWithDebInfo" "Debug" "MinSizeRel"
                                     "Coverage")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE IN_LIST ROCPROFILER_REGISTER_BUILD_TYPES)
    message(
        FATAL_ERROR
            "Unsupported build type '${CMAKE_BUILD_TYPE}'. Options: ${ROCPROFILER_REGISTER_BUILD_TYPES}"
        )
endif()

if(ROCPROFILER_REGISTER_BUILD_CI)
    foreach(_BUILD_TYPE ${ROCPROFILER_REGISTER_BUILD_TYPES})
        string(TOUPPER "${_BUILD_TYPE}" _BUILD_TYPE)

        # remove NDEBUG preprocessor def so that asserts are triggered
        string(REGEX REPLACE ".DNDEBUG" "" CMAKE_C_FLAGS_${_BUILD_TYPE}
                             "${CMAKE_C_FLAGS_${_BUILD_TYPE}}")
        string(REGEX REPLACE ".DNDEBUG" "" CMAKE_CXX_FLAGS_${_BUILD_TYPE}
                             "${CMAKE_CXX_FLAGS_${_BUILD_TYPE}}")
    endforeach()
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                                                 "${ROCPROFILER_REGISTER_BUILD_TYPES}")
endif()

rocprofiler_register_add_cache_option(ROCPROFILER_REGISTER_MEMCHECK "" STRING
                                      "Memory checker type" ADVANCED)

# ASAN is defined by testing team on Jenkins
if(ASAN)
    set(ROCPROFILER_REGISTER_MEMCHECK
        "AddressSanitizer"
        CACHE STRING "Memory checker type (forced by ASAN defined)" FORCE)
endif()

include(rocprofiler_register_memcheck)
