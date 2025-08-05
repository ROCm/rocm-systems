if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(STATUS "")
    message(STATUS "rocprofiler does not support in-source builds.")
    message(STATUS "Delete CMakeCache.txt and CMakeFiles in ${PROJECT_SOURCE_DIR}")
    message(STATUS "and run cmake with `-B <build-directory>`")
    message(STATUS "")
    message(FATAL_ERROR "In-source build detected.")
endif()

option(ROCPROFILER_BUILD_TESTS "Enable building the tests" OFF)
option(ROCPROFILER_BUILD_SAMPLES "Enable building the code samples" OFF)

# CLI and FILE plugins are always built
foreach(_PLUGIN "ATT" "CTF" "PERFETTO")
    option(ROCPROFILER_BUILD_PLUGIN_${_PLUGIN} "Enable building the ${_PLUGIN} plugin" ON)
endforeach()

option(ROCPROFILER_DEBUG_TRACE "Enable debug tracing" OFF)
mark_as_advanced(ROCPROFILER_DEBUG_TRACE)

option(ROCPROFILER_LD_AQLPROFILE "Enable direct loading of AQL-profile HSA extension" OFF)
mark_as_advanced(ROCPROFILER_LD_AQLPROFILE)

option(ROCPROFILER_BUILD_CI "Enable continuous integration additions" OFF)
mark_as_advanced(ROCPROFILER_BUILD_CI)

option(ROCPROFILER_ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
mark_as_advanced(ROCPROFILER_ENABLE_CLANG_TIDY)

set(ROCPROFILER_BUILD_TYPES "Release" "RelWithDebInfo" "Debug" "MinSizeRel" "Coverage")

# export compile commands in the project
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(ROCPROFILER_MEMCHECK
    ""
    CACHE STRING "Memory checker type")
mark_as_advanced(ROCPROFILER_MEMCHECK)

# ASAN is defined by testing team on Jenkins
if(ASAN)
    set(ROCPROFILER_MEMCHECK
        "AddressSanitizer"
        CACHE STRING "Memory checker type (forced by ASAN defined)" FORCE)
endif()

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer" "LeakSanitizer"
                               "MemorySanitizer" "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND NOT ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    message(
        FATAL_ERROR
            "Unsupported memcheck type '${ROCPROFILER_MEMCHECK}'. Options: ${ROCPROFILER_MEMCHECK_TYPES}"
        )
endif()

set_property(CACHE ROCPROFILER_MEMCHECK PROPERTY STRINGS "${ROCPROFILER_MEMCHECK_TYPES}")

add_library(rocprofiler-memcheck INTERFACE)
add_library(rocprofiler::memcheck ALIAS rocprofiler-memcheck)

function(rocprofiler_add_memcheck_flags _TYPE)
    target_compile_options(
        rocprofiler-memcheck INTERFACE $<BUILD_INTERFACE:-g3 -Og -fno-omit-frame-pointer
                                       -fsanitize=${_TYPE}>)
    target_link_options(rocprofiler-memcheck INTERFACE
                        $<BUILD_INTERFACE:-fsanitize=${_TYPE} -Wl,--no-undefined>)
endfunction()

function(rocprofiler_set_memcheck_env _TYPE _LIB_BASE)
    set(_LIBS ${_LIB_BASE})
    foreach(_N 6 5 4 3 2 1 0)
        list(
            APPEND _LIBS
            ${CMAKE_SHARED_LIBRARY_PREFIX}${_LIB_BASE}${CMAKE_SHARED_LIBRARY_SUFFIX}.${_N}
            )
    endforeach()
    foreach(_LIB ${_LIBS})
        if(NOT ${_TYPE}_LIBRARY)
            find_library(${_TYPE}_LIBRARY NAMES ${_LIB} ${ARGN})
        endif()
    endforeach()

    target_link_libraries(rocprofiler-memcheck INTERFACE ${_LIB_BASE})

    if(${_TYPE}_LIBRARY)
        set(ROCPROFILER_MEMCHECK_PRELOAD_ENV
            "LD_PRELOAD=${${_TYPE}_LIBRARY};LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
            CACHE INTERNAL "LD_PRELOAD env variable for tests" FORCE)
    endif()
endfunction()

# always unset so that it doesn't preload if memcheck disabled
unset(ROCPROFILER_MEMCHECK_PRELOAD_ENV CACHE)

if(ROCPROFILER_MEMCHECK STREQUAL "AddressSanitizer")
    rocprofiler_add_memcheck_flags("address")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "asan")
elseif(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    rocprofiler_add_memcheck_flags("leak")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "lsan")
elseif(ROCPROFILER_MEMCHECK STREQUAL "MemorySanitizer")
    rocprofiler_add_memcheck_flags("memory")
elseif(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    rocprofiler_add_memcheck_flags("thread")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "tsan")
elseif(ROCPROFILER_MEMCHECK STREQUAL "UndefinedBehaviorSanitizer")
    rocprofiler_add_memcheck_flags("undefined")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "ubsan")
endif()
