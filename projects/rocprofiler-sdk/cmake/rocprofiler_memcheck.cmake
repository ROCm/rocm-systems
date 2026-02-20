#
#
#
set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer" "LeakSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND NOT ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    message(
        FATAL_ERROR
            "Unsupported memcheck type '${ROCPROFILER_MEMCHECK}'. Options: ${ROCPROFILER_MEMCHECK_TYPES}"
        )
endif()

set_property(CACHE ROCPROFILER_MEMCHECK PROPERTY STRINGS "${ROCPROFILER_MEMCHECK_TYPES}")

function(rocprofiler_add_memcheck_flags _TYPE _LIB_BASE _FLAG)
    target_compile_options(
        rocprofiler-sdk-memcheck
        INTERFACE $<BUILD_INTERFACE:-g3 -Og -fno-omit-frame-pointer
                  -fno-optimize-sibling-calls -fno-inline-functions -fsanitize=${_FLAG}
                  ${ARGN}>)
    target_link_options(rocprofiler-sdk-memcheck INTERFACE
                        $<BUILD_INTERFACE:-fsanitize=${_FLAG} -Wl,--no-undefined>)

    if(NOT EXISTS ${PROJECT_BINARY_DIR}/CMakeFiles/CMakeTmp)
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/CMakeTmp")
    endif()

    execute_process(
        COMMAND ${PROJECT_SOURCE_DIR}/source/scripts/deduce-sanitizer-lib.sh
                lib${_LIB_BASE} ${CMAKE_CXX_COMPILER} -fsanitize=${_FLAG}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/CMakeFiles/CMakeTmp
        RESULT_VARIABLE _DEDUCE_RET
        ERROR_VARIABLE _DEDUCE_ERR
        OUTPUT_VARIABLE _DEDUCE_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_DEDUCE_RET EQUAL 0 AND EXISTS "${_DEDUCE_OUT}")
        set(${_TYPE}_LIBRARY
            "${_DEDUCE_OUT}"
            CACHE FILEPATH "Linked library when compiled with -fsanitize=${_FLAG}")
    endif()
endfunction()

function(rocprofiler_set_memcheck_env _TYPE _LIB_BASE)
    if(NOT ${_TYPE}_LIBRARY)
        set(_LIBS ${_LIB_BASE})

        foreach(_N ${ARGN} 6 5 4 3 2 1 0)
            list(
                APPEND
                _LIBS
                ${CMAKE_SHARED_LIBRARY_PREFIX}${_LIB_BASE}${CMAKE_SHARED_LIBRARY_SUFFIX}.${_N}
                )
        endforeach()

        foreach(_LIB ${_LIBS})
            if(NOT ${_TYPE}_LIBRARY)
                find_library(${_TYPE}_LIBRARY NAMES ${_LIB})
            endif()
        endforeach()
    endif()

    target_link_libraries(rocprofiler-sdk-memcheck INTERFACE ${_LIB_BASE})

    if(${_TYPE}_LIBRARY)
        set(ROCPROFILER_MEMCHECK_PRELOAD_ENV
            "LD_PRELOAD=${${_TYPE}_LIBRARY}"
            CACHE INTERNAL "LD_PRELOAD env variable for tests " FORCE)
        set(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE
            "${${_TYPE}_LIBRARY}"
            CACHE INTERNAL "Library to LD_PRELOAD for tests " FORCE)
    endif()
endfunction()

# always unset so that it doesn't preload if memcheck disabled
unset(ROCPROFILER_MEMCHECK_PRELOAD_ENV CACHE)
unset(ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE CACHE)

# the soversions below are fallbacks in case deduce-sanitizer-lib.sh fails
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                           VERSION_GREATER_EQUAL "13.0.0")
    set(ThreadSanitizer_SOVERSION 2)
    set(AddressSanitizer_SOVERSION 8)
else()
    set(ThreadSanitizer_SOVERSION 0)
    set(AddressSanitizer_SOVERSION 6)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "AddressSanitizer")
    rocprofiler_add_memcheck_flags("${ROCPROFILER_MEMCHECK}" "asan" "address")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "asan"
                                 ${AddressSanitizer_SOVERSION})
elseif(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    rocprofiler_add_memcheck_flags("${ROCPROFILER_MEMCHECK}" "lsan" "leak")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "lsan")
elseif(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    rocprofiler_add_memcheck_flags("${ROCPROFILER_MEMCHECK}" "tsan" "thread")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "tsan"
                                 ${ThreadSanitizer_SOVERSION})
elseif(ROCPROFILER_MEMCHECK STREQUAL "UndefinedBehaviorSanitizer")
    rocprofiler_add_memcheck_flags("${ROCPROFILER_MEMCHECK}" "ubsan" "undefined"
                                   "-fno-sanitize-recover=undefined")
    rocprofiler_set_memcheck_env("${ROCPROFILER_MEMCHECK}" "ubsan")
elseif(NOT ROCPROFILER_MEMCHECK STREQUAL "")
    message(FATAL_ERROR "Unsupported ROCPROFILER_MEMCHECK type: ${ROCPROFILER_MEMCHECK}")
endif()
