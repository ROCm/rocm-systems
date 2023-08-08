#
#
#
set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer" "LeakSanitizer"
                               "MemorySanitizer" "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND NOT ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    message(
        FATAL_ERROR
            "Unsupported memcheck type '${ROCPROFILER_MEMCHECK}'. Options: ${ROCPROFILER_MEMCHECK_TYPES}"
        )
endif()

set_property(CACHE ROCPROFILER_MEMCHECK PROPERTY STRINGS "${ROCPROFILER_MEMCHECK_TYPES}")

function(rocprofiler_add_memcheck_flags _TYPE)
    target_compile_options(
        rocprofiler-memcheck
        INTERFACE $<BUILD_INTERFACE:-g3 -Og -fno-omit-frame-pointer
                  -fno-optimize-sibling-calls -fno-inline-functions -fsanitize=${_TYPE}>)
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
            " LD_PRELOAD=${${_TYPE}_LIBRARY} "
            CACHE INTERNAL " LD_PRELOAD env variable for tests " FORCE)
    endif()
endfunction()

# always unset so that it doesn't preload if memcheck disabled
unset(ROCPROFILER_MEMCHECK_PRELOAD_ENV CACHE)

if(ROCPROFILER_MEMCHECK STREQUAL " AddressSanitizer ")
    rocprofiler_add_memcheck_flags(" address ")
    rocprofiler_set_memcheck_env(" ${ROCPROFILER_MEMCHECK}" "asan ")
elseif(ROCPROFILER_MEMCHECK STREQUAL " LeakSanitizer ")
    rocprofiler_add_memcheck_flags(" leak ")
    rocprofiler_set_memcheck_env(" ${ROCPROFILER_MEMCHECK}" "lsan ")
elseif(ROCPROFILER_MEMCHECK STREQUAL " MemorySanitizer ")
    rocprofiler_add_memcheck_flags(" memory ")
elseif(ROCPROFILER_MEMCHECK STREQUAL " ThreadSanitizer ")
    rocprofiler_add_memcheck_flags(" thread ")
    rocprofiler_set_memcheck_env(" ${ROCPROFILER_MEMCHECK}" "tsan ")
elseif(ROCPROFILER_MEMCHECK STREQUAL " UndefinedBehaviorSanitizer ")
    rocprofiler_add_memcheck_flags(" undefined ")
    rocprofiler_set_memcheck_env(" ${ROCPROFILER_MEMCHECK}" "ubsan ")
endif()
