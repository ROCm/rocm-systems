#
#
#
set(ROCPROFILER_REGISTER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                                        "LeakSanitizer" "UndefinedBehaviorSanitizer")

if(ROCPROFILER_REGISTER_MEMCHECK AND NOT ROCPROFILER_REGISTER_MEMCHECK IN_LIST
                                     ROCPROFILER_REGISTER_MEMCHECK_TYPES)
    message(
        FATAL_ERROR
            "Unsupported memcheck type '${ROCPROFILER_REGISTER_MEMCHECK}'. Options: ${ROCPROFILER_REGISTER_MEMCHECK_TYPES}"
        )
endif()

set_property(CACHE ROCPROFILER_REGISTER_MEMCHECK
             PROPERTY STRINGS "${ROCPROFILER_REGISTER_MEMCHECK_TYPES}")

function(rocprofiler_register_add_memcheck_flags _TYPE _LIB)
    target_compile_options(
        rocprofiler-register-memcheck
        INTERFACE $<BUILD_INTERFACE:-g3 -Og -fno-omit-frame-pointer
                  -fno-optimize-sibling-calls -fno-inline-functions -fsanitize=${_TYPE}>)
    target_link_options(rocprofiler-register-memcheck INTERFACE
                        $<BUILD_INTERFACE:-fsanitize=${_TYPE}>)
endfunction()

function(rocprofiler_register_set_memcheck_env _TYPE _LIB_BASE)
    set(_LIBS ${_LIB_BASE})

    foreach(
        _N
        8
        7
        6
        5
        4
        3
        2
        1
        0)
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

    target_link_libraries(rocprofiler-register-memcheck INTERFACE ${_LIB_BASE})

    if(${_TYPE}_LIBRARY)
        set(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY
            "${${_TYPE}_LIBRARY}"
            CACHE INTERNAL "LD_PRELOAD library for tests " FORCE)
        set(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV
            "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}"
            CACHE INTERNAL "LD_PRELOAD env variable for tests " FORCE)
    endif()
endfunction()

# always unset so that it doesn't preload if memcheck disabled
unset(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV CACHE)

if(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "AddressSanitizer")
    rocprofiler_register_add_memcheck_flags("address" "asan")
    rocprofiler_register_set_memcheck_env("${ROCPROFILER_REGISTER_MEMCHECK}" "asan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "LeakSanitizer")
    rocprofiler_register_add_memcheck_flags("leak" "lsan")
    rocprofiler_register_set_memcheck_env("${ROCPROFILER_REGISTER_MEMCHECK}" "lsan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "ThreadSanitizer")
    rocprofiler_register_add_memcheck_flags("thread" "tsan")
    rocprofiler_register_set_memcheck_env("${ROCPROFILER_REGISTER_MEMCHECK}" "tsan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "UndefinedBehaviorSanitizer")
    rocprofiler_register_add_memcheck_flags("undefined" "ubsan")
    rocprofiler_register_set_memcheck_env("${ROCPROFILER_REGISTER_MEMCHECK}" "ubsan")
endif()
