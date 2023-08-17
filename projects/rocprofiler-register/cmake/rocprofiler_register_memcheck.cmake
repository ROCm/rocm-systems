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
                  -fno-optimize-sibling-calls -fno-inline-functions -fsanitize=${_TYPE}>
                  -static-lib${_LIB})
    target_link_options(
        rocprofiler-register-memcheck INTERFACE $<BUILD_INTERFACE:-fsanitize=${_TYPE}
        -static-lib${_LIB} -Wl,--no-undefined>)
endfunction()

# always unset so that it doesn't preload if memcheck disabled
unset(ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_ENV CACHE)

if(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "AddressSanitizer")
    rocprofiler_register_add_memcheck_flags("address" "asan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "LeakSanitizer")
    rocprofiler_register_add_memcheck_flags("leak" "lsan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "ThreadSanitizer")
    rocprofiler_register_add_memcheck_flags("thread" "tsan")
elseif(ROCPROFILER_REGISTER_MEMCHECK STREQUAL "UndefinedBehaviorSanitizer")
    rocprofiler_register_add_memcheck_flags("undefined" "ubsan")
endif()
