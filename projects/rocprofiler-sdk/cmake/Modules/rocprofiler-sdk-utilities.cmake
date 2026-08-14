#
# Miscellaneous cmake functions for rocprofiler-sdk
#

include_guard(GLOBAL)

# Diagnostic helper: returns a wall-clock timestamp in milliseconds. Falls back to
# second-resolution CMake timestamps if `date` is unavailable (e.g. non-Linux hosts).
function(rocprofiler_sdk_wallclock_ms _VAR)
    execute_process(
        COMMAND date +%s%3N
        RESULT_VARIABLE _date_ret
        OUTPUT_VARIABLE _date_out
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_date_ret EQUAL 0 AND _date_out MATCHES "^[0-9]+$")
        set(${_VAR}
            "${_date_out}"
            PARENT_SCOPE)
    else()
        string(TIMESTAMP _ts "%s")
        math(EXPR _ts_ms "${_ts} * 1000")
        set(${_VAR}
            "${_ts_ms}"
            PARENT_SCOPE)
    endif()
endfunction()

# Diagnostic helper: wraps add_subdirectory() and reports how long processing that
# subdirectory took during configure. Implemented as a macro so it shares the caller's
# directory scope (targets/tests are added exactly as with a bare add_subdirectory).
macro(rocprofiler_sdk_timed_add_subdirectory)
    rocprofiler_sdk_wallclock_ms(_rocprofiler_sdk_asd_t0)
    add_subdirectory(${ARGV})
    rocprofiler_sdk_wallclock_ms(_rocprofiler_sdk_asd_t1)
    math(EXPR _rocprofiler_sdk_asd_elapsed
         "${_rocprofiler_sdk_asd_t1} - ${_rocprofiler_sdk_asd_t0}")
    message(
        STATUS
            "[config-timing] add_subdirectory(${ARGV0}) took ${_rocprofiler_sdk_asd_elapsed} ms"
        )
endmacro()

function(rocprofiler_sdk_get_gfx_architectures _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX;DELIM" "" ${ARGN})

    if(NOT DEFINED ARG_DELIM)
        set(ARG_DELIM ", ")
    endif()

    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}]${ARG_PREFIX} ")

    find_program(
        rocminfo_EXECUTABLE
        NAMES rocminfo
        HINTS ${rocprofiler-sdk_ROOT_DIR} ${rocm_version_DIR} ${ROCM_PATH} /opt/rocm
        PATHS ${rocprofiler-sdk_ROOT_DIR} ${rocm_version_DIR} ${ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin)

    if(rocminfo_EXECUTABLE)
        rocprofiler_sdk_wallclock_ms(_rocminfo_t0)
        execute_process(
            COMMAND ${rocminfo_EXECUTABLE}
            RESULT_VARIABLE rocminfo_RET
            OUTPUT_VARIABLE rocminfo_OUT
            ERROR_VARIABLE rocminfo_ERR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
        rocprofiler_sdk_wallclock_ms(_rocminfo_t1)
        math(EXPR _rocminfo_elapsed "${_rocminfo_t1} - ${_rocminfo_t0}")

        # accumulate across every invocation during this configure so we can see the
        # total cost of repeatedly shelling out to rocminfo
        get_property(_rocminfo_total GLOBAL
                     PROPERTY ROCPROFILER_SDK_ROCMINFO_TOTAL_MS)
        get_property(_rocminfo_count GLOBAL
                     PROPERTY ROCPROFILER_SDK_ROCMINFO_CALL_COUNT)
        if(NOT _rocminfo_total)
            set(_rocminfo_total 0)
        endif()
        if(NOT _rocminfo_count)
            set(_rocminfo_count 0)
        endif()
        math(EXPR _rocminfo_total "${_rocminfo_total} + ${_rocminfo_elapsed}")
        math(EXPR _rocminfo_count "${_rocminfo_count} + 1")
        set_property(GLOBAL PROPERTY ROCPROFILER_SDK_ROCMINFO_TOTAL_MS
                                     "${_rocminfo_total}")
        set_property(GLOBAL PROPERTY ROCPROFILER_SDK_ROCMINFO_CALL_COUNT
                                     "${_rocminfo_count}")
        message(
            STATUS
                "[config-timing] rocminfo call #${_rocminfo_count} took ${_rocminfo_elapsed} ms (cumulative ${_rocminfo_total} ms)"
            )

        if(rocminfo_RET EQUAL 0)
            string(REGEX MATCHALL "gfx([0-9A-Fa-f]+)" rocminfo_GFXINFO "${rocminfo_OUT}")
            list(REMOVE_DUPLICATES rocminfo_GFXINFO)
            set(${_VAR}
                "${rocminfo_GFXINFO}"
                PARENT_SCOPE)

            if(ARG_ECHO)
                string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${rocminfo_GFXINFO}")
                message(STATUS "${ARG_PREFIX}System architectures: ${_GFXINFO_ECHO}")
            endif()
        else()
            message(
                AUTHOR_WARNING
                    "${rocminfo_EXECUTABLE} returned ${rocminfo_RET}\nstderr:\n${rocminfo_ERR}\nstdout:\n${rocminfo_OUT}"
                )
        endif()
    endif()
endfunction()

# In case the underlying architecture does not support PC sampling, this function will
# tell us whether the PC sampling is disabled
function(rocprofiler_sdk_pc_sampling_disabled _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX" "" ${ARGN})

    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}]${ARG_PREFIX} ")

    rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)
    list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)

    if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx90a$"
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx94[0-9]$"
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx95[0-9]$"
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx12[0-9][0-9]$")
        # PC sampling is enabled on this architecture.
        set(${_VAR}
            FALSE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "PC Sampling is enabled for ${pc-sampling-gpu-0-gfx-info}")
        endif()
    else()
        # PC sampling is disabled on this architecture.
        set(${_VAR}
            TRUE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "PC Sampling is disabled for ${pc-sampling-gpu-0-gfx-info}")
        endif()
    endif()
endfunction()

# In case the underlying architecture does not support stochastic PC sampling, this
# function will tell us whether the PC sampling is disabled
function(rocprofiler_sdk_pc_sampling_stochastic_disabled _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX" "" ${ARGN})

    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}]${ARG_PREFIX} ")

    rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)
    list(GET rocprofiler-sdk-tests-gfx-info 0 pc-sampling-gpu-0-gfx-info)

    if("${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx94[0-9]$"
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx95[0-9]$"
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx1250$")
        # PC sampling is enabled on this architecture.
        set(${_VAR}
            FALSE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "PC Sampling is enabled for ${pc-sampling-gpu-0-gfx-info}")
        endif()
    else()
        # PC sampling is disabled on this architecture.
        set(${_VAR}
            TRUE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "PC Sampling is disabled for ${pc-sampling-gpu-0-gfx-info}")
        endif()
    endif()
endfunction()

# Checks whether triple buffer is implemented for architecture: MI3xx and gfx12
function(rocprofiler_sdk_sqtt_triple_buffer_disabled _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX" "" ${ARGN})

    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}]${ARG_PREFIX} ")

    rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)
    list(GET rocprofiler-sdk-tests-gfx-info 0 gpu-0-gfx-info)

    if("${gpu-0-gfx-info}" MATCHES "^gfx(9[4-5][0-9]|12[0-9][0-9])$")
        set(${_VAR}
            FALSE
            PARENT_SCOPE)
    else()
        set(${_VAR}
            TRUE
            PARENT_SCOPE)
    endif()
endfunction()

function(rocprofiler_sdk_spm_disabled _VAR)
    cmake_parse_arguments(ARG "ECHO" "PREFIX" "" ${ARGN})

    set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}]${ARG_PREFIX} ")

    rocprofiler_sdk_get_gfx_architectures(rocprofiler-sdk-tests-gfx-info ECHO)
    list(GET rocprofiler-sdk-tests-gfx-info 0 spm-gpu-0-gfx-info)

    if("${spm-gpu-0-gfx-info}" MATCHES "^gfx94[0-9]$")
        # spm is enabled on this architecture.
        set(${_VAR}
            FALSE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "SPM is enabled for ${spm-gpu-0-gfx-info}")
        endif()
    else()
        # SPM is disabled on this architecture.
        set(${_VAR}
            TRUE
            PARENT_SCOPE)
        if(ARG_ECHO)
            message(STATUS "SPM is disabled for ${spm-gpu-0-gfx-info}")
        endif()
    endif()
endfunction()
