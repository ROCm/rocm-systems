#
# Miscellaneous cmake functions for rocprofiler-sdk
#

include_guard(GLOBAL)

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
        execute_process(
            COMMAND ${rocminfo_EXECUTABLE}
            RESULT_VARIABLE rocminfo_RET
            OUTPUT_VARIABLE rocminfo_OUT
            ERROR_VARIABLE rocminfo_ERR
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

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
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx95[0-9]$")
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
       OR "${pc-sampling-gpu-0-gfx-info}" MATCHES "^gfx95[0-9]$")
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
