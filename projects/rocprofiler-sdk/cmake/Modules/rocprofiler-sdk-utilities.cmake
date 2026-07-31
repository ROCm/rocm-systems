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
            # Fallback for environments where rocminfo enumerates no GPU agent at
            # configure time (WSL, containers without /dev/kfd, cross-compile); use the
            # GPU_TARGETS CMake variable if it was set explicitly.
            if("${rocminfo_GFXINFO}" STREQUAL ""
               AND DEFINED GPU_TARGETS
               AND NOT "${GPU_TARGETS}" STREQUAL "")
                set(rocminfo_GFXINFO "${GPU_TARGETS}")
                message(
                    STATUS
                        "${ARG_PREFIX}rocminfo returned no GPU; using GPU_TARGETS=${GPU_TARGETS}"
                    )
            endif()
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

# Reports whether the KFD device node (/dev/kfd) is present. Absence typically indicates a
# WSL2/DXG environment (or a container without KFD passthrough), where GPU work is
# scheduled through the dxg path instead of the native KFD driver. The result is cached so
# the filesystem check runs only once per configure.
function(rocprofiler_sdk_kfd_available _VAR)
    if(NOT DEFINED ROCPROFILER_SDK_KFD_AVAILABLE)
        if(EXISTS "/dev/kfd")
            set(ROCPROFILER_SDK_KFD_AVAILABLE
                ON
                CACHE INTERNAL "KFD device node present")
        else()
            set(ROCPROFILER_SDK_KFD_AVAILABLE
                OFF
                CACHE INTERNAL "KFD device node present")
        endif()
    endif()
    set(${_VAR}
        "${ROCPROFILER_SDK_KFD_AVAILABLE}"
        PARENT_SCOPE)
endfunction()

# Scale a ctest TIMEOUT for WSL2/DXG. Absence of /dev/kfd indicates a WSL2/DXG (or no-KFD)
# environment (see rocprofiler_sdk_kfd_available), where GPU work is scheduled through the
# dxg path and these workloads run slower in practice, needing more timeout headroom (root
# cause TBD). Returns BASE * 4 when /dev/kfd is absent and BASE unchanged otherwise
# (default BASE=45). Usage: rocprofiler_sdk_wsl_timeout_scale(<out_var> BASE <seconds>)
function(rocprofiler_sdk_wsl_timeout_scale _VAR)
    cmake_parse_arguments(ARG "" "BASE" "" ${ARGN})
    if(NOT DEFINED ARG_BASE)
        set(ARG_BASE 45)
    endif()
    rocprofiler_sdk_kfd_available(_KFD_AVAILABLE)
    if(NOT _KFD_AVAILABLE)
        math(EXPR _WSL_SCALED_TIMEOUT "${ARG_BASE} * 4")
    else()
        set(_WSL_SCALED_TIMEOUT "${ARG_BASE}")
    endif()
    set(${_VAR}
        "${_WSL_SCALED_TIMEOUT}"
        PARENT_SCOPE)
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
