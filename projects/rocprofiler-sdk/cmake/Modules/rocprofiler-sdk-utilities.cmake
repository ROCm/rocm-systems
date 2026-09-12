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

# Minimum amdgpu kernel module version for SPM (see source/docs/how-to/using-spm.rst).
# Parsed from spm_runner_preflight.py so CMake and CI always share one constant.
function(rocprofiler_sdk_get_spm_preflight_script OUT_VAR)
    set(_candidates "")

    # rocprofiler-sdk source/build tree.
    list(APPEND _candidates
         "${CMAKE_CURRENT_LIST_DIR}/../../tests/spm_runner_preflight.py")

    # Installed rocprofiler-sdk package: Modules -> lib/cmake/rocprofiler-sdk/Modules.
    list(
        APPEND
        _candidates
        "${CMAKE_CURRENT_LIST_DIR}/../../../../share/rocprofiler-sdk/tests/spm_runner_preflight.py"
        )

    # Installed rocprofiler-sdk package (find_package consumers and test installs).
    if(DEFINED PACKAGE_PREFIX_DIR)
        list(
            APPEND
            _candidates
            "${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_DATADIR}/rocprofiler-sdk/tests/spm_runner_preflight.py"
            )
    endif()

    foreach(_script IN LISTS _candidates)
        if(EXISTS "${_script}")
            set(${OUT_VAR}
                "${_script}"
                PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${OUT_VAR}
        ""
        PARENT_SCOPE)
endfunction()

function(rocprofiler_sdk_load_spm_min_amdgpu_driver_version OUT_VAR)
    rocprofiler_sdk_get_spm_preflight_script(_preflight)
    if(_preflight STREQUAL "")
        message(
            FATAL_ERROR
                "Cannot locate spm_runner_preflight.py to read SPM_MIN_AMDGPU_DRIVER_VERSION"
            )
    endif()

    file(READ "${_preflight}" _preflight_py)
    string(REGEX MATCH "SPM_MIN_AMDGPU_DRIVER_VERSION[ \t]*=[ \t]*\"([0-9.]+)\"" _match
                 "${_preflight_py}")
    if(NOT CMAKE_MATCH_1)
        message(
            FATAL_ERROR "Failed to parse SPM_MIN_AMDGPU_DRIVER_VERSION from ${_preflight}"
            )
    endif()

    set(${OUT_VAR}
        "${CMAKE_MATCH_1}"
        PARENT_SCOPE)
endfunction()

rocprofiler_sdk_load_spm_min_amdgpu_driver_version(
    _rocprofiler_spm_min_amdgpu_driver_version)
set(ROCPROFILER_SPM_MIN_AMDGPU_DRIVER_VERSION
    "${_rocprofiler_spm_min_amdgpu_driver_version}"
    CACHE
        STRING
        "Minimum /sys/module/amdgpu/version for SPM tests (from spm_runner_preflight.py)"
        FORCE)

function(rocprofiler_sdk_read_amdgpu_driver_version OUT_VAR)
    set(_path "/sys/module/amdgpu/version")
    if(EXISTS "${_path}")
        file(READ "${_path}" _ver)
        string(STRIP "${_ver}" _ver)
    else()
        set(_ver "")
    endif()
    set(${OUT_VAR}
        "${_ver}"
        PARENT_SCOPE)
endfunction()

function(rocprofiler_sdk_version_ge VERSION_A VERSION_B OUT_VAR)
    if("${VERSION_A}" VERSION_GREATER_EQUAL "${VERSION_B}")
        set(_result TRUE)
    else()
        set(_result FALSE)
    endif()
    set(${OUT_VAR}
        "${_result}"
        PARENT_SCOPE)
endfunction()

function(rocprofiler_sdk_spm_driver_supported OUT_VAR)
    rocprofiler_sdk_read_amdgpu_driver_version(_current)
    if(_current STREQUAL "")
        set(${OUT_VAR}
            FALSE
            PARENT_SCOPE)
        return()
    endif()
    rocprofiler_sdk_version_ge("${_current}"
                               "${ROCPROFILER_SPM_MIN_AMDGPU_DRIVER_VERSION}" _ok)
    set(${OUT_VAR}
        "${_ok}"
        PARENT_SCOPE)
endfunction()

# Combines arch and driver gates for SPM integration tests. Not wired into the
# hard-disabled integration suites yet; use when selectively re-enabling tests.
function(rocprofiler_sdk_spm_tests_disabled OUT_VAR)
    rocprofiler_sdk_spm_disabled(_arch_disabled)
    rocprofiler_sdk_spm_driver_supported(_driver_ok)
    if(_arch_disabled OR NOT _driver_ok)
        set(${OUT_VAR}
            TRUE
            PARENT_SCOPE)
        if(NOT _driver_ok)
            rocprofiler_sdk_read_amdgpu_driver_version(_current)
            message(
                STATUS
                    "SPM tests disabled: amdgpu driver '${_current}' < ${ROCPROFILER_SPM_MIN_AMDGPU_DRIVER_VERSION}"
                )
        endif()
    else()
        set(${OUT_VAR}
            FALSE
            PARENT_SCOPE)
    endif()
endfunction()
