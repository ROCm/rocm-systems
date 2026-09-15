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
            set(rocminfo_SUCCEEDED ON)
            string(REGEX MATCHALL "gfx([0-9A-Fa-f]+)" rocminfo_GFXINFO "${rocminfo_OUT}")
            list(REMOVE_DUPLICATES rocminfo_GFXINFO)
        else()
            message(
                AUTHOR_WARNING
                    "${rocminfo_EXECUTABLE} returned ${rocminfo_RET}\nstderr:\n${rocminfo_ERR}\nstdout:\n${rocminfo_OUT}"
                )
        endif()
    endif()

    # Fallback for environments where rocminfo is unavailable or cannot enumerate a GPU at
    # configure time (WSL, containers without /dev/kfd, cross-compile).
    if("${rocminfo_GFXINFO}" STREQUAL "" AND NOT "${GPU_TARGETS}" STREQUAL "")
        set(rocminfo_GFXINFO "${GPU_TARGETS}")
        message(STATUS "using explicit GPU_TARGETS=${GPU_TARGETS}")
    endif()

    # Preserve the prior behavior when rocminfo cannot run and no fallback was provided.
    if(rocminfo_SUCCEEDED OR rocminfo_GFXINFO)
        set(${_VAR}
            "${rocminfo_GFXINFO}"
            PARENT_SCOPE)

        if(ARG_ECHO)
            string(REPLACE ";" "${ARG_DELIM}" _GFXINFO_ECHO "${rocminfo_GFXINFO}")
            message(STATUS "System architectures: ${_GFXINFO_ECHO}")
        endif()
    endif()
endfunction()

# Reports whether the KFD device node (/dev/kfd) is present. Absence typically indicates a
# WSL2/DXG environment (or a container without KFD passthrough), where GPU work is
# scheduled through the dxg path instead of the native KFD driver.
#
# The answer describes the machine running cmake, which is not necessarily the machine
# running ctest: ROCm is routinely configured inside a container that does not pass
# /dev/kfd through even though the bare-metal host does have it. Set
# ROCPROFILER_SDK_ASSUME_KFD to say so explicitly (ON to keep the KFD-gated tests and the
# unscaled timeouts, OFF to force the no-KFD behavior); it overrides the probe entirely.
# Deliberately not cached, so a configure-host change is picked up by re-running cmake
# rather than needing the build tree wiped.
function(rocprofiler_sdk_kfd_available _VAR)
    if(DEFINED ROCPROFILER_SDK_ASSUME_KFD)
        set(_KFD_AVAILABLE "${ROCPROFILER_SDK_ASSUME_KFD}")
    elseif(EXISTS "/dev/kfd")
        set(_KFD_AVAILABLE ON)
    else()
        set(_KFD_AVAILABLE OFF)
    endif()
    set(${_VAR}
        "${_KFD_AVAILABLE}"
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

# Reports whether the hsakmt headers this build compiles against can describe a WSL/DXG
# KMT node. The enumerator needs the adapter LUID it matches DXCore adapters on, the
# HsaStructureSizes handshake it validates the thunk's record layout with, and the gfx
# version override the thunk reports. ROCm 6.2 through 6.4 declare neither the LUID pair
# nor HsaStructureSizes, and 6.2 not the override either. Those releases ship no librocdxg
# at all, so a header set this probe turns down is also one with no thunk for the
# enumerator to read.
#
# Probed rather than derived from a version number on purpose: HsaNodeProperties has
# changed without the ROCm version saying so, and the LUID pair postdates several 7.x
# releases, so only the headers can answer this. The result is cached so the compile probe
# runs once per configure.
function(rocprofiler_sdk_dxg_topology_supported _VAR)
    if(NOT DEFINED ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
        include(CheckCXXSourceCompiles)

        # find_package(hsakmt) may report its include directories as either ordinary or
        # system includes, and only one of the two targets exists in a standalone
        # configure, so collect whichever are present.
        #
        # The target's own directories go first because they are what the real compile
        # sees; ${ROCM_PATH}/include is a fallback for a standalone configure in which
        # neither target exists. Probing ROCM_PATH first reads the system headers even
        # when hsakmt resolved to a different prefix, and a staged install beside an older
        # /opt/rocm then gets the opposite answer.
        set(_DXG_PROBE_INCLUDES)
        foreach(_TARGET hsakmt::hsakmt rocprofiler-sdk-hsakmt-nolink)
            if(TARGET ${_TARGET})
                foreach(_PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
                    get_target_property(_DIRS ${_TARGET} ${_PROPERTY})
                    if(_DIRS)
                        list(APPEND _DXG_PROBE_INCLUDES ${_DIRS})
                    endif()
                endforeach()
            endif()
        endforeach()
        list(APPEND _DXG_PROBE_INCLUDES "${ROCM_PATH}/include")
        list(REMOVE_DUPLICATES _DXG_PROBE_INCLUDES)

        set(CMAKE_REQUIRED_INCLUDES ${_DXG_PROBE_INCLUDES})
        set(CMAKE_REQUIRED_QUIET ON)
        check_cxx_source_compiles(
            "
#include <hsakmt/hsakmttypes.h>

int main()
{
    HsaNodeProperties props = {};
    HsaStructureSizes sizes = {};
    sizes.StructureSizes          = sizeof(HsaStructureSizes);
    sizes.SizeOfHsaNodeProperties = sizeof(HsaNodeProperties);
    return (int) (props.LuidLowPart + props.LuidHighPart +
                  props.OverrideEngineId.ui32.Major + sizes.SizeOfHsaNodeProperties);
}
"
            ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED)
    endif()
    set(${_VAR}
        "${ROCPROFILER_SDK_DXG_TOPOLOGY_SUPPORTED}"
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
