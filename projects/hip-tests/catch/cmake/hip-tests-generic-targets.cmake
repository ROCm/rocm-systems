# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Keep generic-target tests buildable when the installed compiler predates a
# target added to hip-tests. Preserve every candidate the compiler supports,
# including the feature variants required by gfx9-4-generic.
function(hip_tests_get_generic_target_options output_variable targets_variable)
    # Global properties last for this configure only. Reuse probes across test
    # directories without carrying stale toolchain results into a reconfigure.
    string(
        SHA256 _probe_key
        "${CMAKE_HIP_COMPILER}|${HIP_DEVICE_BUILD_FLAGS_NO_ARCH}|${HIP_PATH_OPT}"
    )
    set(_property "HIP_TESTS_GENERIC_TARGETS_${_probe_key}")
    get_property(_probed GLOBAL PROPERTY "${_property}" SET)
    if(_probed)
        get_property(_targets GLOBAL PROPERTY "${_property}")
    else()
        set(_generic_targets
            gfx9-generic
            gfx9-4-generic:sramecc+:xnack-
            gfx9-4-generic:sramecc-:xnack-
            gfx9-4-generic:sramecc+:xnack+
            gfx9-4-generic:sramecc-:xnack+
            gfx10-1-generic
            gfx10-3-generic
            gfx11-generic
            gfx11-7-generic
            gfx12-generic
            gfx12-5-generic
        )

        # A recognized CPU name is insufficient: an older ROCm installation can lack
        # that target's device libraries. Ask the HIP driver to plan the same device
        # compilation as the tests; -### validates targets and libraries without
        # compiling or running a program.
        set(_probe_source
            "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hip-tests-generic-target.hip"
        )
        file(
            WRITE "${_probe_source}"
            "extern \"C\" __global__ void generic_target_probe() {}\n"
        )
        set(_targets)
        set(_diagnostics)
        foreach(_target IN LISTS _generic_targets)
            execute_process(
                COMMAND
                    "${CMAKE_HIP_COMPILER}" ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH}
                    ${HIP_PATH_OPT} "-###" --cuda-device-only
                    "--offload-arch=${_target}" -x hip -c "${_probe_source}" -o
                    "${_probe_source}.o"
                RESULT_VARIABLE _probe_result
                OUTPUT_VARIABLE _probe_stdout
                ERROR_VARIABLE _probe_stderr
            )
            if(_probe_result EQUAL 0)
                list(APPEND _targets "${_target}")
            else()
                message(
                    STATUS
                    "HIP toolchain does not support generic target ${_target}"
                )
                string(
                    APPEND _diagnostics
                    "\n${_target} (result: ${_probe_result}):\n${_probe_stdout}${_probe_stderr}"
                )
            endif()
        endforeach()
        if(NOT _targets)
            message(
                FATAL_ERROR
                "The HIP toolchain supports none of the generic targets required by hip-tests:\n"
                "${_diagnostics}"
            )
        endif()
        set_property(GLOBAL PROPERTY "${_property}" "${_targets}")
    endif()

    set(_options)
    foreach(_target IN LISTS _targets)
        list(APPEND _options "--offload-arch=${_target}")
    endforeach()
    set(${output_variable} "${_options}" PARENT_SCOPE)
    set(${targets_variable} "${_targets}" PARENT_SCOPE)
endfunction()
