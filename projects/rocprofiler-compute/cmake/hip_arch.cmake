## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

function(resolve_hip_architectures)
    if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
        set(ROCPROFCOMPUTE_HIP_ARCHITECTURES_SOURCE
            "CMAKE_HIP_ARCHITECTURES"
            PARENT_SCOPE
        )
        return()
    endif()

    if(DEFINED GPU_TARGETS AND NOT "${GPU_TARGETS}" STREQUAL "")
        set(_resolved_architectures "")
        foreach(_gpu_target IN LISTS GPU_TARGETS)
            string(
                REGEX REPLACE
                ":(xnack|sramecc)[+-]"
                ""
                _hip_architecture
                "${_gpu_target}"
            )
            list(APPEND _resolved_architectures "${_hip_architecture}")
        endforeach()

        set(CMAKE_HIP_ARCHITECTURES
            "${_resolved_architectures}"
            CACHE STRING
            "HIP architectures for rocprofiler-compute test binaries"
            FORCE
        )
        message(
            STATUS
            "HIP architectures for test binaries: ${_resolved_architectures} (from GPU_TARGETS)"
        )
        set(ROCPROFCOMPUTE_HIP_ARCHITECTURES_SOURCE "GPU_TARGETS" PARENT_SCOPE)
        return()
    endif()

    set(ROCPROFCOMPUTE_HIP_ARCHITECTURES_SOURCE "implicit" PARENT_SCOPE)
endfunction()

function(warn_if_using_implicit_hip_architecture)
    if(
        NOT DEFINED ROCPROFCOMPUTE_HIP_ARCHITECTURES_SOURCE
        OR NOT ROCPROFCOMPUTE_HIP_ARCHITECTURES_SOURCE STREQUAL "implicit"
    )
        return()
    endif()

    if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
        message(
            WARNING
            "No GPU_TARGETS or CMAKE_HIP_ARCHITECTURES was provided; HIP binaries will use detected/default CMAKE_HIP_ARCHITECTURES='${CMAKE_HIP_ARCHITECTURES}'. Set -DGPU_TARGETS=... or -DCMAKE_HIP_ARCHITECTURES=... to avoid environment-dependent binaries."
        )
    else()
        message(
            WARNING
            "No GPU_TARGETS or CMAKE_HIP_ARCHITECTURES was provided, and CMake did not report a concrete HIP architecture after enabling HIP. HIP test binaries will use the compiler default. Set -DGPU_TARGETS=... or -DCMAKE_HIP_ARCHITECTURES=... to avoid environment-dependent binaries."
        )
    endif()
endfunction()
