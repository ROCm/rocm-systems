# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Detect if CMake is running inside a Docker (or compatible) container.
# Used to disable tests that require GPU/hardware or host features not
# available in containers.
#
# Sets <out_var> to ON if inside a container, OFF otherwise.
# Detection: /.dockerenv exists (Docker), or /proc/1/cgroup contains
# "docker" or "containerd" (Docker/containerd/podman-compat at configure time).
#
function(rocprofiler_systems_detect_docker out_var)
    set(${out_var} OFF PARENT_SCOPE)
    # Docker and many runtimes create /.dockerenv in the container root
    if(EXISTS "/.dockerenv")
        set(${out_var} ON PARENT_SCOPE)
        return()
    endif()
    # Fallback: cgroup often shows docker/containerd paths in containers
    if(EXISTS "/proc/1/cgroup")
        file(READ "/proc/1/cgroup" _cgroup)
        if(_cgroup MATCHES "docker|containerd")
            set(${out_var} ON PARENT_SCOPE)
        endif()
    endif()
endfunction()
