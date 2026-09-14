# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Backends that expose a simulated device to a virtual machine monitor. Each one
# binds a transport-agnostic device model to one VMM interface, so these are the
# only places an external VMM library may be linked.
if(ROCJITSU_ENABLE_VFIO)
    add_subdirectory(vfu)
endif()
