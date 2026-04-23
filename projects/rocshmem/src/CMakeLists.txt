###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# ADD ROCSHMEM TARGET FOR FILES IN CURRENT DIRECTORY
###############################################################################
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    backend_bc.cpp
    backend_bc_device.cpp
    build_info.cpp
    constmem.cpp
    envvar.cpp
    context_host.cpp
    context_device.cpp
    mpi_instance.cpp
    rocshmem_gpu.cpp
    rocshmem.cpp
    team.cpp
    team_tracker.cpp
    util.cpp
    wf_coal_policy.cpp
    ipc_policy.cpp
    sdma_policy.cpp
)

set(
  ROCSHMEM_COMPILE_FLAGS
  -fgpu-rdc
  #   xnack allows address translation fault recovery
  #   required option for managed heap configs
  # -mxnack
)
if (BUILD_CODE_COVERAGE)
  set(ROCSHMEM_COMPILE_FLAGS ${ROCSHMEM_COMPILE_FLAGS} -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(${PROJECT_NAME} PUBLIC -fprofile-instr-generate)
endif()
target_compile_options(${PROJECT_NAME} PUBLIC ${ROCSHMEM_COMPILE_FLAGS})
#target_link_options(
  #${PROJECT_NAME}
  #PUBLIC
    #--hip-link
#)

###############################################################################
# DEVICE DEBUG INFO WORKAROUND
###############################################################################
# rocshmem_gpu.cpp and team.cpp trigger a clang/ROCm compiler bug: the HIP
# heterogeneous DWARF extensions generate invalid DIExpression metadata in
# device-side bitcode, causing the LLVM LTO verifier inside lld to abort at
# link time when any binary links against librocshmem.a with -fgpu-rdc.
#
# Root cause: the device sub-pass emits heterogeneous DWARF extensions for
# GPU-address-space-qualified variables that the LTO verifier cannot accept.
#
# Fix: -gno-heterogeneous-dwarf suppresses the heterogeneous DWARF extensions
# that generate invalid DIExpression metadata in the device sub-pass, causing
# the LLVM LTO verifier inside lld to abort at link time.  Standard DWARF
# DIExpressions are emitted instead and pass the verifier.  The flag is passed
# only to the device sub-pass via -Xarch_device; the host sub-pass continues
# to receive full -g.
set_property(
  SOURCE rocshmem_gpu.cpp team.cpp
  TARGET_DIRECTORY ${PROJECT_NAME}
  APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CONFIG:Debug,RelWithDebInfo>:-Xarch_device>"
    "$<$<CONFIG:Debug,RelWithDebInfo>:-gno-heterogeneous-dwarf>"
)

###############################################################################
# ROCSHMEM TARGET FOR BACKENDS
###############################################################################
if (USE_GDA)
add_subdirectory(gda)
endif()
if (USE_RO)
add_subdirectory(reverse_offload)
endif()
if (USE_IPC)
add_subdirectory(ipc)
endif()
if (USE_SDMA)
add_subdirectory(sdma)
endif()
add_subdirectory(host)
add_subdirectory(containers)
add_subdirectory(memory)
add_subdirectory(sync)
add_subdirectory(bootstrap)
add_subdirectory(memfabric)
