#==================================================================================================
# RCCL src/device/CMakeLists.txt
#
# PURPOSE / STATUS
# ----------------
# This file mirrors the structure of NCCL's src/device/CMakeLists.txt (tag v2.29.7-1), which
# builds the `nccl_device` OBJECT library consumed by src/CMakeLists.txt.
#
# IMPORTANT: This file is NOT currently wired into the RCCL build. RCCL's src/CMakeLists.txt does
# NOT call add_subdirectory(device); the authoritative, live device-build logic lives in:
#     * projects/rccl/src/CMakeLists.txt   (configure-time generation, source collection,
#                                           ENABLE_DEVICE_LINKER source filtering, and wire-up)
#     * projects/rccl/cmake/DeviceLinker.cmake  (the ENABLE_DEVICE_LINKER=ON device pipeline)
#
# The blocks below are a STRUCTURAL MIRROR kept here so that future merges of upstream changes to
# NCCL's src/device/CMakeLists.txt have a clear, NCCL-shaped landing spot in RCCL. The RCCL
# equivalent of each NCCL block was COPIED (not moved) from the locations above; the originals
# remain active there and carry a comment pointing back to this file. If this file is ever wired in
# via add_subdirectory(device), the duplicated logic must be removed from src/CMakeLists.txt to
# avoid double execution.
#
# Unlike NCCL (which always builds a separate nccl_device OBJECT library), RCCL has TWO device-build
# paths selected by ENABLE_DEVICE_LINKER:
#     * OFF (conventional): device sources are compiled directly into librccl with relocatable
#                           device code (-fgpu-rdc); there is NO separate device library.
#     * ON  (device linker): cmake/DeviceLinker.cmake compiles/links device code separately into
#                            DEVICE_LINKER_OBJECTS that are linked into librccl.
#==================================================================================================


#==================================================================================================
# ORIGINAL NCCL CODE -- kept verbatim for merge reference (NCCL src/device/CMakeLists.txt L1-58).
# Surrounded by a CMake block comment so it is preserved but not executed. CUDA-specific (uses
# CUDAToolkit_*, .cu sources compiled by CMake's CUDA language, and an OBJECT library); RCCL's
# equivalent appears below.
#==================================================================================================
#[[
# Run the scripts once during configuration to get the file lists
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    OUTPUT_VARIABLE files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${files}" files)
list(TRANSFORM files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    OUTPUT_VARIABLE symmetric_files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${symmetric_files}" symmetric_files)
list(TRANSFORM symmetric_files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)

# Create custom commands to generate source files with proper dependencies
add_custom_command(
    OUTPUT  ${files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating device source files"
)

add_custom_command(
    OUTPUT  ${symmetric_files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating symmetric device source files"
)

# Add library target
add_library(nccl_device OBJECT
            ${files}
            ${symmetric_files}
            ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
)

set_target_properties(nccl_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set include directories for the target
target_include_directories(nccl_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl_device nccl_header)
]]


set(DEVICE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/all_gather.h
  ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_pivot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_gda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_gda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_kernel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/op128.h
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll128.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_simple.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_kernel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/rccl_metadata.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sendrecv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/hierarchical_ag_shuffle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack_defs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather_gin.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_reduce.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/data_ops.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__funcs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/kernel.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/primitives.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter_gin.cuh
)

# Add device sources to parent scope
set(DEVICE_SOURCES ${DEVICE_SOURCES} PARENT_SCOPE)
