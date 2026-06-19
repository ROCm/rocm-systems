# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#
# RCCL Source Directory CMakeLists.txt
# This file handles source collection, hipification, and library creation
#==================================================================================================

# When RCCL is included as a subproject (e.g. rocm-systems), CMAKE_SOURCE_DIR points to the
# super-repo root. Use PROJECT_SOURCE_DIR for RCCL-relative paths.
set(RCCL_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
# Use PROJECT_BINARY_DIR to keep hipify at project root level
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
set(HIPIFY_SRC "${HIPIFY_DIR}/src")
set(GEN_DIR "${HIPIFY_DIR}/gensrc")

# Source files
set(LIBSRCFILES
    bootstrap.cc
    channel.cc
    ce_coll.cc
    collectives.cc
    debug.cc
    enqueue.cc
    group.cc
    init.cc
    proxy.cc
    transport.cc
    mnnvl.cc
    allocator.cc
    sym_kernels.cc
    dev_runtime.cc
    dev_runtime_segments.cc
    mem_manager.cc
)

# Add NVTX support if enabled
# RCCL-DELTA: if(NVTX)
    list(APPEND LIBSRCFILES init_nvtx.cc)
# RCCL-DELTA: endif()

#[[ >>> RCCL-DELTA
# Add compatibility shim if using static cudart
if(CUDARTLIB STREQUAL "cudart_static")
    list(APPEND LIBSRCFILES enhcompat.cc)
endif()
]] # <<< RCCL-DELTA

# >>> RCCL-DELTA: RCCL-only top-level sources (commDump, DDA device kernels, rccl_wrap).
# DDA sources live under algorithms/dda/ after develop commit ae0a12ac88.
list(APPEND LIBSRCFILES
    commDump.cc
    algorithms/dda/all_reduce/dda_all_reduce_ipc.cu
    algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu
    algorithms/dda/all_gather/dda_all_gather_ipc.cu
    algorithms/dda/alltoall/dda_alltoall_ipc.cu
    algorithms/dda/all_reduce/dda_all_reduce_fabric.cu
    algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu
    algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu
    algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu
    algorithms/dda/all_gather/dda_all_gather_fabric.cu
    algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu
    algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu
    algorithms/dda/alltoall/dda_alltoall_fabric.cu
    algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu
    algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu
    algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll.cu
    algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu
    algorithms/dda/device/device_buffer.cc
    algorithms/dda/fabric/fabric_gpu_barrier.cu
    algorithms/dda/fabric/fabric_init.cu
    algorithms/dda/fabric/fabric_mem_handler.cc
    algorithms/dda/ipc/ipc_gpu_barrier.cu
    algorithms/dda/ipc/ipc_init.cu
    algorithms/dda/ipc/ipc_mem_handler.cc
    rccl_wrap.cc
)
# <<< RCCL-DELTA

#[[ >>> RCCL-DELTA
# Configure pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl.pc.in
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc
    @ONLY
)
]] # <<< RCCL-DELTA

# Add files from subdirectories
add_subdirectory(transport)
add_subdirectory(misc)
add_subdirectory(register)
add_subdirectory(graph)

if (NCCL_OS_LINUX)
add_subdirectory(plugin)
add_subdirectory(ras)
endif()

add_subdirectory(device)
add_subdirectory(nccl_device)
add_subdirectory(scheduler)
add_subdirectory(os)
add_subdirectory(rma)
# >>> RCCL-DELTA: DDA headers consolidated under algorithms/dda/
add_subdirectory(algorithms/dda)
# src/include is an RCCL-only header tree (NCCL never lists headers in CMake). It
# sets INCLUDE_HEADERS so the hipify loop copies/converts every header into HIPIFY_DIR.
# (net_ib_cast is wired from within src/transport/CMakeLists.txt, mirroring how NCCL nests net_ib.)
add_subdirectory(param)
add_subdirectory(devcomm)
add_subdirectory(include)
# <<< RCCL-DELTA

# GIN is Linux-only (requires InfiniBand verbs)
if(NCCL_OS_LINUX)
    add_subdirectory(gin)
endif()

#[[ >>> RCCL-DELTA
if(NCCL_OS_LINUX)
    add_compile_options(-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=)
endif()
]] # <<< RCCL-DELTA

# >>> RCCL-DELTA
# Conditional source files based on build options
if(USE_AMDSMI)
  set(SMI_SOURCES
    include/amdsmi_wrap.h
    misc/amdsmi_wrap.cc
  )
else()
  set(SMI_SOURCES
    include/amdsmi_wrap.h
    misc/amdsmi_wrap.cc
    include/rocm_smi_wrap.h
    misc/rocm_smi_wrap.cc
  )
endif()
list(APPEND LIBSRCFILES ${SMI_SOURCES})

# Prepend source directory to all source files for proper paths from root
list(TRANSFORM LIBSRCFILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
# <<< RCCL-DELTA

# Add all source files
list(APPEND LIBSRCFILES
    ${DEVICE_SOURCES} # RCCL-DELTA
    ${TRANSPORT_SOURCES}
    ${MISC_SOURCES}
    ${REGISTER_SOURCES}
    ${GRAPH_SOURCES}
    ${PLUGIN_SOURCES}
    ${RAS_SOURCES}
    ${SYM_SOURCES}
    ${SCHEDULER_SOURCES}
    ${OS_SOURCES}
    ${DOCA_SOURCES}
    ${RMA_SOURCES}
    ${DDA_HEADERS} # RCCL-DELTA
    ${PARAM_SOURCES}
    ${INCLUDE_HEADERS} # RCCL-DELTA
    ${NET_IB_CAST_SOURCES} # RCCL-DELTA
    ${DEVCOMM_SOURCES}
)

# Add GIN sources only on Linux
if(NCCL_OS_LINUX)
    list(APPEND LIBSRCFILES ${GIN_SOURCES})
endif()

# >>> RCCL-DELTA
########################## Hipify source files #################################
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()
set(NCCL_DEVICE_HIP_FILES "")

## Loop over each source file to hipify
foreach(FULL_SRC_FILE ${LIBSRCFILES})
  # Check that file exists
  if (NOT EXISTS ${FULL_SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${FULL_SRC_FILE}")
  endif()

  file(RELATIVE_PATH SRC_FILE ${RCCL_SOURCE_DIR} ${FULL_SRC_FILE})

  # Establish hipified copy of the source file
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  # Make sure the file name is unique and there is no duplicate
  #add_file_unique(HIP_SOURCES ${HIP_FILE})

  # Convert .cu files to .cpp so that they get processed properly
  string(REPLACE "\.cuh" "\.h" HIP_FILE ${HIP_FILE})
  string(REPLACE "\.cu" "\.cu.cpp" HIP_FILE ${HIP_FILE})
  list(APPEND HIP_SOURCES ${HIP_FILE})

  if(SRC_FILE STREQUAL "src/include/nccl_device.h" OR SRC_FILE MATCHES "^src/include/nccl_device/")
    list(APPEND NCCL_DEVICE_HIP_FILES ${HIP_FILE})
  endif()

  # Create a custom command to create hipified source code
  if (FAULT_INJECTION)
    add_custom_command(
      OUTPUT ${HIP_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${FULL_SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${FULL_SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${FULL_SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${FULL_SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

# Append all found generated files to the list
list(APPEND HIP_SOURCES ${DEVICE_GEN_SOURCES})
foreach(HIP_FILE ${DEVICE_REMOVED_SOURCES})
  list(REMOVE_ITEM HIP_SOURCES ${HIP_FILE})
endforeach()
# <<< RCCL-DELTA

###################### Create a shared RCCL library ############################
add_library(rccl SHARED)

target_sources(rccl PRIVATE ${HIP_SOURCES})
if(NCCL_OS_WINDOWS)
    # target_link_libraries(nccl PRIVATE nccl_device) # RCCL-DELTA: Handled by device subdirectory
else()
    # target_sources(nccl PRIVATE $<TARGET_OBJECTS:nccl_device>) # RCCL-DELTA: Handled by device subdirectory
endif()

# Include directories; add DOCA only on non-Windows (skip for Windows platform)
set(RCCL_PRIVATE_INCLUDES
    ${PROJECT_BINARY_DIR}/include        # for generated rccl.h header
    ${HIPIFY_SRC}                        # for hipified headers
    ${HIPIFY_SRC}/device
    ${HIPIFY_SRC}/device/network/unpack
    ${HIPIFY_SRC}/include
    ${HIPIFY_SRC}/include
    ${HIPIFY_SRC}/include/mlx5
    ${HIPIFY_SRC}/include/nccl_device
    ${HIPIFY_SRC}/include/ionic
    ${HIPIFY_SRC}/include/plugin
    ${HIPIFY_SRC}/transport/net_ib_cast
    ${HIPIFY_SRC}/devcomm
    ${HIPIFY_SRC}/transport
    ${HIPIFY_SRC}/transport/net_ib
    ${GEN_DIR}
    ${HSA_INCLUDE_PATH}
    ${SMI_INCLUDE_DIR}
    ${ROCMCORE_PATH}/include
)
#[[ >>> RCCL-DELTA
if(NOT NCCL_OS_WINDOWS AND DOCA_HOME)
    list(APPEND NCCL_PRIVATE_INCLUDES ${DOCA_HOME}/include)
endif()
]] # <<< RCCL-DELTA
if(DEMANGLE_DIR)
    list(APPEND RCCL_PRIVATE_INCLUDES ${DEMANGLE_DIR})
endif()
if(ROCTX_ENABLE)
    list(APPEND RCCL_PRIVATE_INCLUDES ${ROCTRACER_INCLUDE_DIR})
endif()
target_include_directories(rccl
#[[ >>> RCCL-DELTA
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
]] # <<< RCCL-DELTA
  PRIVATE
    ${RCCL_PRIVATE_INCLUDES}
)

#[[ >>> RCCL-DELTA
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/include/nccl.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include
    COMMAND ${CMAKE_COMMAND} -DNCCL_MAJOR=${NCCL_MAJOR} -DNCCL_MINOR=${NCCL_MINOR} -DNCCL_PATCH=${NCCL_PATCH} -DNCCL_SUFFIX=${NCCL_SUFFIX} -DNCCL_VERSION_CODE=${NCCL_VERSION_CODE} -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in -DOUTPUT_FILE=${CMAKE_BINARY_DIR}/include/nccl.h -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_header.cmake
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in ${CMAKE_CURRENT_SOURCE_DIR}/generate_header.cmake
)
]]
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rccl")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in ${PROJECT_BINARY_DIR}/include/rccl/rccl.h) # For external linking
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in ${PROJECT_BINARY_DIR}/include/nccl.h)      # Used by some internal files
# <<< RCCL-DELTA

# [NCCL src/CMakeLists.txt L104-122] Copy public device headers into the build tree and build
# the nccl_header aggregate target (then add_dependencies(nccl/nccl_device nccl_header)). RCCL
# copies the hipified nccl_device headers via copy_nccl_device_headers and wires it as an rccl
# dependency (see add_dependencies at the add_library block below).
set(NCCL_DEVICE_HEADER "${PROJECT_BINARY_DIR}/include/nccl_device.h")
set(NCCL_DEVICE_HEADER_STAMP "${PROJECT_BINARY_DIR}/include/.nccl_device_headers.stamp")
# Copy hip_compat.h as-is (no hipification — contains both CUDA and HIP paths)
set(HIP_COMPAT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_device/hip_compat.h")
set(HIP_COMPAT_DST "${HIPIFY_SRC}/include/nccl_device/hip_compat.h")
add_custom_command(
  OUTPUT ${HIP_COMPAT_DST}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HIP_COMPAT_SRC} ${HIP_COMPAT_DST}
  MAIN_DEPENDENCY ${HIP_COMPAT_SRC}
  COMMENT "Copying hip_compat.h (no hipification)"
)
list(APPEND HIP_SOURCES ${HIP_COMPAT_DST})
list(APPEND NCCL_DEVICE_HIP_FILES ${HIP_COMPAT_DST})

# proxy_trace headers are not produced by hipify-perl; copy from src so the
# hipify include tree matches (unit tests search hipify paths before src/include).
set(PROXY_TRACE_HDR_SRC "${RCCL_SOURCE_DIR}/src/include/proxy_trace/proxy_trace.h")
set(PROXY_TRACE_HDR_DST "${HIPIFY_SRC}/include/proxy_trace/proxy_trace.h")
add_custom_command(
  OUTPUT ${PROXY_TRACE_HDR_DST}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${HIPIFY_SRC}/include/proxy_trace"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROXY_TRACE_HDR_SRC} ${PROXY_TRACE_HDR_DST}
  MAIN_DEPENDENCY ${PROXY_TRACE_HDR_SRC}
  COMMENT "Copying proxy_trace/proxy_trace.h (no hipification)"
)
list(APPEND HIP_SOURCES ${PROXY_TRACE_HDR_DST})

# copy_directory_if_different was added in CMake 3.26.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
  add_custom_command(
    OUTPUT ${NCCL_DEVICE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPIFY_SRC}/include/nccl_device.h" "${PROJECT_BINARY_DIR}/include/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${HIPIFY_SRC}/include/nccl_device" "${PROJECT_BINARY_DIR}/include/nccl_device"
    DEPENDS ${NCCL_DEVICE_HIP_FILES}
    VERBATIM
  )
  set(COPY_NCCL_DEVICE_HEADERS_OUTPUT ${NCCL_DEVICE_HEADER})
else()
  # On older CMake, copy_directory is unconditional and can retrigger rebuilds.
  # Use a touched stamp to keep OUTPUT >= DEPENDS after successful staging.
  add_custom_command(
    OUTPUT ${NCCL_DEVICE_HEADER_STAMP}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${HIPIFY_SRC}/include/nccl_device.h" "${PROJECT_BINARY_DIR}/include/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HIPIFY_SRC}/include/nccl_device" "${PROJECT_BINARY_DIR}/include/nccl_device"
    COMMAND ${CMAKE_COMMAND} -E touch "${NCCL_DEVICE_HEADER_STAMP}"
    DEPENDS ${NCCL_DEVICE_HIP_FILES}
    COMMENT "Staging nccl_device headers into include/"
    VERBATIM
  )
  set(COPY_NCCL_DEVICE_HEADERS_OUTPUT ${NCCL_DEVICE_HEADER_STAMP})
endif()
add_custom_target(copy_nccl_device_headers DEPENDS ${COPY_NCCL_DEVICE_HEADERS_OUTPUT})

# Adding custom target to hipify all the source files
# This is required to make sure that all the hipified source files are
# available before compiling the unit tests executable(s)
add_custom_target(hipify_all DEPENDS ${HIP_SOURCES})

# Serialize copy after full hipify: prevents previous race condition where copying of nccl device headers
# could proceed before hipify_all stage had completed.
add_dependencies(copy_nccl_device_headers hipify_all)

# net_ib/ headers collide with other basenames (common.h, gin.h, p2p.h) and are hipified
# with _tmp suffixes. Create symlinks so that #include "common.h" etc. resolve correctly
# when compiling files in hipify/src/transport/net_ib/.
#set(NET_IB_HIPIFY_DIR "${HIPIFY_SRC}/transport/net_ib")
#set(NET_IB_HEADER_SYMLINKS "")
#foreach(_hdr common gin p2p)
#  add_custom_command(
#    OUTPUT  "${NET_IB_HIPIFY_DIR}/${_hdr}.h"
#    DEPENDS "${NET_IB_HIPIFY_DIR}/${_hdr}_tmp.h"
#    COMMAND ${CMAKE_COMMAND} -E create_symlink
#            "${_hdr}_tmp.h" "${NET_IB_HIPIFY_DIR}/${_hdr}.h"
#    COMMENT "Symlinking net_ib/${_hdr}_tmp.h -> ${_hdr}.h"
#  )
#  list(APPEND NET_IB_HEADER_SYMLINKS "${NET_IB_HIPIFY_DIR}/${_hdr}.h")
#endforeach()
#add_custom_target(net_ib_header_symlinks DEPENDS ${NET_IB_HEADER_SYMLINKS})

# The top-level include/rma.h and include/rma/rma.h share the basename "rma.h",
# so add_file_unique() renames the second-listed one (include/rma/rma.h) to
# include/rma/rma_tmp.h. comm.h (included nearly everywhere, host + device) does
# #include "rma/rma.h", so symlink it back to the _tmp output, same as net_ib.
#set(RMA_HIPIFY_DIR "${HIPIFY_DIR}/src/include/rma")
#add_custom_command(
#  OUTPUT  "${RMA_HIPIFY_DIR}/rma.h"
#  DEPENDS "${RMA_HIPIFY_DIR}/rma_tmp.h"
#  COMMAND ${CMAKE_COMMAND} -E create_symlink
#          "rma_tmp.h" "${RMA_HIPIFY_DIR}/rma.h"
#  COMMENT "Symlinking rma/rma_tmp.h -> rma/rma.h"
#)
#add_custom_target(rma_header_symlinks DEPENDS "${RMA_HIPIFY_DIR}/rma.h")

# hipify_all is the contract for "the staged tree is ready to compile against",
# including for consumers that build it as a standalone prerequisite
# (test/host/run_host_tests.sh). The symlinks above are part of that tree: the
# hipified sources still #include the un-suffixed names, so staging without them
# yields a tree that cannot compile.
#add_dependencies(hipify_all net_ib_header_symlinks rma_header_symlinks)

# nccl_device/{core,comm,gin}.h share basenames with the top-level src/include/
# {core,comm,gin}.h, so add_file_unique() renames the nccl_device copies to
# *_tmp.h in the hipify tree. Hipified sources have their #includes rewritten to
# the _tmp names, but the IR bitcode wrapper (bindings/ir/nccl_device_wrapper.h)
# is staged verbatim (not hipified) and includes "nccl_device/core.h" directly.
# Symlink the un-suffixed names back to the _tmp outputs, same as net_ib/ and
# rma/ above, so the EMIT_LLVM_IR build can resolve them.
#set(NCCL_DEVICE_HIPIFY_DIR "${HIPIFY_DIR}/src/include/nccl_device")
#set(NCCL_DEVICE_HEADER_SYMLINKS "")
#foreach(_hdr core comm gin)
#  add_custom_command(
#    OUTPUT  "${NCCL_DEVICE_HIPIFY_DIR}/${_hdr}.h"
#    DEPENDS "${NCCL_DEVICE_HIPIFY_DIR}/${_hdr}_tmp.h"
#    COMMAND ${CMAKE_COMMAND} -E create_symlink
#            "${_hdr}_tmp.h" "${NCCL_DEVICE_HIPIFY_DIR}/${_hdr}.h"
#    COMMENT "Symlinking nccl_device/${_hdr}_tmp.h -> ${_hdr}.h"
#  )
#  list(APPEND NCCL_DEVICE_HEADER_SYMLINKS "${NCCL_DEVICE_HIPIFY_DIR}/${_hdr}.h")
#endforeach()
#add_custom_target(nccl_device_header_symlinks DEPENDS ${NCCL_DEVICE_HEADER_SYMLINKS})
#
## Create the symlinks before the nccl_device headers are copied into
## ${PROJECT_BINARY_DIR}/include (consumed by the IR bitcode build).
#add_dependencies(copy_nccl_device_headers nccl_device_header_symlinks)

## Set RCCL dependencies
add_dependencies(rccl copy_nccl_device_headers) # net_ib_header_symlinks rma_header_symlinks)
# <<< RCCL-DELTA

# Set version and output name
#[[ >>> RCCL-DELTA
# On Windows, use nccl.dll/nccl.lib (no "lib" prefix) so the import library matches what linkers expect
# On Linux, use libnccl.so (PREFIX "lib" for Unix convention)
set_target_properties(nccl PROPERTIES
    VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
    SOVERSION ${NCCL_MAJOR}
    OUTPUT_NAME "nccl"
)
if(NCCL_OS_LINUX)
    set_target_properties(nccl PROPERTIES PREFIX "lib")
endif()
]]
rocm_set_soversion(rccl "1.0")
# <<< RCCL-DELTA

#
# Generate nccl_git_version.h
#
#[[ >>> RCCL-DELTA
set(GIT_VERSION_FILE "${CMAKE_BINARY_DIR}/obj/include/nccl_git_version.h")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/obj/include")
add_custom_target(generate_git_version ALL
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/misc/generate_git_version.py ${GIT_VERSION_FILE}
  COMMENT "Checking git version"
  BYPRODUCTS ${GIT_VERSION_FILE}
)
add_dependencies(nccl generate_git_version)
]]
# Create initial empty file at configure time
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp "")

# Add a custom target that always runs at build time to update git version
add_custom_target(update_git_version
  ALL
  COMMAND ${CMAKE_COMMAND} -DRCCL_SOURCE_DIR=${RCCL_SOURCE_DIR} -DRCCL_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${RCCL_SOURCE_DIR}/cmake/scripts/git_version.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp
  COMMENT "Updating git version information"
  VERBATIM
)
target_sources(rccl PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

## Ensure git version is updated before building rccl
add_dependencies(rccl update_git_version)
# <<< RCCL-DELTA

#[[ >>> RCCL-DELTA
# Git version overrides (set via cmake -DNCCL_GIT_BRANCH=xxx -DNCCL_GIT_COMMIT_HASH=yyy)
set(NCCL_GIT_BRANCH "" CACHE STRING "Override git branch name")
set(NCCL_GIT_COMMIT_HASH "" CACHE STRING "Override git commit hash")
if(NCCL_GIT_BRANCH)
    target_compile_definitions(nccl PRIVATE NCCL_GIT_BRANCH="${NCCL_GIT_BRANCH}")
endif()
if(NCCL_GIT_COMMIT_HASH)
    target_compile_definitions(nccl PRIVATE NCCL_GIT_COMMIT_HASH="${NCCL_GIT_COMMIT_HASH}")
endif()

# Set CUDA specific flags (separable compilation + device link for cross-TU device symbols).
set_target_properties(nccl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
]] # <<< RCCL-DELTA

# <<< RCCL-DELTA
## Set RCCL compile definitions
if(NCCL_OS_LINUX)
  target_compile_definitions(rccl PRIVATE NCCL_OS_LINUX)
elseif(NCCL_OS_WINDOWS)
  target_compile_definitions(rccl PRIVATE NCCL_OS_WINDOWS)
endif()
if(USE_AMDSMI)
  target_compile_definitions(rccl PRIVATE USE_AMDSMI)
  if(AMDSMI_FABRIC_API)
    target_compile_definitions(rccl PRIVATE AMDSMI_FABRIC_DIRECT)
  endif()
else()
  if(HAVE_ROCM_SMI64CONFIG)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI64CONFIG)
  endif()
  if(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX)
    target_compile_definitions(rccl PRIVATE USE_ROCM_SMI_THREAD_ONLY_MUTEX)
  endif()
endif()
if(ENABLE_WARP_SPEED)
  target_compile_definitions(rccl PRIVATE ENABLE_WARP_SPEED)
endif()
if(ENABLE_ROCSHMEM)
  target_compile_definitions(rccl PRIVATE ENABLE_ROCSHMEM)
endif()
if(ENABLE_ROCSHMEM_GIN)
  target_compile_definitions(rccl PRIVATE ENABLE_ROCSHMEM_GIN)
endif()
# Uncached (fine-grained) VMM/cuMem allocation type. Available natively on
# HIP >= 7.12.60540 and via the ROCm 7.0.2.x backport range [7.0.51831, 7.0.60000).
if(HIP_VMM_UNCACHED_MEMORY_AVAILABLE AND
   (("${hip_version_string}" VERSION_GREATER_EQUAL "7.12.60540") OR
    (("${hip_version_string}" VERSION_GREATER_EQUAL "7.0.51831") AND
     ("${hip_version_string}" VERSION_LESS "7.0.60000"))))
  target_compile_definitions(rccl PRIVATE HIP_VMM_UNCACHED_MEMORY)
  message(STATUS "HIP_VMM_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_VMM_UNCACHED_MEMORY disabled")
endif()

# ==== rocSHMEM integration (optional, mutually exclusive) ====
# ENABLE_ROCSHMEM: alltoall_wg offload.  Links librocshmem.a into librccl.so
#   via -fgpu-rdc --hip-link.  Does not enable GIN plugins.
# ENABLE_ROCSHMEM_GIN: GIN plugins (GDA, SDMA).  Device symbols (QueuePair)
#   resolved via per-arch bitcode in the device linker pipeline +
#   -mlink-builtin-bitcode for symmetric kernels.  Host symbols left unresolved
#   (--allow-shlib-undefined), resolved at runtime from the executable.
#   Does not enable alltoall_wg offload.
if (ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN)
  add_rocshmem_targets()
  # Ensure rocSHMEM is fully built/installed before compiling rccl
  if (TARGET rocshmem_ext)
    add_dependencies(rccl rocshmem_ext)
  endif()

  if (ROCSHMEM_INCLUDE_DIR)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_INCLUDE_DIR})
  endif()

  if (ROCSHMEM_SOURCE_DIR)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_SOURCE_DIR}/src)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_SOURCE_DIR}/include)
    if (ROCSHMEM_BUILD_DIR)
      target_include_directories(rccl PRIVATE ${ROCSHMEM_BUILD_DIR})
    else()
      target_include_directories(rccl PRIVATE ${ROCSHMEM_SOURCE_DIR}/build/include ${ROCSHMEM_SOURCE_DIR}/build/include/rocshmem)
    endif()
  endif()
endif()
if (ENABLE_ROCSHMEM)
  target_link_libraries(rccl PRIVATE ${IBVERBS})
endif()

if(QP_TRACKING)
  target_compile_definitions(rccl PRIVATE ENABLE_QP_TRACKING)
endif()
if(${HIP_CONTIGUOUS_MEMORY})
  target_compile_definitions(rccl PRIVATE HIP_CONTIGUOUS_MEMORY)
  message(STATUS "HIP_CONTIGUOUS_MEMORY enabled")
else()
  message(STATUS "HIP_CONTIGUOUS_MEMORY disabled")
endif()
if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  target_compile_definitions(rccl PRIVATE HIP_UNCACHED_MEMORY)
  message(STATUS "HIP_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_UNCACHED_MEMORY disabled - requires HIP version >= 5.7.31920")
  # keep --hipcc-func-supp on older HIP and compiler
  if(NOT IFC_ENABLED)
    target_compile_options(rccl PRIVATE --hipcc-func-supp)
    message(STATUS "--hipcc-func-supp enabled")
  else()
    message(STATUS "--hipcc-func-supp disabled")
  endif()
endif()
if (HIP_HOST_UNCACHED_MEMORY)
  target_compile_definitions(rccl PRIVATE HIP_HOST_UNCACHED_MEMORY)
  message(STATUS "HIP_HOST_UNCACHED_MEMORY enabled")
else()
  message(STATUS "HIP_HOST_UNCACHED_MEMORY disabled")
endif()
if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
  # HIP_FABRIC_API is used as both a cmake variable (ON/OFF, for status messages below)
  # and a compile definition (bare #define, for #ifdef guards in C++). The names are
  # intentionally identical; the cmake variable is not queried by any downstream cmake logic.
  set(HIP_FABRIC_API ON)
  target_compile_definitions(rccl PRIVATE HIP_FABRIC_API)
  message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
else()
  set(HIP_FABRIC_API OFF)
  if(NOT HIP_FABRIC_API_FUNC)
    message(STATUS "hipMemImportFromShareableHandle not found - disabling HIP_FABRIC_API")
  endif()
  if(NOT HIP_FABRIC_HANDLE_TYPE)
    message(STATUS "hipMemFabricHandle_t type not found - disabling HIP_FABRIC_API")
  endif()
endif()
if (BUILD_BFD)
  if (HAVE_BFD)
    target_compile_definitions(rccl PRIVATE HAVE_BFD)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_FLAGS)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_FLAGS)
  endif()
  if (HAVE_DECL_BFD_GET_SECTION_VMA)
    target_compile_definitions(rccl PRIVATE HAVE_DECL_BFD_GET_SECTION_VMA)
  endif()
  if (HAVE_TWO_ARG_BFD_SECTION_SIZE)
    target_compile_definitions(rccl PRIVATE HAVE_TWO_ARG_BFD_SECTION_SIZE)
  endif()
endif()
if (IFC_ENABLED)
  target_compile_definitions(rccl PRIVATE USE_INDIRECT_FUNCTION_CALL)
endif()
if(DEMANGLE_DIR)
  target_compile_definitions(rccl PRIVATE "HAVE_CPLUS_DEMANGLE=1")
  target_compile_definitions(rccl PRIVATE "HAVE_DECL_BASENAME=1")
endif()
if(LL128_ENABLED)
  target_compile_definitions(rccl PRIVATE ENABLE_LL128)
endif()
if(CE_ENABLED)
  target_compile_definitions(rccl PRIVATE CE_BATCH_ASYNC_SUPPORTED)
  message(STATUS "CE_BATCH_ASYNC_SUPPORTED compile definition enabled")
endif()
if(CUMEM_DMABUF_EXPORT_ENABLED)
  target_compile_definitions(rccl PRIVATE RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED)
  message(STATUS "RCCL_CUMEM_DMABUF_EXPORT_SUPPORTED compile definition enabled")
endif()

## Set RCCL compile options
if (HAVE_PARALLEL_JOBS)
  target_compile_options(rccl PRIVATE -parallel-jobs=12)
endif()

if (ROCM_INSTALL_VERSION VERSION_GREATER_EQUAL "60200")
  target_compile_options(rccl PRIVATE --offload-compress)    # Compress GPU code at compile time.
  target_link_libraries(rccl PRIVATE --offload-compress)     # Compress GPU code at link time.
  message(STATUS "--offload-compress enabled - ROCm version >= 6.2.0")
else()
  message(STATUS "--offload-compress disabled - ROCm version < 6.2.0")
endif()

target_compile_options(rccl PRIVATE -Werror=uninitialized)
target_compile_options(rccl PRIVATE -Werror=sometimes-uninitialized)
target_compile_options(rccl PRIVATE -Wall)
target_compile_options(rccl PRIVATE -Werror=deprecated-copy-with-user-provided-copy)
target_compile_options(rccl PRIVATE -Wno-format-nonliteral)
target_compile_options(rccl PRIVATE -Wno-unused-function)

if(ENABLE_DEVICE_LINKER)
  target_compile_definitions(rccl PRIVATE RCCL_DEVICE_LINKER)
  target_compile_options(rccl PRIVATE --offload-host-only)
else()
  target_compile_options(rccl PRIVATE -fgpu-rdc)
endif()

if(QUIET_WARNINGS)
  target_compile_options(rccl PRIVATE -Wno-invalid-offsetof)
  target_compile_options(rccl PRIVATE -Wno-unused-result)
  target_compile_options(rccl PRIVATE -Wno-macro-redefined)
  target_compile_options(rccl PRIVATE -Wno-unused-label)
  target_compile_options(rccl PRIVATE -Wno-unused-variable)
  target_compile_options(rccl PRIVATE -Wno-unused-private-field)
  target_compile_options(rccl PRIVATE -Wno-null-conversion)
  target_compile_options(rccl PRIVATE -Wno-missing-braces)
endif()

## Set RCCL compile and linker options for unit tests and code coverage
if(ENABLE_CODE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(FATAL_ERROR "Code coverage is enabled, but the build type is '${CMAKE_BUILD_TYPE}'. "
                        "Code coverage requires 'Debug' build types to expose internal symbols. "
                        "Please set CMAKE_BUILD_TYPE to 'Debug' and reconfigure.")
  endif()

  message(STATUS "Code coverage is enabled with build type '${CMAKE_BUILD_TYPE}'.")

  target_compile_options(rccl PRIVATE
    -fvisibility=default
    "SHELL:-Xarch_host -fprofile-instr-generate"
    "SHELL:-Xarch_host -fcoverage-mapping")

  set(COVERAGE_SHARED_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN
  )

  set(COVERAGE_EXE_LINKER_FLAGS
    -fprofile-generate
    -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN/../lib
  )

  target_link_options(rccl PRIVATE ${COVERAGE_SHARED_LINKER_FLAGS})
  target_link_options(rccl PRIVATE ${COVERAGE_EXE_LINKER_FLAGS})

  # Expose explicit profile-flush wrappers so test harnesses that bypass
  # atexit() handlers (e.g. ProcessIsolatedTestRunner using _exit()) can
  # still flush librccl's private LLVM profile runtime instance.
  target_sources(rccl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/misc/coverage_flush.cc)
elseif(BUILD_TESTS) # Enable default/hidden visibility based on build type and ROCM_INSTALL_VERSION
  if (ROCM_INSTALL_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(rccl PRIVATE -fvisibility=default)
  else()
    target_compile_options(rccl PRIVATE -fvisibility=hidden)
  endif()
else()  # Enable hidden visibility for library without tests/code coverage enabled
  target_compile_options(rccl PRIVATE -fvisibility=hidden)
endif()

if (HAVE_KERNARG_PRELOAD AND NOT DISABLE_KERNARG_PRELOAD)
  target_compile_options(rccl PRIVATE -mllvm --amdgpu-kernarg-preload-count=16)
endif()

if (REPORT_KERNEL_RESOURCE_USE)
  target_link_options(rccl PRIVATE -Rpass-analysis=kernel-resource-usage)
endif()

if (DUMP_ASM) # Save temporary files from kernel compilation
  message(STATUS "Disassembling librccl.so to asm")
  # Maintain symbols but without changing code.  Keep additional data in dwarf section of binary.
  target_compile_options(rccl PRIVATE -gline-tables-only)
  set(OBJ_DUMP ${ROCM_PATH}/llvm/bin/llvm-objdump)

  add_custom_command(TARGET rccl POST_BUILD
    COMMENT "Disassembling RCCL library"
    COMMAND /bin/bash -c "${OBJ_DUMP} --offload-fatbin librccl.so"
    VERBATIM
  )
  foreach(GPUARCH ${GPU_TARGETS})
    add_custom_command(TARGET rccl POST_BUILD
        COMMENT "Disassembling RCCL library to dump assembly for ${GPUARCH}"
        COMMAND /bin/bash -c "${OBJ_DUMP} -d -l --source --symbolize-operands librccl.so.0.hipv4-amdgcn-amd-amdhsa--${GPUARCH} > librccl.${GPUARCH}.s"
        VERBATIM
    )
  endforeach()
endif()

## NOTE: This is currently being handled by rocm-cmake, however may need to be re-enabled in the future
#foreach(target ${GPU_TARGETS})
#  target_compile_options(rccl PRIVATE --offload-arch=${target})
#endforeach()

if(TIMETRACE)
  target_compile_options(rccl PRIVATE -ftime-trace)
endif()
if (FAULT_INJECTION)
  target_compile_definitions(rccl PRIVATE ENABLE_FAULT_INJECTION)
  message(STATUS "Fault injection enabled")
endif()

if (NOT DWORDX4_INTRINSICS)
  message(STATUS "Force disable dwordx4 intrinsics")
  target_compile_definitions(rccl PRIVATE DWORDX4_INTRINSICS_FORCE_OFF)
endif()
## Set RCCL linked library directories
target_link_directories(rccl PRIVATE ${SMI_LIB_DIR})

if (ROCM_INSTALL_VERSION VERSION_GREATER_EQUAL "60100")
    option(RCCL_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
else()
    if(RCCL_ROCPROFILER_REGISTER)
        message(AUTHOR_WARNING "RCCL_ROCPROFILER_REGISTER is not valid option for ROCm < 6.2. Current ROCm version: ${ROCM_INSTALL_VERSION}")
    endif()
    set(RCCL_ROCPROFILER_REGISTER OFF CACHE BOOL "" FORCE)
endif()
if(RCCL_ROCPROFILER_REGISTER)
  find_package(rocprofiler-register REQUIRED)
  target_compile_definitions(rccl PRIVATE RCCL_ROCPROFILER_REGISTER=1)
  target_link_libraries(
      rccl PRIVATE rocprofiler-register::rocprofiler-register)
endif()
# Device Linker: include pipeline and wire up dependencies
# (placed after all target_compile_definitions so DeviceLinker.cmake
#  can read the rccl target's definitions directly)
#==================================================================================================
# MIRROR NOTE: this wire-up (and the -fgpu-rdc / --offload-host-only selection in the compile-flags
# block above) is also mirrored as NCCL's add_library(nccl_device ...) analog in
# projects/rccl/src/device/CMakeLists.txt. This copy here is the authoritative, live one.
if(ENABLE_DEVICE_LINKER)
  include(${PROJECT_SOURCE_DIR}/cmake/DeviceLinker.cmake)
  add_dependencies(rccl device_linker_build)
  target_link_libraries(rccl PRIVATE ${DEVICE_LINKER_OBJECTS})
endif()
# <<< RCCL-DELTA

# Link libraries
#[[ >>> RCCL-DELTA
target_link_libraries(nccl
    PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
    CUDA::cudart
    ${EXTRA_LIBS}
)
]]
if (HAVE_BFD)
  target_link_libraries(rccl PRIVATE bfd)
  if(HAVE_IBERTY)
    target_link_libraries(rccl PRIVATE iberty z)
  endif()
endif()
if (ROCTX_ENABLE)
  target_link_libraries(rccl PRIVATE ${ROCTX_LIB})
endif()
if(NOT ENABLE_DEVICE_LINKER)
  target_link_libraries(rccl PRIVATE   -fgpu-rdc)             # Required when linking relocatable device code
endif()
target_link_libraries(rccl PRIVATE   Threads::Threads)
target_link_libraries(rccl INTERFACE hip::host)
if(ENABLE_DEVICE_LINKER)
  # Device code is compiled separately by the DeviceLinker pipeline.
  # Link hip::host for the runtime library but add -x hip manually so HIP
  # builtins (warpSize, etc.) are available. Crucially, do NOT propagate
  # --offload-arch from hip::device — that would make the compiler try to
  # compile device code in host-only .cc files.
  target_link_libraries(rccl PRIVATE hip::host)
  target_compile_options(rccl PRIVATE -x hip)
  # api_trace.c exists only to include api_trace.h under strict C rules; global
  # -x hip makes Clang treat it as C++ and hits #error there. Per-file -x c is
  # appended after target flags so this unit is compiled as C (same driver).
  set_source_files_properties(
    "${HIPIFY_DIR}/src/misc/api_trace.c"
    PROPERTIES
      COMPILE_OPTIONS "-x;c"
  )
else()
  target_link_libraries(rccl PRIVATE hip::device)
endif()
target_link_libraries(rccl PRIVATE   dl)
# Direct-link the HSA runtime. RCCL already depends on it transitively through the
# HIP runtime, so this is runtime-neutral. The linker records DT_NEEDED
# libhsa-runtime64.so.1 automatically (no SONAME string in source) and it resolves
# via librccl's existing RPATH, the same entry that already resolves libamdhip64.
# This replaces the previous dlopen("libhsa-runtime64.so") which broke on runtime
# trees (e.g. TheRock pip wheels) that ship no unversioned developer symlink.
target_link_libraries(rccl PRIVATE   hsa-runtime64::hsa-runtime64)
target_link_libraries(rccl PRIVATE   ${SMI_LIBRARIES})
# librocm-core provides getROCmVersion() for reporting the runtime ROCm version.
if(ROCM_MAJOR_VERSION GREATER_EQUAL 6)
  find_package(rocm-core QUIET)
  if(TARGET rocm-core)
    target_link_libraries(rccl PRIVATE rocm-core)
  else()
    find_library(ROCM_CORE_LIB NAMES rocm-core
      HINTS "${ROCMCORE_PATH}" "${ROCM_PATH}"
      PATH_SUFFIXES lib lib64)
    if(NOT ROCM_CORE_LIB)
      message(FATAL_ERROR "Could not find librocm-core")
    endif()
    target_link_libraries(rccl PRIVATE ${ROCM_CORE_LIB})
  endif()
endif()
# Wrap fmt in $<BUILD_INTERFACE:...> so it's only used while building rccl
# and is omitted from the exported rccl-targets export set. Without this,
# a FetchContent-built fmt-header-only target leaks into rccl's link
# interface and breaks install(EXPORT) for static builds.
target_link_libraries(rccl PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
if(ENABLE_ROCSHMEM)
  # librocshmem.a is compiled with -fgpu-rdc (relocatable device code).
  # --hip-link tells amdclang++ to perform the RDC device-link step as part
  # of the final librccl.so link.
  target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE hsakmt)
  target_link_libraries(rccl PRIVATE ${IBVERBS})
  set(_rocshmem_offload_arch_flags "")
  foreach(_gpu_raw ${GPU_TARGETS})
    string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
    list(APPEND _rocshmem_offload_arch_flags "--offload-arch=${_gpu}")
  endforeach()
  list(REMOVE_DUPLICATES _rocshmem_offload_arch_flags)
  target_link_options(rccl PRIVATE -fgpu-rdc --hip-link ${_rocshmem_offload_arch_flags})
elseif(ENABLE_ROCSHMEM_GIN)
  # GIN device templates (Put/Flush/etc.) reference rocshmem device symbols.
  # With the device linker (default), per-arch .bc files are injected directly
  # into the per-arch device link step via DeviceLinker.cmake.
  # Host symbols from GIN plugins are left unresolved via --allow-shlib-undefined.
  target_link_options(rccl PRIVATE "LINKER:--allow-shlib-undefined")
endif()
# <<< RCCL-DELTA

#[[ >>> RCCL-DELTA
# Add version script for symbol visibility control
target_link_options(nccl PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libnccl.map"
)

# Set output directories for nccl shared library
set_target_properties(nccl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# On Windows, export all symbols to generate import library (nccl.lib) for DLL linking
if(NCCL_OS_WINDOWS)
    set_target_properties(nccl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
]]
## Set RCCL link options
## Find out available memory
execute_process(
  COMMAND bash "-c" "test -r /sys/fs/cgroup/memory.max && cat /sys/fs/cgroup/memory.max || echo"
  OUTPUT_VARIABLE memory_max_string)
if (${memory_max_string} MATCHES "^[0-9]+")
  math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024 * 1024)")
else()
  execute_process(
    COMMAND bash "-c" "free | grep -o '[[:digit:]]*' | head -1"
    OUTPUT_VARIABLE memory_max_string)
  ## memory_max_string holds the free memory in KB
  if (${memory_max_string} MATCHES "^[0-9]+")
    math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024)") ## KB to GB conversion
  else()
    cmake_host_system_information(RESULT memory_max_string QUERY AVAILABLE_PHYSICAL_MEMORY )
    math(EXPR detected_memory_gb "${memory_max_string} / 1024")
  endif()
endif()
set(RCCL_MAX_MEMORY ${detected_memory_gb} CACHE STRING "Maximum memory (in GB) during linking")
set(RCCL_MEMORY_PER_LINK_JOB 32 CACHE STRING "Memory (in GB) reserved for each linker job")
# Use the minimum of detected memory and user-specified memory
if (RCCL_MAX_MEMORY LESS detected_memory_gb)
  set(memory_in_gb ${RCCL_MAX_MEMORY})
else()
  set(memory_in_gb ${detected_memory_gb})
endif()

## Reserve 16GB for each linker job. Limit max number of linker jobs to 16
if (HAVE_PARALLEL_JOBS)
  math(EXPR num_linker_jobs "(${memory_in_gb} + ${RCCL_MEMORY_PER_LINK_JOB} - 1) / ${RCCL_MEMORY_PER_LINK_JOB}")
  if (${num_linker_jobs} GREATER_EQUAL "16")
    set(num_linker_jobs "16")
  elseif (${num_linker_jobs} LESS_EQUAL "0")
    set(num_linker_jobs "1")
  endif()
  message(STATUS "Use ${num_linker_jobs} jobs for linking")
  target_link_options(rccl PRIVATE -parallel-jobs=${num_linker_jobs})       # Use multiple threads to link
endif()
if(BUILD_ADDRESS_SANITIZER)
  target_link_options(rccl PRIVATE -fuse-ld=lld)
  if(NOT ENABLE_DEVICE_LINKER)
    ## ---- ASAN device-link acceleration ----------------------------------------
    ## With -fgpu-rdc, the offload linker merges device bitcode from all object
    ## files and performs full LTO (optimization + codegen) for each GPU target.
    ## --lto-O0 + --lto-CGO0: skip LTO optimization and backend optimization.
    ## --------------------------------------------------------------------------
    target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-O0")
    target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
  endif()
endif()
if(CMAKE_BUILD_SUBTYPE MATCHES "DebugFast" AND NOT ENABLE_DEVICE_LINKER)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-O0")
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
endif()
if(TIMETRACE)
  target_link_options(rccl PRIVATE -ftime-trace)
endif()

# .hip_fatbin tail-placement linker script.
#
# Splices the merged .hip_fatbin output section to the end of librccl.so via
# lld's INSERT AFTER directive (the default linker script is preserved).
# Keeps text<->data distances small so host PC32/PLT32/GOTPCRELX/32S
# relocations stay in range when the bundled device image grows past
# INT32_MAX.  Applies to the host link only; the offload device-link does
# not see the script, so it works identically for ENABLE_DEVICE_LINKER=ON
# (custom pipeline) and OFF (-fgpu-rdc --hip-link).  No-op for static builds
# (which go through --emit-static-lib and never run ld).
if(BUILD_SHARED_LIBS AND RCCL_HIP_FATBIN_AT_TAIL)
  set(_rccl_fatbin_script "${PROJECT_SOURCE_DIR}/cmake/linker_scripts/hip_fatbin_tail.ld")
  if(NOT EXISTS "${_rccl_fatbin_script}")
    message(FATAL_ERROR "RCCL_HIP_FATBIN_AT_TAIL=ON but linker script not found: ${_rccl_fatbin_script}")
  endif()
  target_link_options(rccl PRIVATE "LINKER:-T,${_rccl_fatbin_script}")
  set_property(TARGET rccl APPEND PROPERTY LINK_DEPENDS "${_rccl_fatbin_script}")
  message(STATUS "RCCL: .hip_fatbin will be placed at end of librccl.so (${_rccl_fatbin_script})")

  # Post-build safety check: parse the ELF section table and confirm the
  # host sections (everything except the inert .hip_fatbin spacer) fit within
  # a 2 GiB window, which is the exact condition for small-code-model host
  # relocations (PC32/PLT32/GOTPCRELX/32S) to stay in range.  Fails the build
  # if the window exceeds 2 GiB -- e.g. if the linker silently ignored the
  # tail-placement script -- so the regression is caught here rather than as
  # a confusing relocation overflow once the fatbin grows.  Skips cleanly on
  # non-x86-64 hosts.
  add_custom_command(TARGET rccl POST_BUILD
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/cmake/scripts/check_hip_fatbin_tail.py
            $<TARGET_FILE:rccl>
    COMMENT "Verifying host sections stay within 2 GiB of each other in librccl.so"
    VERBATIM
  )
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD AND NOT ENABLE_DEVICE_LINKER AND NOT DISABLE_KERNARG_PRELOAD)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
endif()

## Track linking time
set_property(TARGET rccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
# <<< RCCL-DELTA

###################### Create a ras binary executable ############################
if(NOT NCCL_OS_WINDOWS)
set(RAS_BINSRCFILES "${HIPIFY_DIR}/src/ras/client.cc")

add_executable(rcclras ${RAS_BINSRCFILES})

target_include_directories(rcclras
  PRIVATE
    ${PROJECT_BINARY_DIR}/include
    ${HIPIFY_DIR}/src
    ${HIPIFY_DIR}/src/include
)

#[[ >>> RCCL-DELTA
add_dependencies(ncclras nccl_header)
]] # <<< RCCL-DELTA

target_link_libraries(rcclras
    PRIVATE
    hip::host
    dl
)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl hip::device)
else()
  add_dependencies(rcclras rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_CURRENT_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()

#[[ >>> RCCL-DELTA
# Set output directory for ncclras executable
set_target_properties(ncclras PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
]] # <<< RCCL-DELTA
endif()

###################### Create a param tool binary executable #####################
#[[ >>> RCCL-DELTA
if(NOT NCCL_OS_WINDOWS)
add_executable(ncclparam param/ncclparam.cc)

target_include_directories(ncclparam
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_dependencies(ncclparam nccl_header)

target_link_libraries(ncclparam
    PRIVATE
    nccl
    CUDA::cudart
)

set_target_properties(ncclparam PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
endif()
]] # <<< RCCL-DELTA

###################### Create a static RCCL library ############################
#[[ >>> RCCL-DELTA
add_library(nccl_static STATIC ${LIBSRCFILES})
if(NCCL_OS_WINDOWS)
    target_link_libraries(nccl_static PRIVATE nccl_device)
else()
    target_sources(nccl_static PRIVATE $<TARGET_OBJECTS:nccl_device>)
endif()

# Include directories
target_include_directories(nccl_static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    transport/net_ib/gdaki/doca-gpunetio/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CMAKE_BINARY_DIR}/obj/include
)

# Add dependency on nccl_header
add_dependencies(nccl_static nccl_header)
add_dependencies(nccl_static generate_git_version)

# Git version overrides for static lib
if(NCCL_GIT_BRANCH)
    target_compile_definitions(nccl_static PRIVATE NCCL_GIT_BRANCH="${NCCL_GIT_BRANCH}")
endif()
if(NCCL_GIT_COMMIT_HASH)
    target_compile_definitions(nccl_static PRIVATE NCCL_GIT_COMMIT_HASH="${NCCL_GIT_COMMIT_HASH}")
endif()

# Link libraries
target_link_libraries(nccl_static
    PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
    CUDA::cudart
    ${EXTRA_LIBS}
)

# Set CUDA specific flags (match nccl shared library for consistency).
set_target_properties(nccl_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Set output directory for nccl_static library
set_target_properties(nccl_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
]]
if(NOT BUILD_SHARED_LIBS)
  # To create a static lib with `-fgpu-rdc`, you need `--emit-static-lib` and `--hip-link`.
  # You also need to invoke amdclang++ again to trigger GPU code generation.
  set(static_link_flags
    ${CXXFLAGS}
    --hip-link
    -fgpu-rdc
    --emit-static-lib
  )

  # Find all the libraries we need to link at link time to include them in the clang link
  # command line.
  get_target_property(rccl_libs rccl LINK_LIBRARIES)
  foreach(target ${rccl_libs})
    if(TARGET ${target})
      get_target_property(location ${target} LOCATION)
      if(location)
        LIST(APPEND static_link_flags -l${location})
      endif()
    endif()
  endforeach()

  foreach(target ${GPU_TARGETS})
    list(APPEND static_link_flags --offload-arch=${target})
  endforeach()
  list(JOIN static_link_flags " " flags_str)

  # Invoking amdclang++ this way will produce a static archive, so just override ARCHIVE_CREATE.
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_CXX_COMPILER> ${flags_str} -o <TARGET> <OBJECTS>")
endif()
# <<< RCCL-DELTA

###################### Install targets / headers / pkg-config ############################
#[[ >>> RCCL-DELTA
install(TARGETS nccl nccl_static
  EXPORT NCCLTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
# On Windows, nccl/nccl_static link to nccl_device (STATIC); export set must include it.
if(NCCL_OS_WINDOWS)
  install(TARGETS nccl_device
    EXPORT NCCLTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
endif()

# Install tools, but do not export them as linkable imported targets.
if(NOT NCCL_OS_WINDOWS)
install(TARGETS ncclras ncclparam
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
endif()

# Install generated + copied public headers from the build tree.
install(DIRECTORY "${CMAKE_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Install pkg-config metadata generated during configure.
install(FILES "${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
]]
rocm_install_targets(TARGETS rccl)
rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/rccl/rccl.h ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin/nccl_net.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl)
rocm_install(FILES       ${CMAKE_CURRENT_SOURCE_DIR}/include/api_trace.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rccl/amd_detail)
rocm_install(FILES       ${PROJECT_BINARY_DIR}/include/nccl.h
                         ${PROJECT_BINARY_DIR}/include/nccl_device.h
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
rocm_install(DIRECTORY   ${PROJECT_BINARY_DIR}/include/nccl_device/
             DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nccl_device)
## Install RCCL RAS client (rcclras is built above; NCCL installs ncclras at L268-271)
rocm_install(TARGETS rcclras)
# <<< RCCL-DELTA

set(HIPIFY_DIR ${HIPIFY_DIR} PARENT_SCOPE)
