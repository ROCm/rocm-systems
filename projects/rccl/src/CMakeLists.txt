# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#
# RCCL Source Directory CMakeLists.txt
# This file handles source collection, hipification, and library creation
#==================================================================================================

# When RCCL is included as a subproject (e.g. rocm-systems), CMAKE_SOURCE_DIR points to the
# super-repo root. Use PROJECT_SOURCE_DIR for RCCL-relative paths.
set(RCCL_SOURCE_DIR "${PROJECT_SOURCE_DIR}")

# Collect headers + RCCL-managed source files
#==================================================================================================
# This list intentionally holds ONLY:
#   * every header that must be hipified into the staging tree (including headers that
#     live under wired subdirectories - the leaf CMakeLists.txt track sources, not headers), and
#   * compiled sources that live directly under src/ with no per-directory CMakeLists.txt
#     (the top-level src/ translation units + the hand-hipified device/*.cu sources).
#
# Compiled sources that live under a wired subdirectory (misc, graph, register, ras, scheduler,
# os, gin, rma, plugin, nccl_device, transport) are pulled from those leaf CMakeLists.txt files
# further below. Per-subdirectory RCCL deltas (RCCL-only sources, excluded NCCL sources, and
# CUDA->ROCm build swaps) live inside the leaf files in fenced RCCL-DELTA blocks, so this list
# must stay free of subdir sources for them to take effect.
set(SRC_FILES
  allocator.cc
  bootstrap.cc
  ce_coll.cc
  channel.cc
  collectives.cc
  commDump.cc
  mem_manager.cc
  debug.cc
  dev_runtime.cc
  dda_all_reduce_ipc.cu
  device_buffer.cc
  enqueue.cc
  group.cc
  ipc_gpu_barrier.cu
  ipc_init.cu
  ipc_mem_handler.cc
  init.cc
  init_nvtx.cc
  mnnvl.cc
  proxy.cc
  rccl_wrap.cc
  sym_kernels.cc
  transport.cc
  # device/*.h, device/network/**.h and device/symmetric/*.{h,cuh} headers are staged via
  # src/device/CMakeLists.txt (DEVICE_SOURCES). The two hand-written device sources below stay
  # here because they are routed through the RCCL device-linker filters in this file.
  device/common.cu
  device/onerank.cu
  # graph/*.h headers are staged via src/graph/CMakeLists.txt.
  include/alloc.h
  include/mem_manager.h
  include/allocator.h
  include/algorithms/CollCommon.h
  include/algorithms/all_reduce/all_reduce_dda.h
  include/alt_rsmi.h
  include/archinfo.h
  include/api_trace.h
  include/argcheck.h
  include/BfdBacktrace.hpp
  include/bitops.h
  include/bootstrap.h
  include/ce_coll.h
  include/ce_fault_inject.h
  include/channel.h
  include/checks.h
  include/collectives.h
  include/coll_net.h
  include/cudawrap.h
  include/comm.h
  include/compiler.h
  include/compiler/gcc.h
  include/compiler/msvc.h
  include/core.h
  include/cpuset.h
# include/cudawrap.h
  include/dda_all_reduce_ipc.h
  include/debug.h
  include/dev_runtime.h
  include/device.h
  include/env.h
  include/device_buffer.h
  include/enqueue.h
  include/gdrwrap.h
  include/git_version.h
  include/graph.h
  include/group.h
  include/gin.h
  include/gin/gin_host.h
  include/gin/gin_host_proxy.h
  include/hip_rocm_version_info.h
  include/ibvcore.h
  include/ibvsymbols.h
  include/ibvwrap.h
  include/info.h
  include/ipc_gpu_barrier.h
  include/ipc_init.h
  include/ipc_init_detail.h
  include/ipc_mem_handler.h
  include/ipcsocket.h
  include/mem_manager.h
  include/mnnvl.h
  include/nccl_common.h
  include/nccl_device.h
  include/nccl_device/net_device.h
  include/net.h
  include/nvmlwrap.h
  include/nvtx.h
  include/nvtx_payload_schemas.h
  include/nvtx_stub.h
  include/p2p.h
  include/param.h
  include/profiler.h
  include/proxy.h
  include/ras.h
  include/rccl_common.h
  include/rccl_vars.h
  include/register.h
  include/register_inline.h
  include/rccl_float8.h
  include/rocmwrap.h
  include/roctx.h
  include/recorder.h
  include/scheduler.h
  include/shm.h
  include/shmutils.h
  include/signals.h
  include/socket.h
  include/strongstream.h
  include/sym_kernels.h
  include/timer.h
  include/transport.h
  include/trees.h
  include/tuner.h
  include/utils.h
  include/mlx5/mlx5dvcore.h
  include/mlx5/mlx5dvsymbols.h
  include/mlx5/mlx5dvwrap.h
  include/ionic/ionicdvcore.h
  include/ionic/ionicdvsymbols.h
  include/ionic/ionicdvwrap.h
  include/nccl_device/comm.h
  include/nccl_device/coop.h
  include/nccl_device/core.h
  include/nccl_device/barrier.h
  include/nccl_device/gin.h
  include/nccl_device/gin_barrier.h
  include/nccl_device/gin/gdaki/gin_gdaki.h
  include/nccl_device/gin/gdaki/gin_gdaki_device_host_common.h
  include/nccl_device/gin/gin_device_api.h
  include/nccl_device/gin/gin_device_common.h
  include/nccl_device/gin/gin_device_host_common.h
  include/nccl_device/gin/proxy/gin_proxy.h
  include/nccl_device/gin/proxy/gin_proxy_device_host_common.h
  include/nccl_device/ll_a2a.h
  include/nccl_device/lsa_barrier.h
  include/nccl_device/ptr.h
  include/nccl_device/rccl_ptr.h
  include/nccl_device/utility.h
  include/nccl_device/impl/comm__funcs.h
  include/nccl_device/impl/comm__types.h
  include/nccl_device/impl/core__funcs.h
  include/nccl_device/impl/core__types.h
  include/nccl_device/impl/barrier__funcs.h
  include/nccl_device/impl/barrier__types.h
  include/nccl_device/impl/gin__funcs.h
  include/nccl_device/impl/gin__types.h
  include/nccl_device/impl/gin_barrier__funcs.h
  include/nccl_device/impl/gin_barrier__types.h
  include/nccl_device/impl/ll_a2a__funcs.h
  include/nccl_device/impl/ll_a2a__types.h
  include/nccl_device/impl/lsa_barrier__funcs.h
  include/nccl_device/impl/lsa_barrier__types.h
  include/nccl_device/impl/ptr__funcs.h
  include/nccl_device/impl/ptr__types.h
  include/nccl_device/impl/multimem__funcs.h
  include/nccl_device/impl/reduce_copy__funcs.h
  include/nccl_device/impl/reduce_copy__impl.h
  include/nccl_device/impl/reduce_copy__types.h
  include/nccl_device/impl/vector__funcs.h
  include/nccl_device/impl/vector__types.h
  include/nccl_device/reduce_copy.h
  include/npkit/npkit.h
  include/npkit/npkit_event.h
  include/npkit/npkit_struct.h
  include/nvtx3/nvToolsExt.h
  include/nvtx3/nvToolsExtCounters.h
  include/nvtx3/nvToolsExtCuda.h
  include/nvtx3/nvToolsExtCudaRt.h
  include/nvtx3/nvToolsExtMem.h
  include/nvtx3/nvToolsExtMemCudaRt.h
  include/nvtx3/nvToolsExtOpenCL.h
  include/nvtx3/nvToolsExtPayload.h
  include/nvtx3/nvToolsExtPayloadHelper.h
  include/nvtx3/nvToolsExtSemanticsCounters.h
  include/nvtx3/nvToolsExtSemanticsScope.h
  include/nvtx3/nvToolsExtSync.h
  include/nvtx3/nvtx3.hpp
  include/nvtx3/nvtxDetail/nvtxExtHelperMacros.h
  include/nvtx3/nvtxDetail/nvtxExtImpl.h
  include/nvtx3/nvtxDetail/nvtxExtImplCounters_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplMem_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplMemCudaRt_v1.h
  include/nvtx3/nvtxDetail/nvtxExtImplPayload_v1.h
  include/nvtx3/nvtxDetail/nvtxExtInit.h
  include/nvtx3/nvtxDetail/nvtxExtPayloadHelperInternal.h
  include/nvtx3/nvtxDetail/nvtxExtPayloadTypeInfo.h
  include/nvtx3/nvtxDetail/nvtxExtTypes.h
  include/nvtx3/nvtxDetail/nvtxImpl.h
  include/nvtx3/nvtxDetail/nvtxImplCore.h
  include/nvtx3/nvtxDetail/nvtxImplCuda_v3.h
  include/nvtx3/nvtxDetail/nvtxImplCudaRt_v3.h
  include/nvtx3/nvtxDetail/nvtxImplOpenCL_v3.h
  include/nvtx3/nvtxDetail/nvtxImplSync_v3.h
  include/nvtx3/nvtxDetail/nvtxInit.h
  include/nvtx3/nvtxDetail/nvtxInitDecls.h
  include/nvtx3/nvtxDetail/nvtxInitDefs.h
  include/nvtx3/nvtxDetail/nvtxLinkOnce.h
  include/nvtx3/nvtxDetail/nvtxTypes.h
  include/proxy_trace/proxy_trace.h
  include/plugin/nccl_net.h
  include/plugin/nccl_profiler.h
  include/plugin/nccl_tuner.h
  include/plugin/nccl_env.h
  include/plugin/env/env_v1.h
  include/plugin/plugin.h
  include/plugin/net/net_v6.h
  include/plugin/net/net_v7.h
  include/plugin/net/net_v8.h
  include/plugin/net/net_v9.h
  include/plugin/net/net_v10.h
  include/plugin/net/net_v11.h
  include/plugin/profiler/net_ib_v1.h
  include/plugin/profiler/net_ib.h
  include/plugin/profiler/net_socket_v1.h
  include/plugin/profiler/net_socket.h
  include/plugin/profiler/profiler_v1.h
  include/plugin/profiler/profiler_v2.h
  include/plugin/profiler/profiler_v3.h
  include/plugin/profiler/profiler_v4.h
  include/plugin/profiler/profiler_v5.h
  include/plugin/profiler/profiler_v6.h
  include/plugin/tuner/tuner_v2.h
  include/plugin/tuner/tuner_v3.h
  include/plugin/tuner/tuner_v4.h
  include/plugin/tuner/tuner_v5.h
  include/plugin/nccl_gin.h
  include/plugin/gin/gin_v11.h
  include/plugin/gin/gin_v12.h
  include/os.h
  include/rma/rma.h
  include/rma/rma_ce.h
  include/rma/rma_proxy.h
  # ras/ras_internal.h is staged via src/ras/CMakeLists.txt.
  # transport/*.h (incl. net_ib_cast/*.h, net_ib/gin.h, net_ib/gdaki/gin_host_gdaki.h) are
  # staged via src/transport/CMakeLists.txt and src/transport/net_ib_cast/CMakeLists.txt.
  # net_ib_rocm.cc is generated by rocmIb.cmake directly into the hipify staging area
  # so it is not listed here - it's added to HIP_SOURCES separately below
  include/latency_profiler/CollTrace.h
  include/latency_profiler/CollTraceEvent.h
  include/latency_profiler/CollTraceFunc.h
  include/latency_profiler/CollTraceUtils.h
  include/latency_profiler/EventQueue.h
)

# Prepend src/ to all source files for proper paths from root
list(TRANSFORM SRC_FILES PREPEND "src/")

# AMD SMI wrapper headers (conditional). The matching .cc sources are owned by
# src/misc/CMakeLists.txt; only the headers need to be staged for hipify here.
#==================================================================================================
list(APPEND SRC_FILES src/include/amdsmi_wrap.h)
if(NOT USE_AMDSMI)
  list(APPEND SRC_FILES src/include/rocm_smi_wrap.h)
endif()

# Hipify staging directories
#==================================================================================================
# Defined here (before the subdirectories are added) because the wired device/ leaf generates
# its sources directly into ${GEN_DIR} / ${GEN_SYM_DIR} at configure time.
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
set(GEN_DIR "${HIPIFY_DIR}/gensrc")
set(GEN_SYM_DIR "${GEN_DIR}/symmetric")

# Compiled translation units sourced from per-directory CMakeLists.txt (NCCL-aligned)
#==================================================================================================
# Each leaf CMakeLists.txt owns its subdirectory's source list, so upstream NCCL source-list
# changes merge cleanly into the leaf AND take effect here. RCCL's per-subdirectory deltas live
# inside the leaves in fenced RCCL-DELTA blocks:
#   * misc/      - excludes cudawrap/gdrwrap/nvmlwrap/git_version; adds RCCL-only sources + SMI .cc
#   * graph/     - adds rome_models.cc / rome_topo_consensus.cc
#   * ras/       - adds client.cc (also compiled into librccl)
#   * transport/ - adds monolithic net_ib.cc + net_ib_cast; disables modular net_ib/ + DOCA
#   * device/    - replaces NCCL's CUDA OBJECT library with RCCL's HIP generate.py into ${GEN_DIR}
add_subdirectory(misc)
add_subdirectory(graph)
add_subdirectory(register)
add_subdirectory(ras)
add_subdirectory(scheduler)
add_subdirectory(os)
add_subdirectory(gin)
add_subdirectory(rma)
add_subdirectory(plugin)
add_subdirectory(nccl_device)
add_subdirectory(transport)
add_subdirectory(device)

set(SUBDIR_LIB_SOURCES
  ${MISC_SOURCES}
  ${GRAPH_SOURCES}
  ${REGISTER_SOURCES}
  ${RAS_SOURCES}
  ${SCHEDULER_SOURCES}
  ${OS_SOURCES}
  ${GIN_SOURCES}
  ${RMA_SOURCES}
  ${PLUGIN_SOURCES}
  ${SYM_SOURCES}
  ${TRANSPORT_SOURCES}
  ${DEVICE_SOURCES}
)

# Leaf files emit absolute paths (${CMAKE_CURRENT_SOURCE_DIR}/...). Convert each to a
# project-root-relative path (src/<dir>/<file>) to match the hipify staging convention.
set(SUBDIR_LIB_SOURCES_REL "")
foreach(_abs IN LISTS SUBDIR_LIB_SOURCES)
  file(RELATIVE_PATH _rel "${PROJECT_SOURCE_DIR}" "${_abs}")
  list(APPEND SUBDIR_LIB_SOURCES_REL "${_rel}")
endforeach()
list(APPEND SRC_FILES ${SUBDIR_LIB_SOURCES_REL})

# net_ib_cast sources already carry the src/ prefix (exported by the transport/ leaf).
list(APPEND SRC_FILES ${NET_IB_CAST_SOURCES})

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()
# HIPIFY_DIR / GEN_DIR / GEN_SYM_DIR were defined earlier (before the subdirectories were
# added) so the wired device/ leaf could generate into the staging tree.
set(NCCL_DEVICE_HIP_FILES "")

## Loop over each source file to hipify
foreach(SRC_FILE ${SRC_FILES})
  # Check that file exists
  if (NOT EXISTS ${RCCL_SOURCE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${RCCL_SOURCE_DIR}/${SRC_FILE}")
  endif()

  # Establish hipified copy of the source file
  set(HIP_FILE "${HIPIFY_DIR}/${SRC_FILE}")
  get_filename_component(HIP_FILE_DIR ${HIP_FILE} DIRECTORY)

  # Make sure the file name is unique and there is no duplicate
  add_file_unique(HIP_SOURCES ${HIP_FILE})

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
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${RCCL_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${RCCL_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -experimental -quiet-warnings ${RCCL_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${RCCL_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${RCCL_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

set(NCCL_DEVICE_HEADER "${PROJECT_BINARY_DIR}/include/rccl/nccl_device.h")
# Copy hip_compat.h as-is (no hipification — contains both CUDA and HIP paths)
set(HIP_COMPAT_SRC "${RCCL_SOURCE_DIR}/src/include/nccl_device/hip_compat.h")
set(HIP_COMPAT_DST "${HIPIFY_DIR}/src/include/nccl_device/hip_compat.h")
add_custom_command(
  OUTPUT ${HIP_COMPAT_DST}
  COMMAND ${CMAKE_COMMAND} -E copy ${HIP_COMPAT_SRC} ${HIP_COMPAT_DST}
  MAIN_DEPENDENCY ${HIP_COMPAT_SRC}
  COMMENT "Copying hip_compat.h (no hipification)"
)
list(APPEND HIP_SOURCES ${HIP_COMPAT_DST})
list(APPEND NCCL_DEVICE_HIP_FILES ${HIP_COMPAT_DST})

add_custom_command(
  OUTPUT ${NCCL_DEVICE_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/include"
  COMMAND ${CMAKE_COMMAND} -E copy "${HIPIFY_DIR}/src/include/nccl_device.h" "${PROJECT_BINARY_DIR}/include/"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${HIPIFY_DIR}/src/include/nccl_device" "${PROJECT_BINARY_DIR}/include/nccl_device"
  DEPENDS ${NCCL_DEVICE_HIP_FILES}
  VERBATIM
)
add_custom_target(copy_nccl_device_headers DEPENDS ${NCCL_DEVICE_HEADER})

# Adding custom target to hipify all the source files
# This is required to make sure that all the hipified source files are
# available before compiling the unit tests executable(s)
add_custom_target(hipify_all DEPENDS ${HIP_SOURCES})

# Serialize copy after full hipify: prevents previous race condition where copying of nccl device headers
# could proceed before hipify_all stage had completed.
add_dependencies(copy_nccl_device_headers hipify_all)

# Add the generated net_ib_rocm.cc file (created by rocmIb.cmake in the staging area)
# This file is patched and transformed directly into HIPIFY_DIR, bypassing the hipify step
list(APPEND HIP_SOURCES "${HIPIFY_DIR}/src/transport/net_ib_rocm.cc")

# Collect device/host tables + collective functions generated by the device/ leaf
#==================================================================================================
# The RCCL device generators run inside src/device/CMakeLists.txt (at configure time, when that
# subdirectory was added above) and emit into ${GEN_DIR} / ${GEN_SYM_DIR}. Pick up the results
# here and fold them into the HIP source list.

# Find the generated files in the output directory
file(GLOB_RECURSE GENERATED_FILES "${GEN_DIR}/*")

# Append all found generated files to the list
foreach(file ${GENERATED_FILES})
  list(APPEND HIP_SOURCES ${file})
endforeach()

# Specialized kernels are only used by the device linker pipeline;
# always exclude them from the normal HIP compilation to avoid
# duplicate symbols with the aggregate per-collective .cpp files.
list(FILTER HIP_SOURCES EXCLUDE REGEX "gensrc/specialized/")

# Device Linker: filter out device sources that are built by the pipeline
#==================================================================================================
if(ENABLE_DEVICE_LINKER)
  # Remove per-collective .cpp files from gensrc/ and any subdirs (replaced by specialized/symmetric pipelines)
  list(FILTER HIP_SOURCES EXCLUDE REGEX "gensrc/.*\\.cpp$")
  # Add back host-only generated files
  list(APPEND HIP_SOURCES "${GEN_DIR}/host_table.cpp")
  if(EXISTS "${GEN_DIR}/device_table.cpp")
    list(APPEND HIP_SOURCES "${GEN_DIR}/device_table.cpp")
  endif()
  # Remove files compiled separately by the device pipeline
  list(FILTER HIP_SOURCES EXCLUDE REGEX "common\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "onerank\\.cu\\.cpp$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "collectives\\.cc$")
  list(FILTER HIP_SOURCES EXCLUDE REGEX "dda_all_reduce_ipc\\.cu\\.cpp$")
  message(STATUS "Device Linker: filtered HIP_SOURCES (removed device .cpp under gensrc/ + subdirs, common.cu, onerank.cu, collectives.cc, dda_all_reduce_ipc.cu.cpp)")
endif()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
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

list(APPEND HIP_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)

# Set up RCCL library
#==================================================================================================
## Set RCCL source files
add_library(rccl ${HIP_SOURCES})

## Set RCCL dependencies
## Ensure git version is updated before building rccl
add_dependencies(rccl update_git_version copy_nccl_device_headers)

## Set RCCL include directories
target_include_directories(rccl PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src)                    # for hipified headers
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device/network/unpack)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/mlx5)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/ionic)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/plugin)
target_include_directories(rccl PRIVATE ${NET_IB_CAST_INCLUDE_DIRS})
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/gensrc)
target_include_directories(rccl PRIVATE ${HSA_INCLUDE_PATH})
target_include_directories(rccl PRIVATE ${SMI_INCLUDE_DIR})
target_include_directories(rccl PRIVATE ${ROCMCORE_PATH}/include)
if(DEMANGLE_DIR)
  target_include_directories(rccl PRIVATE ${DEMANGLE_DIR})
endif()
if(ROCTX_ENABLE)
  target_include_directories(rccl PRIVATE ${ROCTRACER_INCLUDE_DIR})
endif()

# Configure the rccl target (RCCL-only: compile definitions, compile options,
# link libraries, link options, NPKit, rocprofiler-register, soversion)
#==================================================================================================
# This entire block has no NCCL CMake counterpart, so it is factored into
# cmake/RcclTargetConfig.cmake (which pulls in cmake/Npkit.cmake) to keep this
# file's NCCL-aligned source-collection and target-creation regions small.
include(${PROJECT_SOURCE_DIR}/cmake/RcclTargetConfig.cmake)
rccl_configure_target(rccl)

# Device Linker: include pipeline and wire up dependencies
#==================================================================================================
# Kept at directory scope (NOT inside rccl_configure_target): DeviceLinker.cmake
# calls enable_language(RCCLDEV), which CMake forbids inside a function()/block().
# It runs after rccl_configure_target() so DeviceLinker.cmake can read the rccl
# target's fully-populated COMPILE_DEFINITIONS / INCLUDE_DIRECTORIES directly.
if(ENABLE_DEVICE_LINKER)
  include(${PROJECT_SOURCE_DIR}/cmake/DeviceLinker.cmake)
  add_dependencies(rccl device_linker_build)
  target_link_libraries(rccl PRIVATE ${DEVICE_LINKER_OBJECTS})
endif()

# >>> RCCL-DELTA: static-library archive override (no NCCL CMake counterpart)
# NCCL emits its static library with add_library(nccl_static STATIC ...). RCCL's
# device code is built as relocatable device code (-fgpu-rdc), so producing a
# static librccl.a requires re-invoking amdclang++ with --hip-link / -fgpu-rdc /
# --emit-static-lib to trigger GPU codegen. This overrides CMAKE_CXX_ARCHIVE_CREATE
# in directory scope; it must NOT move into a function() (PARENT_SCOPE semantics).
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
# <<< RCCL-DELTA: static-library archive override

###################### Create a ras binary executable ############################
# NCCL builds this section as the `ncclras` executable from ras/client.cc. RCCL's
# delta: the target is `rcclras`, sources/includes come from the hipify staging
# tree, and it links HIP runtime libraries instead of CUDA. Keep this section
# aligned with NCCL's RAS section so upstream changes land in the same place.
# >>> RCCL-DELTA: rcclras (HIP) in place of ncclras (CUDA)
message("Building rccl RAS client executable")

add_executable(rcclras "${HIPIFY_DIR}/src/ras/client.cc")

target_include_directories(rcclras PRIVATE ${PROJECT_BINARY_DIR}/include)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src)
target_include_directories(rcclras PRIVATE ${HIPIFY_DIR}/src/include)

target_link_libraries(rcclras PRIVATE hip::host)
target_link_libraries(rcclras PRIVATE dl)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl hip::device)
else()
  add_dependencies(rcclras rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_CURRENT_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()
# <<< RCCL-DELTA: rcclras (HIP) in place of ncclras (CUDA)

# Export HIPIFY_DIR to parent scope for use in install rules
set(HIPIFY_DIR ${HIPIFY_DIR} PARENT_SCOPE)
