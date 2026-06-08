# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#
# RCCL Source Directory CMakeLists.txt
# This file handles source collection, hipification, and library creation
#==================================================================================================

# When RCCL is included as a subproject (e.g. rocm-systems), CMAKE_SOURCE_DIR points to the
# super-repo root. Use PROJECT_SOURCE_DIR for RCCL-relative paths.
set(RCCL_SOURCE_DIR "${PROJECT_SOURCE_DIR}")

# net_ib_cast: populate NET_IB_CAST_SOURCES and NET_IB_CAST_INCLUDE_DIRS
add_subdirectory(transport/net_ib_cast)

# Collect list of all source files
#==================================================================================================
# E.g: find src -type f \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
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
  device/all_gather.h
  device/all_reduce.h
  device/alltoall_pivot.h
  device/alltoall_gda.h
  device/alltoallv_gda.h
  device/broadcast.h
  device/common.h
  device/common_kernel.h
  device/op128.h
  device/primitives.h
  device/prims_ll128.h
  device/prims_ll.h
  device/prims_simple.h
  device/reduce.h
  device/reduce_kernel.h
  device/reduce_scatter.h
  device/rccl_metadata.h
  device/sendrecv.h
  device/hierarchical_ag_shuffle.h
  device/common.cu
  device/onerank.cu
  device/network/unpack/unpack_defs.h
  device/network/unpack/unpack.h
  device/symmetric/all_gather.cuh
  device/symmetric/all_gather_gin.cuh
  device/symmetric/all_reduce.cuh
  device/symmetric/data_ops.cuh
  device/symmetric/gin_scratch.h
  device/symmetric/gin_scratch__funcs.h
  device/symmetric/gin_scratch__types.h
  device/symmetric/kernel.cuh
  device/symmetric/primitives.cuh
  device/symmetric/reduce_scatter.cuh
  device/symmetric/reduce_scatter_gin.cuh
  graph/connect.cc
  graph/paths.cc
  graph/rings.cc
  graph/rings.h
  graph/rome_models.cc
  graph/rome_models.h
  graph/rome_topo_consensus.cc
  graph/rome_topo_consensus.h
  graph/search.cc
  graph/topo.cc
  graph/topo.h
  graph/trees.cc
  graph/tuning.cc
  graph/xml.cc
  graph/xml.h
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
  misc/alt_rsmi.cc
  misc/archinfo.cc
  misc/argcheck.cc
  misc/api_trace.c
  misc/api_trace.cc
# misc/cudawrap.cc
# misc/gdrwrap.cc
  misc/ibvsymbols.cc
  misc/ibvwrap.cc
  misc/ipcsocket.cc
  misc/mlx5dvsymbols.cc
  misc/mlx5dvwrap.cc
  misc/ionicdvsymbols.cc
  misc/ionicdvwrap.cc
  misc/npkit.cc
# misc/nvmlwrap.cc
  misc/nvmlwrap_stub.cc
  misc/param.cc
  misc/rocmwrap.cc
  misc/roctx.cc
  misc/recorder.cc
  misc/shmutils.cc
  misc/signals.cc
  misc/socket.cc
  misc/strongstream.cc
  misc/utils.cc
  misc/proxy_trace/proxy_trace.cc
  nccl_device/core.cc
  nccl_device/gin_barrier.cc
  nccl_device/gin_scratch.cc
  nccl_device/ll_a2a.cc
  nccl_device/lsa_barrier.cc
  os/linux.cc
  include/os.h
  rma/rma.cc
  rma/rma_ce.cc
  rma/rma_proxy.cc
  include/rma/rma.h
  include/rma/rma_ce.h
  include/rma/rma_proxy.h
  plugin/net.cc
  plugin/env.cc
  plugin/env/env_v1.cc
  plugin/plugin_open.cc
  plugin/profiler.cc
  plugin/tuner.cc
  plugin/net/net_v6.cc
  plugin/net/net_v7.cc
  plugin/net/net_v8.cc
  plugin/net/net_v9.cc
  plugin/net/net_v10.cc
  plugin/net/net_v11.cc
  plugin/profiler/profiler_v1.cc
  plugin/profiler/profiler_v2.cc
  plugin/profiler/profiler_v3.cc
  plugin/profiler/profiler_v4.cc
  plugin/profiler/profiler_v5.cc
  plugin/profiler/profiler_v6.cc
  plugin/tuner/tuner_v2.cc
  plugin/tuner/tuner_v3.cc
  plugin/tuner/tuner_v4.cc
  plugin/tuner/tuner_v5.cc
  plugin/tuner/csv_tuner.cc
  plugin/gin.cc
  plugin/gin/gin_v11.cc
  plugin/gin/gin_v12.cc
  ras/client.cc
  ras/client_support.cc
  ras/collectives.cc
  ras/peers.cc
  ras/ras.cc
  ras/ras_internal.h
  ras/rasnet.cc
  register/coll_reg.cc
  register/register.cc
  register/sendrecv_reg.cc
  scheduler/symmetric_sched.cc
  scheduler/allgatherv_sched.cc
  transport/coll_net.cc
  transport/generic.cc
  transport/net.cc
  transport/net_ib.cc
  transport/net_ib_limits.h
  transport/net_ib_cast/net_ib_cast_inspect.h
  transport/net_ib_cast/net_ib_fault_inject.h
  transport/net_ib/gin.h
  # net_ib_rocm.cc is generated by rocmIb.cmake directly into the hipify staging area
  # so it is not listed here - it's added to HIP_SOURCES separately below
  transport/net_socket.cc
  transport/nvls.cc
  transport/p2p.cc
  transport/profiler.cc
  transport/shm.cc
  gin/gin_host.cc
  gin/gin_host_proxy.cc
  transport/net_ib/gdaki/gin_host_gdaki.h
  include/latency_profiler/CollTrace.h
  include/latency_profiler/CollTraceEvent.h
  include/latency_profiler/CollTraceFunc.h
  include/latency_profiler/CollTraceUtils.h
  include/latency_profiler/EventQueue.h
  misc/latency_profiler/CollTrace.cc
  misc/latency_profiler/CollTraceEvent.cc
  misc/latency_profiler/CollTraceFunc.cc
  misc/latency_profiler/CollTraceUtils.cc
)

# Prepend src/ to all source files for proper paths from root
list(TRANSFORM SRC_FILES PREPEND "src/")

# Conditional source files based on build options
#==================================================================================================
if(USE_AMDSMI)
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
  )
else()
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
    src/include/rocm_smi_wrap.h
    src/misc/rocm_smi_wrap.cc
  )
endif()
list(APPEND SRC_FILES ${SMI_SOURCES})
# NET_IB_CAST_SOURCES already carry the src/ prefix (set by transport/net_ib_cast/CMakeLists.txt)
list(APPEND SRC_FILES ${NET_IB_CAST_SOURCES})

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()
# Use PROJECT_BINARY_DIR to keep hipify at project root level
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
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

# Generate device/host tables and all the collective functions that are going to be in librccl.so
#==================================================================================================
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "RCCL requires Python3 for generating host/device tables")
endif()

set(GEN_DIR "${HIPIFY_DIR}/gensrc")
set(GEN_SYM_DIR "${GEN_DIR}/symmetric")

if(ONLY_FUNCS)
  message(WARNING "Using ONLY_FUNCS = ${ONLY_FUNCS}. Not meant for release builds.")
endif()

# Execute the python script to generate required collective functions
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${RCCL_SOURCE_DIR}/src/device/generate.py ${GEN_DIR} ${IFC_ENABLED} ${COLLTRACE} OFF ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ENABLE_ROCSHMEM} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if (gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${RCCL_SOURCE_DIR}/src/device/generate.py failed")
endif()

if (GENERATE_SYM_KERNELS)
  # Execute the python script to generate required symmetric memory kernels
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${RCCL_SOURCE_DIR}/src/device/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if (gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${RCCL_SOURCE_DIR}/src/device/symmetric/generate.py failed")
  endif()
endif()

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

## Set RCCL compile definitions
if(COLLTRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_COLLTRACE)
endif()
if(ENABLE_MSCCL_KERNEL)
  message(WARNING "MSCCL is deprecated and will be removed in a future version of RCCL.")
  target_compile_definitions(rccl PRIVATE COMPILE_MSCCL_KERNEL)
endif()
if(ENABLE_MSCCLPP)
  target_compile_definitions(rccl PRIVATE ENABLE_MSCCLPP)
endif()

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
if("${hip_version_string}" VERSION_GREATER_EQUAL "7.12.60540")
  target_compile_definitions(rccl PRIVATE HIP_VMM_UNCACHED_MEMORY)
endif()

# ==== rocSHMEM integration (optional) ====
if (ENABLE_ROCSHMEM)
  add_rocshmem_targets()
  # Ensure rocSHMEM is fully built/installed before compiling rccl
  if (TARGET rocshmem_ext)
    add_dependencies(rccl rocshmem_ext)
  endif()

  if (ROCSHMEM_INCLUDE_DIR)
    target_include_directories(rccl PRIVATE ${ROCSHMEM_INCLUDE_DIR})
  endif()

  # Moved to where MSCCL target_links
  ## target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
endif()

# NPKit flags
## May be better to move these to a separate file
if(ENABLE_NPKIT)
  message(WARNING "NPKit is deprecated and will be removed in a future version of RCCL. Please consider using alternative profiling tools.")
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_DIRECT_SEND_FROM_OUTPUT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_FROM_OUTPUT_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_WAIT_PEER_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_SIMPLE_REDUCE_OR_COPY_MULTI_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_WAIT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_WAIT_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_TEST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_TEST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_NET_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_UPDOWN_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_REDUCE_TREE_SPLIT_BROADCAST_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_LOCAL_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_SEND_RECV_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_REDUCE_SCATTER_RING_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_PRIM_COLLECT_DATA_PROCESS_TIME)
endif()

if(PROFILE)
  target_compile_definitions(rccl PRIVATE ENABLE_PROFILING)
endif()
if(ROCTX_ENABLE)
  target_compile_definitions(rccl PRIVATE ROCTX_ENABLE)
else()
  target_compile_definitions(rccl PRIVATE NVTX_NO_IMPL)
  target_compile_definitions(rccl PRIVATE NVTX_DISABLE)
endif()
if(TRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_TRACE)
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

## Set RCCL compile options
if (HAVE_PARALLEL_JOBS)
  target_compile_options(rccl PRIVATE -parallel-jobs=12)
endif()

if (ROCM_VERSION VERSION_GREATER_EQUAL "60200")
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
elseif(BUILD_TESTS) # Enable default/hidden visibility based on build type and ROCM_VERSION
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(rccl PRIVATE -fvisibility=default)
  else()
    target_compile_options(rccl PRIVATE -fvisibility=hidden)
  endif()
else()  # Enable hidden visibility for library without tests/code coverage enabled
  target_compile_options(rccl PRIVATE -fvisibility=hidden)
endif()

if (HAVE_KERNARG_PRELOAD)
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

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl PRIVATE -fsanitize=address)
  target_link_options(rccl PRIVATE -fsanitize=address -shared-libasan)
  if(DEFINED ASAN_RUNTIME_DIR)
    target_link_options(rccl PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
  endif()
endif()
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

if (ROCM_VERSION VERSION_GREATER_EQUAL "60100")
    option(RCCL_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
else()
    if(RCCL_ROCPROFILER_REGISTER)
        message(AUTHOR_WARNING "RCCL_ROCPROFILER_REGISTER is not valid option for ROCm < 6.2. Current ROCm version: ${ROCM_VERSION}")
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
if(ENABLE_DEVICE_LINKER)
  include(${PROJECT_SOURCE_DIR}/cmake/DeviceLinker.cmake)
  add_dependencies(rccl device_linker_build)
  target_link_libraries(rccl PRIVATE ${DEVICE_LINKER_OBJECTS})
endif()

## Set RCCL linked libraries
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
target_link_libraries(rccl PRIVATE   ${SMI_LIBRARIES})
# Wrap fmt in $<BUILD_INTERFACE:...> so it's only used while building rccl
# and is omitted from the exported rccl-targets export set. Without this,
# a FetchContent-built fmt-header-only target leaks into rccl's link
# interface and breaks install(EXPORT) for static builds.
target_link_libraries(rccl PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
if(ENABLE_ROCSHMEM)
  target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
  # librocshmem.a is compiled with -fgpu-rdc (relocatable device code).
  # The __hip_gpubin_handle_* symbols referenced by __hip_module_ctor in the
  # librocshmem.a objects are only defined by the RDC device-link step.
  # --hip-link tells amdclang++ to perform that step as part of the final
  # librccl.so link; -fgpu-rdc marks the link as RDC-mode.  --offload-arch
  # flags use RCCL's GPU_TARGETS (stripped of feature suffixes like :xnack-)
  # so the device-link covers every arch RCCL is built for.
  # RCCL's own device code (already embedded in device_build/common.o by the
  # custom device-linker pipeline) is NOT compiled with -fgpu-rdc and is
  # therefore unaffected by this device-link step.
  set(_rocshmem_offload_arch_flags "")
  foreach(_gpu_raw ${GPU_TARGETS})
    string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
    list(APPEND _rocshmem_offload_arch_flags "--offload-arch=${_gpu}")
  endforeach()
  list(REMOVE_DUPLICATES _rocshmem_offload_arch_flags)
  target_link_options(rccl PRIVATE -fgpu-rdc --hip-link ${_rocshmem_offload_arch_flags})
endif()

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

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD AND NOT ENABLE_DEVICE_LINKER)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
endif()

if(ENABLE_MSCCLPP)
  include(${RCCL_SOURCE_DIR}/cmake/MSCCLPP.cmake)
endif()

## Track linking time
set_property(TARGET rccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

## Setup librccl.so version
rocm_set_soversion(rccl "1.0")

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

# Build RCCL RAS client executable
#==================================================================================================
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

# Export HIPIFY_DIR to parent scope for use in install rules
set(HIPIFY_DIR ${HIPIFY_DIR} PARENT_SCOPE)
