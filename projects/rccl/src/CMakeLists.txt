## RCCL library construction (moved from the root `CMakeLists.txt` as part of
## the NCCL CMake conformance work).
##
## NOTE: This file intentionally keeps RCCL's current behavior (hipify flow,
## generated sources, options/flags, etc.). It's a refactor-only move.

# Collect list of all source files
#==================================================================================================
# E.g: find src -type f \( -name "*.cc" -o -name "*.h" -o -name "*.hpp" \) | sort
set(SRC_FILES
  src/allocator.cc
  src/bootstrap.cc
  src/ce_coll.cc
  src/channel.cc
  src/collectives.cc
  src/commDump.cc
  src/debug.cc
  src/dev_runtime.cc
  src/enqueue.cc
  src/group.cc
  src/init.cc
  src/init_nvtx.cc
  src/mnnvl.cc
  src/msccl.cc
  src/proxy.cc
  src/rccl_wrap.cc
  src/sym_kernels.cc
  src/transport.cc
  src/device/all_gather.h
  src/device/all_reduce.h
  src/device/alltoall_pivot.h
  src/device/alltoall_gda.h
  src/device/broadcast.h
  src/device/common.h
  src/device/common_kernel.h
  src/device/op128.h
  src/device/primitives.h
  src/device/prims_ll128.h
  src/device/prims_ll.h
  src/device/prims_simple.h
  src/device/reduce.h
  src/device/reduce_kernel.h
  src/device/reduce_scatter.h
  src/device/rccl_metadata.h
  src/device/rccl_ptr.h
  src/device/sendrecv.h
  src/device/common.cu
  src/device/onerank.cu
  src/device/network/unpack/unpack_defs.h
  src/device/network/unpack/unpack.h
  src/device/symmetric/all_gather.cuh
  src/device/symmetric/all_reduce.cuh
  src/device/symmetric/kernel.cuh
  src/device/symmetric/primitives.cuh
  src/device/symmetric/reduce_scatter.cuh
  src/graph/connect.cc
  src/graph/paths.cc
  src/graph/rings.cc
  src/graph/rings.h
  src/graph/rome_models.cc
  src/graph/rome_models.h
  src/graph/search.cc
  src/graph/topo.cc
  src/graph/topo.h
  src/graph/trees.cc
  src/graph/tuning.cc
  src/graph/xml.cc
  src/graph/xml.h
  src/include/alloc.h
  src/include/allocator.h
  src/include/alt_rsmi.h
  src/include/archinfo.h
  src/include/api_trace.h
  src/include/argcheck.h
  src/include/BfdBacktrace.hpp
  src/include/bitops.h
  src/include/bootstrap.h
  src/include/ce_coll.h
  src/include/channel.h
  src/include/checks.h
  src/include/collectives.h
  src/include/coll_net.h
  src/include/comm.h
  src/include/core.h
  src/include/cpuset.h
# src/include/cudawrap.h
  src/include/debug.h
  src/include/dev_runtime.h
  src/include/device.h
  src/include/enqueue.h
  src/include/gdrwrap.h
  src/include/git_version.h
  src/include/graph.h
  src/include/group.h
  src/include/hip_rocm_version_info.h
  src/include/ibvcore.h
  src/include/ibvsymbols.h
  src/include/ibvwrap.h
  src/include/info.h
  src/include/ipcsocket.h
  src/include/mnnvl.h
  src/include/nccl_common.h
  src/include/nccl_device.h
  src/include/net_device.h
  src/include/net.h
  src/include/nvmlwrap.h
  src/include/nvtx.h
  src/include/nvtx_payload_schemas.h
  src/include/nvtx_stub.h
  src/include/p2p.h
  src/include/param.h
  src/include/profiler.h
  src/include/proxy.h
  src/include/ras.h
  src/include/rccl_common.h
  src/include/rccl_vars.h
  src/include/register.h
  src/include/register_inline.h
  src/include/rccl_float8.h
  src/include/rocmwrap.h
  src/include/roctx.h
  src/include/recorder.h
  src/include/scheduler.h
  src/include/shm.h
  src/include/shmutils.h
  src/include/signals.h
  src/include/socket.h
  src/include/strongstream.h
  src/include/sym_kernels.h
  src/include/timer.h
  src/include/transport.h
  src/include/trees.h
  src/include/tuner.h
  src/include/utils.h
  src/include/mlx5/mlx5dvcore.h
  src/include/mlx5/mlx5dvsymbols.h
  src/include/mlx5/mlx5dvwrap.h
  src/include/ionic/ionicdvcore.h
  src/include/ionic/ionicdvsymbols.h
  src/include/ionic/ionicdvwrap.h
  src/include/msccl/msccl_lifecycle.h
  src/include/msccl/msccl_parser.h
  src/include/msccl/msccl_scheduler.h
  src/include/msccl/msccl_setup.h
  src/include/msccl/msccl_status.h
  src/include/msccl/msccl_struct.h
  src/include/nccl_device/comm.h
  src/include/nccl_device/coop.h
  src/include/nccl_device/core.h
  src/include/nccl_device/ll_a2a.h
  src/include/nccl_device/mem_barrier.h
  src/include/nccl_device/ptr.h
  src/include/nccl_device/utility.h
  src/include/nccl_device/impl/comm__funcs.h
  src/include/nccl_device/impl/comm__types.h
  src/include/nccl_device/impl/core__funcs.h
  src/include/nccl_device/impl/core__types.h
  src/include/nccl_device/impl/ll_a2a__funcs.h
  src/include/nccl_device/impl/ll_a2a__types.h
  src/include/nccl_device/impl/mem_barrier__funcs.h
  src/include/nccl_device/impl/mem_barrier__types.h
  src/include/nccl_device/impl/ptr__funcs.h
  src/include/nccl_device/impl/ptr__types.h
  src/include/npkit/npkit.h
  src/include/npkit/npkit_event.h
  src/include/npkit/npkit_struct.h
  src/include/nvtx3/nvToolsExt.h
  src/include/nvtx3/nvToolsExtCounters.h
  src/include/nvtx3/nvToolsExtCuda.h
  src/include/nvtx3/nvToolsExtCudaRt.h
  src/include/nvtx3/nvToolsExtMem.h
  src/include/nvtx3/nvToolsExtMemCudaRt.h
  src/include/nvtx3/nvToolsExtOpenCL.h
  src/include/nvtx3/nvToolsExtPayload.h
  src/include/nvtx3/nvToolsExtPayloadHelper.h
  src/include/nvtx3/nvToolsExtSemanticsCounters.h
  src/include/nvtx3/nvToolsExtSemanticsScope.h
  src/include/nvtx3/nvToolsExtSync.h
  src/include/nvtx3/nvtx3.hpp
  src/include/nvtx3/nvtxDetail/nvtxExtHelperMacros.h
  src/include/nvtx3/nvtxDetail/nvtxExtImpl.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplCounters_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplMem_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplMemCudaRt_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtImplPayload_v1.h
  src/include/nvtx3/nvtxDetail/nvtxExtInit.h
  src/include/nvtx3/nvtxDetail/nvtxExtPayloadHelperInternal.h
  src/include/nvtx3/nvtxDetail/nvtxExtPayloadTypeInfo.h
  src/include/nvtx3/nvtxDetail/nvtxExtTypes.h
  src/include/nvtx3/nvtxDetail/nvtxImpl.h
  src/include/nvtx3/nvtxDetail/nvtxImplCore.h
  src/include/nvtx3/nvtxDetail/nvtxImplCuda_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplCudaRt_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplOpenCL_v3.h
  src/include/nvtx3/nvtxDetail/nvtxImplSync_v3.h
  src/include/nvtx3/nvtxDetail/nvtxInit.h
  src/include/nvtx3/nvtxDetail/nvtxInitDecls.h
  src/include/nvtx3/nvtxDetail/nvtxInitDefs.h
  src/include/nvtx3/nvtxDetail/nvtxLinkOnce.h
  src/include/nvtx3/nvtxDetail/nvtxTypes.h
  src/include/proxy_trace/proxy_trace.h
  src/include/plugin/nccl_net.h
  src/include/plugin/nccl_profiler.h
  src/include/plugin/nccl_tuner.h
  src/include/plugin/plugin.h
  src/include/plugin/net/net_v6.h
  src/include/plugin/net/net_v7.h
  src/include/plugin/net/net_v8.h
  src/include/plugin/net/net_v9.h
  src/include/plugin/net/net_v10.h
  src/include/plugin/net/net_v11.h
  src/include/plugin/profiler/net_ib_v1.h
  src/include/plugin/profiler/net_ib.h
  src/include/plugin/profiler/net_socket_v1.h
  src/include/plugin/profiler/net_socket.h
  src/include/plugin/profiler/profiler_v1.h
  src/include/plugin/profiler/profiler_v2.h
  src/include/plugin/profiler/profiler_v3.h
  src/include/plugin/profiler/profiler_v4.h
  src/include/plugin/profiler/profiler_v5.h
  src/include/plugin/tuner/tuner_v2.h
  src/include/plugin/tuner/tuner_v3.h
  src/include/plugin/tuner/tuner_v4.h
  src/include/plugin/tuner/tuner_v5.h
  src/misc/alt_rsmi.cc
  src/misc/archinfo.cc
  src/misc/argcheck.cc
  src/misc/api_trace.c
  src/misc/api_trace.cc
# src/misc/cudawrap.cc
# src/misc/gdrwrap.cc
  src/misc/ibvsymbols.cc
  src/misc/ibvwrap.cc
  src/misc/ipcsocket.cc
  src/misc/mlx5dvsymbols.cc
  src/misc/mlx5dvwrap.cc
  src/misc/ionicdvsymbols.cc
  src/misc/ionicdvwrap.cc
  src/misc/npkit.cc
# src/misc/nvmlwrap.cc
  src/misc/nvmlwrap_stub.cc
  src/misc/param.cc
  src/misc/rocmwrap.cc
  src/misc/roctx.cc
  src/misc/recorder.cc
  src/misc/shmutils.cc
  src/misc/signals.cc
  src/misc/socket.cc
  src/misc/strongstream.cc
  src/misc/utils.cc
  src/misc/msccl/msccl_lifecycle.cc
  src/misc/msccl/msccl_parser.cc
  src/misc/msccl/msccl_setup.cc
  src/misc/msccl/msccl_status.cc
  src/misc/proxy_trace/proxy_trace.cc
  src/nccl_device/core.cc
  src/nccl_device/ll_a2a.cc
  src/nccl_device/mem_barrier.cc
  src/plugin/net.cc
  src/plugin/plugin_open.cc
  src/plugin/profiler.cc
  src/plugin/tuner.cc
  src/plugin/net/net_v6.cc
  src/plugin/net/net_v7.cc
  src/plugin/net/net_v8.cc
  src/plugin/net/net_v9.cc
  src/plugin/net/net_v10.cc
  src/plugin/net/net_v11.cc
  src/plugin/profiler/profiler_v1.cc
  src/plugin/profiler/profiler_v2.cc
  src/plugin/profiler/profiler_v3.cc
  src/plugin/profiler/profiler_v4.cc
  src/plugin/profiler/profiler_v5.cc
  src/plugin/tuner/tuner_v2.cc
  src/plugin/tuner/tuner_v3.cc
  src/plugin/tuner/tuner_v4.cc
  src/plugin/tuner/tuner_v5.cc
  src/ras/client.cc
  src/ras/client_support.cc
  src/ras/collectives.cc
  src/ras/peers.cc
  src/ras/ras.cc
  src/ras/ras_internal.h
  src/ras/rasnet.cc
  src/register/coll_reg.cc
  src/register/register.cc
  src/register/sendrecv_reg.cc
  src/scheduler/symmetric_sched.cc
  src/transport/coll_net.cc
  src/transport/generic.cc
  src/transport/net.cc
  src/transport/net_ib.cc
  # net_ib_rocm.cc is generated by rocmIb.cmake directly into the hipify staging area
  # so it is not listed here - it's added to HIP_SOURCES separately below
  src/transport/net_socket.cc
  src/transport/nvls.cc
  src/transport/p2p.cc
  src/transport/profiler.cc
  src/transport/shm.cc
  src/include/latency_profiler/CollTrace.h
  src/include/latency_profiler/CollTraceEvent.h
  src/include/latency_profiler/CollTraceFunc.h
  src/include/latency_profiler/CollTraceUtils.h
  src/include/latency_profiler/EventQueue.h
  src/misc/latency_profiler/CollTrace.cc
  src/misc/latency_profiler/CollTraceEvent.cc
  src/misc/latency_profiler/CollTraceFunc.cc
  src/misc/latency_profiler/CollTraceUtils.cc
)

if(USE_AMDSMI)
  set(SMI_SOURCES
    src/include/amdsmi_wrap.h
    src/misc/amdsmi_wrap.cc
  )
else()
  set(SMI_SOURCES
    src/include/rocm_smi_wrap.h
    src/misc/rocm_smi_wrap.cc
  )
endif()
list(APPEND SRC_FILES ${SMI_SOURCES})

if (ENABLE_MSCCL_KERNEL)
  set(MSCCL_KERNEL_SOURCES
    src/device/msccl_kernel_impl.h
    src/include/msccl/msccl_kernel.h
  )
  list(APPEND SRC_FILES ${MSCCL_KERNEL_SOURCES})
endif()

if (ENABLE_MSCCLPP)
  set(MSCCLPP_SOURCES
    src/include/mscclpp/mscclpp_nccl.h
    src/misc/mscclpp/mscclpp_nccl.cc
  )
  list(APPEND SRC_FILES ${MSCCLPP_SOURCES})
endif()

# Hipify source files (copy of source generated into hipify directory)
#==================================================================================================
find_program(hipify-perl_executable hipify-perl)
if(NOT hipify-perl_executable)
  message(FATAL_ERROR "hipify-perl not found")
endif()

# Keep build output layout identical to the historical root-CMake build.
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")

## Loop over each source file to hipify
foreach(SRC_FILE ${SRC_FILES})
  # Check that file exists
  if (NOT EXISTS ${PROJECT_SOURCE_DIR}/${SRC_FILE})
    message(FATAL_ERROR "Unable to find file listed in CMakeLists.txt: ${PROJECT_SOURCE_DIR}/${SRC_FILE}")
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

  # Create a custom command to create hipified source code
  if (FAULT_INJECTION)
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${PROJECT_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${PROJECT_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${PROJECT_SOURCE_DIR}/cmake/scripts/add_faults.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  else()
    add_custom_command(
      OUTPUT ${HIP_FILE}
      COMMAND mkdir -p ${HIP_FILE_DIR}
              && ${hipify-perl_executable} -quiet-warnings ${PROJECT_SOURCE_DIR}/${SRC_FILE} -o ${HIP_FILE}
              && ${CMAKE_COMMAND} -E env bash ${PROJECT_SOURCE_DIR}/cmake/scripts/add_unroll.sh ${HIP_FILE}
      MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/${SRC_FILE}
      COMMENT "Hipifying ${SRC_FILE} -> ${HIP_FILE}"
    )
  endif()
endforeach()

# Adding custom target to hipify all the source files
# This is required to make sure that all the hipified source files are
# available before compiling the unit tests executable(s)
add_custom_target(hipify_all DEPENDS ${HIP_SOURCES})

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
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/device/generate.py ${GEN_DIR} ${IFC_ENABLED} ${COLLTRACE} ${ENABLE_MSCCL_KERNEL} ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if (gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${PROJECT_SOURCE_DIR}/src/device/generate.py failed")
endif()

if (GENERATE_SYM_KERNELS)
  # Execute the python script to generate required symmetric memory kernels
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/device/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if (gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${PROJECT_SOURCE_DIR}/src/device/symmetric/generate.py failed")
  endif()
endif()

# Find the generated files in the output directory
file(GLOB_RECURSE GENERATED_FILES "${GEN_DIR}/*")

# Append all found generated files to the list
foreach(file ${GENERATED_FILES})
  list(APPEND HIP_SOURCES ${file})
endforeach()

# Create an initial git_version.cpp file (that will be updated with latest git version)
#==================================================================================================
# Create initial empty file at configure time
file(WRITE ${PROJECT_BINARY_DIR}/git_version.cpp "")

# Add a custom target that always runs at build time to update git version
add_custom_target(update_git_version
  ALL
  COMMAND ${CMAKE_COMMAND} -DRCCL_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DRCCL_BINARY_DIR=${PROJECT_BINARY_DIR} -P ${PROJECT_SOURCE_DIR}/cmake/scripts/git_version.cmake
  BYPRODUCTS ${PROJECT_BINARY_DIR}/git_version.cpp
  COMMENT "Updating git version information"
  VERBATIM
)

list(APPEND HIP_SOURCES ${PROJECT_BINARY_DIR}/git_version.cpp)

# Set up RCCL library
#==================================================================================================
## Set RCCL source files
add_library(rccl ${HIP_SOURCES})

# Keep librccl outputs in the top-level build dir 
# existing packaging/tests scripts that link via ${PROJECT_BINARY_DIR}.
set_target_properties(rccl PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
  target_compile_definitions(rccl PRIVATE HIP_UNCACHED_MEMORY)
else()
  # Preserve legacy fallback behavior for older HIP/compiler stacks.
  if(NOT IFC_ENABLED)
    target_compile_options(rccl PRIVATE --hipcc-func-supp)
  endif()
endif()

if(HIP_HOST_UNCACHED_MEMORY)
  target_compile_definitions(rccl PRIVATE HIP_HOST_UNCACHED_MEMORY)
endif()

if(GENERATE_SYM_KERNELS)
  target_compile_definitions(rccl PRIVATE GENERATE_SYM_KERNELS)
endif()

## Set RCCL dependencies
## Ensure git version is updated before building rccl
add_dependencies(rccl update_git_version)

## Set RCCL include directories
target_include_directories(rccl PRIVATE ${PROJECT_BINARY_DIR}/include)        # for generated rccl.h header
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src)                    # for hipfied headers
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/device/network/unpack)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/mlx5)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/nccl_device)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/ionic)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/src/include/plugin)
target_include_directories(rccl PRIVATE ${HIPIFY_DIR}/gensrc)
target_include_directories(rccl PRIVATE ${HSA_INCLUDE_PATH})
target_include_directories(rccl PRIVATE ${ROCM_SMI_INCLUDE_DIR})
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

if(USE_AMDSMI)
  target_compile_definitions(rccl PRIVATE USE_AMDSMI)
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
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_GENERIC_OP_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_REDUCE_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_REDUCE_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_SEND_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_SEND_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RUN_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RUN_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_RECV_REDUCE_COPY_EXIT)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_INIT_ENTRY)
  target_compile_definitions(rccl PRIVATE ENABLE_NPKIT_EVENT_MSCCL_INIT_EXIT)
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
if(${HIP_CONTIGUOUS_MEMORY})
  target_compile_definitions(rccl PRIVATE HIP_CONTIGUOUS_MEMORY)
  message(STATUS "HIP_CONTIGUOUS_MEMORY enabled")
else()
  message(STATUS "HIP_CONTIGUOUS_MEMORY disabled")
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
target_compile_options(rccl PRIVATE -fgpu-rdc)

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
    -fvisibility=default -Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping)

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

if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(rccl PRIVATE -fsanitize=address -shared-libasan)
endif()
if(TIMETRACE)
  target_compile_options(rccl PRIVATE -ftime-trace)
endif()
if (FAULT_INJECTION)
  target_compile_definitions(rccl PRIVATE ENABLE_FAULT_INJECTION)
  message(STATUS "Fault injection enabled")
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
target_link_libraries(rccl PRIVATE   -fgpu-rdc)             # Required when linking relocatable device code
target_link_libraries(rccl PRIVATE   Threads::Threads)
target_link_libraries(rccl INTERFACE hip::host)
target_link_libraries(rccl PRIVATE   hip::device)
target_link_libraries(rccl PRIVATE   dl)
target_link_libraries(rccl PRIVATE   ${SMI_LIBRARIES})
target_link_libraries(rccl PRIVATE fmt::fmt-header-only)
if(ENABLE_MSCCLPP)
  add_mscclpp_targets()
  target_link_libraries(rccl PRIVATE mscclpp_nccl)
endif()
if(ENABLE_ROCSHMEM)
  target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
  target_link_libraries(rccl PRIVATE ${IBVERBS})
endif()

## Set RCCL link options
## Find out available memory
execute_process(
  COMMAND bash "-c" "cat /sys/fs/cgroup/memory.max"
  OUTPUT_VARIABLE memory_max_string)
if (${memory_max_string} MATCHES "^[0-9]+")
  math(EXPR memory_in_gb "${memory_max_string} / (1024 * 1024 * 1024)")
else()
  execute_process(
    COMMAND bash "-c" "free | grep -o '[[:digit:]]*' | head -1"
    OUTPUT_VARIABLE memory_max_string)
  ## memory_max_string holds the free memory in KB
  if (${memory_max_string} MATCHES "^[0-9]+")
    math(EXPR memory_in_gb "${memory_max_string} / (1024 * 1024)") ## KB to GB conversion
  else()
    cmake_host_system_information(RESULT memory_max_string QUERY AVAILABLE_PHYSICAL_MEMORY )
    math(EXPR memory_in_gb "${memory_max_string} / 1024")
  endif()
endif()
## Reserve 16GB for each linker job. Limit max number of linker jobs to 16
if (HAVE_PARALLEL_JOBS)
  math(EXPR num_linker_jobs "(${memory_in_gb} + 15) / 16")
  if (${num_linker_jobs} GREATER_EQUAL "16")
    set(num_linker_jobs "16")
  endif()
  message(STATUS "Use ${num_linker_jobs} jobs for linking")
  target_link_options(rccl PRIVATE -parallel-jobs=${num_linker_jobs})       # Use multiple threads to link
endif()
if(BUILD_ADDRESS_SANITIZER)
  target_link_options(rccl PRIVATE -fuse-ld=lld)
endif()
if(TIMETRACE)
  target_link_options(rccl PRIVATE -ftime-trace)
endif()

if(NOT BUILD_SHARED_LIBS)
  message(STATUS "Building static RCCL library")
else()
  message(STATUS "Building shared RCCL library")
endif()
if (HAVE_KERNARG_PRELOAD)
  target_link_options(rccl PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
endif()

## Track linking time
set_property(TARGET rccl PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")
