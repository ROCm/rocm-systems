# ---------------------------------------------------------------------------
# src/algorithms/dda/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set_source_files_properties(
  ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_gpu_barrier.cu.cpp
  ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_mem_handler.cc
  ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
  PROPERTIES GENERATED TRUE)
set(_MICRO_DDA_DIR ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda)
# hipify renames .cu to .cu.cpp
set(_MICRO_DDA_DEFS
  DDA_ALL_GATHER_FABRIC_CC_PATH="${_MICRO_DDA_DIR}/all_gather/dda_all_gather_fabric.cu.cpp"
  DDA_ALL_GATHER_FABRIC_LL_CC_PATH="${_MICRO_DDA_DIR}/all_gather/dda_all_gather_fabric_ll.cu.cpp"
  DDA_ALL_GATHER_FABRIC_LL128_CC_PATH="${_MICRO_DDA_DIR}/all_gather/dda_all_gather_fabric_ll128.cu.cpp"
  DDA_ALL_GATHER_IPC_CC_PATH="${_MICRO_DDA_DIR}/all_gather/dda_all_gather_ipc.cu.cpp"
  DDA_ALL_REDUCE_FABRIC_CC_PATH="${_MICRO_DDA_DIR}/all_reduce/dda_all_reduce_fabric.cu.cpp"
  DDA_ALL_REDUCE_FABRIC_LL_CC_PATH="${_MICRO_DDA_DIR}/all_reduce/dda_all_reduce_fabric_ll.cu.cpp"
  DDA_ALL_REDUCE_FABRIC_LL128_CC_PATH="${_MICRO_DDA_DIR}/all_reduce/dda_all_reduce_fabric_ll128.cu.cpp"
  DDA_ALL_REDUCE_IPC_CC_PATH="${_MICRO_DDA_DIR}/all_reduce/dda_all_reduce_ipc.cu.cpp"
  DDA_ALLTOALL_FABRIC_CC_PATH="${_MICRO_DDA_DIR}/alltoall/dda_alltoall_fabric.cu.cpp"
  DDA_ALLTOALL_FABRIC_LL_CC_PATH="${_MICRO_DDA_DIR}/alltoall/dda_alltoall_fabric_ll.cu.cpp"
  DDA_ALLTOALL_FABRIC_LL128_CC_PATH="${_MICRO_DDA_DIR}/alltoall/dda_alltoall_fabric_ll128.cu.cpp"
  DDA_ALLTOALL_IPC_CC_PATH="${_MICRO_DDA_DIR}/alltoall/dda_alltoall_ipc.cu.cpp"
  DDA_BROADCAST_FABRIC_LL_CC_PATH="${_MICRO_DDA_DIR}/broadcast/dda_broadcast_fabric_ll.cu.cpp"
  DDA_IPC_GPU_BARRIER_CU_PATH="${_MICRO_DDA_DIR}/ipc/ipc_gpu_barrier.cu.cpp"
  DDA_IPC_INIT_CU_PATH="${_MICRO_DDA_DIR}/ipc/ipc_init.cu.cpp"
  DDA_IPC_MEM_HANDLER_CC_PATH="${_MICRO_DDA_DIR}/ipc/ipc_mem_handler.cc"
  DDA_REDUCE_SCATTER_FABRIC_CC_PATH="${_MICRO_DDA_DIR}/reduce_scatter/dda_reduce_scatter_fabric.cu.cpp"
  DDA_REDUCE_SCATTER_FABRIC_LL_CC_PATH="${_MICRO_DDA_DIR}/reduce_scatter/dda_reduce_scatter_fabric_ll.cu.cpp"
  DDA_REDUCE_SCATTER_FABRIC_LL128_CC_PATH="${_MICRO_DDA_DIR}/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu.cpp"
  DDA_REDUCE_SCATTER_IPC_CC_PATH="${_MICRO_DDA_DIR}/reduce_scatter/dda_reduce_scatter_ipc.cu.cpp"
  FABRIC_GPU_BARRIER_CC_PATH="${_MICRO_DDA_DIR}/fabric/fabric_gpu_barrier.cu.cpp"
  FABRIC_INIT_CC_PATH="${_MICRO_DDA_DIR}/fabric/fabric_init.cu.cpp"
  FABRIC_MEM_HANDLER_CC_PATH="${_MICRO_DDA_DIR}/fabric/fabric_mem_handler.cc"
  IPC_INIT_CC_PATH="${_MICRO_DDA_DIR}/ipc/ipc_init.cu.cpp")

# The launchers define __global__ kernels; an embedded empty GPU binary keeps clang from referencing an
# undefined __hip_fatbin_<cuid>. Registration is faked in dda-coll-seams.cc and fakes/hip_fakes.cc.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary" "")

# One src/algorithms/dda/ test file per binary; see rccl_add_unit_micro_binary(). KERNELS embeds the empty GPU binary.
macro(rccl_add_dda_micro_binary _name _test)
  cmake_parse_arguments(_radmb "KERNELS" "" "" ${ARGN})
  rccl_add_unit_micro_binary(${_name} algorithms/dda/${_test} ${_radmb_UNPARSED_ARGUMENTS}
    OBJECT_PREFIX rccl-dda-obj
    # alltoall_dda_fabric_tdm.h includes "tdm/tdmCopy.h".
    COMMON_INCLUDES ${RCCL_HOST_HIPIFY_DIR}/src/device
    COMMON_DEFS ${_MICRO_DDA_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_dda.yaml")
  if(_radmb_KERNELS)
    target_compile_options(${_name} PRIVATE
      "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${CMAKE_CURRENT_BINARY_DIR}/empty_gpubinary")
  endif()
endmacro()

# Launchers record their launches through DdaCollSeams.h, defined once in dda-coll-seams.cc.
set(_MICRO_DDA_COLL_SOURCES
  algorithms/dda/dda-coll-seams.cc
  fakes/hip_fakes.cc
  fakes/nccl_fakes.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllGatherFabric all_gather/dda-all-gather-fabric-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllGatherFabricLl all_gather/dda-all-gather-fabric-ll-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllGatherFabricLl128 all_gather/dda-all-gather-fabric-ll128-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllGatherIpc all_gather/dda-all-gather-ipc-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllReduceFabric all_reduce/dda-all-reduce-fabric-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllReduceFabricLl all_reduce/dda-all-reduce-fabric-ll-test.cc
  KERNELS
  SOURCES
    ${_MICRO_DDA_COLL_SOURCES}
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllReduceFabricLl128 all_reduce/dda-all-reduce-fabric-ll128-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAllReduceIpc all_reduce/dda-all-reduce-ipc-test.cc
  KERNELS
  SOURCES
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAlltoallFabric alltoall/dda-alltoall-fabric-test.cc
  KERNELS
  SOURCES
    ${_MICRO_DDA_COLL_SOURCES}
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAlltoallFabricLl alltoall/dda-alltoall-fabric-ll-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAlltoallFabricLl128 alltoall/dda-alltoall-fabric-ll128-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaAlltoallIpc alltoall/dda-alltoall-ipc-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaBroadcastFabricLl broadcast/dda-broadcast-fabric-ll-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaReduceScatterFabric reduce_scatter/dda-reduce-scatter-fabric-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaReduceScatterFabricLl
    reduce_scatter/dda-reduce-scatter-fabric-ll-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaReduceScatterFabricLl128
    reduce_scatter/dda-reduce-scatter-fabric-ll128-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaReduceScatterIpc reduce_scatter/dda-reduce-scatter-ipc-test.cc
  KERNELS
  SOURCES ${_MICRO_DDA_COLL_SOURCES})

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaFabricGpuBarrier fabric/fabric-gpu-barrier-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_mem_handler.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaFabricInit fabric/fabric-init-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_gpu_barrier.cu.cpp
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/fabric/fabric_mem_handler.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaFabricMemHandler fabric/fabric-mem-handler-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc)

# ipc_init.cu drives ipc_mem_handler.cc and ipc_gpu_barrier.cu; the test includes all three.
rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaIpcInit ipc/ipc-init-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/comm_fakes.cc
    fakes/env_fakes.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/rccl_wrap_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)

rccl_add_dda_micro_binary(rccl-UnitTestsMicroDdaIpcLifecycle ipc/dda-ipc-lifecycle-test.cc
  SOURCES
    fakes/bootstrap_stubs.cc
    fakes/hip_fakes.cc
    fakes/nccl_fakes.cc
    fakes/utils_fakes.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/algorithms/dda/device/device_buffer.cc
    ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc)
