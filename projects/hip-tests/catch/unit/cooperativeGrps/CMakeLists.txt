# Common Tests - Test independent of all platforms
set(TEST_SRC
  thread_block.cc
  thread_block_tile.cc
  coalesced_group_tiled_partition.cc
  hipCGThreadBlockType_old.cc
  hipCGGridGroupType_old.cc
  hipCGTiledPartitionType_old.cc
  hipCGThreadBlockTileTypeShfl_old.cc
  hipCGCoalescedGroups_old.cc
  hipLaunchCooperativeKernel_old.cc
  coalesced_groups_shfl_down_old.cc
  coalesced_groups_shfl_up_old.cc
  coalesced_group.cc
  grid_group.cc
  coalesced_tiled_groups_metagrp.cc
  binary_partition.cc
  cg_ballot.cc
  cg_any_all.cc
  split_barrier.cc
)

set(AMD_TEST_SRC 
  multi_grid_group.cc
  hipCGMultiGridGroupType_old.cc
  hipLaunchCooperativeKernelMultiDevice_old.cc
)

if(HIP_PLATFORM STREQUAL "nvidia")
  set(COMPUTE_ARCH "-arch=compute_75")
  set(EXPERIMENTAL_ABI "-D_CG_ABI_EXPERIMENTAL")
  set(GENCODE_PARAMS "-rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=compute_75 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(hipCGMultiGridGroupType_old.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI} ${GENCODE_PARAMS}")
  set_source_files_properties(hipLaunchCooperativeKernelMultiDevice_old.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI} ${GENCODE_PARAMS}")
  set_source_files_properties(multi_grid_group.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI} ${GENCODE_PARAMS}")
  set_source_files_properties(hipCGMultiGridGroupType_old.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI}")
  set_source_files_properties(hipLaunchCooperativeKernelMultiDevice_old.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI}")
  set_source_files_properties(multi_grid_group.cc PROPERTIES COMPILE_FLAGS "${EXPERIMENTAL_ABI}")
  set_source_files_properties(cg_ballot.cc PROPERTIES COMPILE_FLAGS "${COMPUTE_ARCH}")
  set_source_files_properties(cg_any_all.cc PROPERTIES COMPILE_FLAGS "${COMPUTE_ARCH}")
  set_source_files_properties(binary_partition.cc PROPERTIES COMPILE_FLAGS "${COMPUTE_ARCH}")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(coopGrpTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  set(${TEST_SRC} ${TEST_SRC} ${AMD_TEST_SRC})
  set_source_files_properties(cg_ballot.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  set_source_files_properties(cg_any_all.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  set_source_files_properties(binary_partition.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
endif()
