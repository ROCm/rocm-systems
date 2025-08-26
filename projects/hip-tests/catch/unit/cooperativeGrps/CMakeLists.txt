# Common Tests - Test independent of all platforms
set(TEST_SRC
  thread_block.cc
  thread_block_tile.cc
  coalesced_group_tiled_partition.cc
  hipCGThreadBlockType_old.cc
  hipCGMultiGridGroupType_old.cc
  hipCGGridGroupType_old.cc
  hipCGTiledPartitionType_old.cc
  hipCGThreadBlockTileTypeShfl_old.cc
  hipCGCoalescedGroups_old.cc
  hipLaunchCooperativeKernel_old.cc
  hipLaunchCooperativeKernelMultiDevice_old.cc
  multi_grid_group.cc
  coalesced_groups_shfl_down_old.cc
  coalesced_groups_shfl_up_old.cc
  coalesced_group.cc
  grid_group.cc
  coalesced_tiled_groups_metagrp.cc
  binary_partition.cc
  cg_ballot.cc
  cg_any_all.cc
)

if(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(hipCGMultiGridGroupType_old.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(hipLaunchCooperativeKernelMultiDevice_old.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(multi_grid_group.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(cg_ballot.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(cg_any_all.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(binary_partition.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS "-rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80, -gencode arch=compute_86,code=sm_86, -gencode=arch=compute_86,code=compute_86")
else()
  set_source_files_properties(cg_ballot.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  set_source_files_properties(cg_any_all.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  set_source_files_properties(binary_partition.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
endif()
