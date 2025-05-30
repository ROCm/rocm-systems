# Common Tests - Test independent of all platforms
set(TEST_SRC
  warp_ballot.cc
  warp_any.cc
  warp_all.cc
  hipMatchSyncAllTests.cc
  hipMatchSyncAnyTests.cc
  hipShflSyncDownTests.cc
  hipShflSyncUpTests.cc
  hipShflSyncXorTests.cc
  hipShflSyncTests.cc
  hipVoteSyncTests.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC}
    warp_shfl.cc
    warp_shfl_xor.cc
    warp_shfl_up.cc
    warp_shfl_down.cc
    warp_reduce.cc
    hipShflUpDownTest.cc
    hipShflTests.cc
  )
endif()

 if(HIP_PLATFORM MATCHES "nvidia")
  set_source_files_properties(hipMatchSyncAllTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipMatchSyncAnyTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipShflSyncDownTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipShflSyncUpTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipShflSyncXorTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipShflSyncTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
  set_source_files_properties(hipVoteSyncTests.cc PROPERTIES COMPILE_FLAGS "-arch=sm_70")
 endif()

hip_add_exe_to_target(NAME WarpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
