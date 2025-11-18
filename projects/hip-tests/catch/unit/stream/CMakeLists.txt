set(COMMON_SHARED_SRC streamCommon.cc)

set(TEST_SRC
    hipStreamCreate.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipMultiStream.cc
    hipStreamAddCallback.cc
    hipStreamCreateWithFlags.cc
    hipStreamCreateWithPriority.cc
    hipStreamDestroy.cc
    hipAPIStreamDisable.cc
    hipStreamLegacy.cc
    hipStreamValue.cc
    hipStreamSynchronize.cc
    hipStreamQuery.cc
    hipDeviceGetStreamPriorityRange.cc
    hipStreamACb_StrmSyncTiming.cc
    hipLaunchHostFunc.cc
    hipStreamGetDevice.cc
    hipStreamLegacy_Ext.cc
    hipStreamLegacy_compiler_options.cc
    hipStreamGetId.cc
    hipStreamSetGetAttributes.cc
    hipStreamCopyAttributes.cc)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} hipStreamGetCUMask.cc hipStreamWithCUMask.cc
               hipStreamACb_MultiThread.cc hipStreamWaitEvent.cc
	       hipStreamGetFlags_spt.cc
               hipStreamGetPriority_spt.cc
               hipStreamQuery_spt.cc
               hipStreamSynchronize_spt.cc
               hipStreamBatchMemOp.cc)
else()
  set(TEST_SRC
      ${TEST_SRC}
      # hipStreamAttachMemAsync_old.cc   # Disabling it on nvidia due to issue
      # in function definition of hipStreamAttachMemAsync
      # Fixing would break ABI, to be re-enabled when the fix is made.
      hipStreamACb_MultiThread.cc)

  # set_source_files_properties(hipStreamAttachMemAsync_old.cc PROPERTIES
  # COMPILE_FLAGS -std=c++17)
endif()

if(HIP_PLATFORM MATCHES "amd")
        set_source_files_properties(hipStreamLegacy_compiler_options.cc PROPERTIES COMPILE_OPTIONS "-fgpu-default-stream=per-thread")
endif()
if(HIP_PLATFORM MATCHES "nvidia")
        set_source_files_properties(hipStreamLegacy_compiler_options.cc PROPERTIES COMPILE_OPTIONS "--default-stream=per-thread")
endif()

hip_add_exe_to_target(
  NAME StreamTest
  TEST_SRC ${TEST_SRC} TEST_TARGET_NAME build_tests
  COMPILE_OPTIONS -std=c++17 COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

add_executable(hipStreamLegacy_exe EXCLUDE_FROM_ALL hipStreamLegacy_exe.cc)
add_dependencies(StreamTest hipStreamLegacy_exe)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipStreamLegacy_exe)
