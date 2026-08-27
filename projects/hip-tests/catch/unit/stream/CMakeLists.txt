# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
    hipSharedQueueAnyOrderOverlap.cc
    hipStreamAttachMemAsync.cc
    # TODO renable it TheRock is not finding it
    hipStreamCopyAttributes.cc
  )

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} hipStreamGetCUMask.cc hipStreamWithCUMask.cc
               hipStreamACb_MultiThread.cc hipStreamWaitEvent.cc
	       hipStreamGetFlags_spt.cc
               hipStreamGetPriority_spt.cc
               hipStreamQuery_spt.cc
               hipStreamSynchronize_spt.cc
               hipStreamBatchMemOp.cc
               hipStreamValueCapture.cc
               hipStreamPriorityAqlLog.cc
               # disabled on NVIDIA due to lack of hipStreamGetCaptureInfo_v2 in cuda 13.0+
                hipStreamLegacy.cc

             )
  if(WIN32)
    set(TEST_SRC ${TEST_SRC} hipStreamKilledWorkerThread.cc)
  endif()
else()
  set(TEST_SRC
      ${TEST_SRC}
      hipStreamACb_MultiThread.cc)
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
  COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

add_executable(hipStreamLegacy_exe EXCLUDE_FROM_ALL hipStreamLegacy_exe.cc)
set_source_files_properties(hipStreamLegacy_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})

add_dependencies(StreamTest hipStreamLegacy_exe)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipStreamLegacy_exe)

if(WIN32)
  set(TEST_SRC hipGracefulExit.cc)
  hip_add_exe_to_target(
    NAME GracefulExit
    TEST_SRC ${TEST_SRC}
    TEST_TARGET_NAME build_tests)

  add_executable(hipGracefulExit_exe EXCLUDE_FROM_ALL hipGracefulExit_exe.cc)
  set_source_files_properties(hipGracefulExit_exe.cc PROPERTIES LANGUAGE HIP)
  set_target_properties(hipGracefulExit_exe PROPERTIES LINKER_LANGUAGE HIP)
  target_link_libraries(hipGracefulExit_exe hip::host hip::device)

  add_dependencies(GracefulExit hipGracefulExit_exe)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipGracefulExit_exe)
endif()

if(WIN32 AND HIP_PLATFORM MATCHES "amd")
  add_executable(hipStreamKilledWorkerThread_exe EXCLUDE_FROM_ALL
                 hipStreamKilledWorkerThread_exe.cc)
  set_source_files_properties(hipStreamKilledWorkerThread_exe.cc PROPERTIES
                              LANGUAGE ${GPGPU_LANGUAGE})
  target_link_libraries(hipStreamKilledWorkerThread_exe ${GPGPU_LINKER_LIBRARIES})
  add_dependencies(StreamTest hipStreamKilledWorkerThread_exe)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipStreamKilledWorkerThread_exe)
endif()
